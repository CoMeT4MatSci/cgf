import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import Trajectory
from cgf.utils.training import get_optimal_coeffs
from numpy.polynomial.polynomial import polyfit

parser = argparse.ArgumentParser(description="A to run training and optionally run a CG calculation with an SW defect.")
parser.add_argument('-p', required=True, help='path to folder')
parser.add_argument('-sw', action='store_true', help='Do a CG SW defect calculation if -sw')
parser.add_argument('-r0', type=float, help='Specify r0 param. Otherwise based on unit cell parameter')
args = parser.parse_args()


path = Path(args.p)

traj = Trajectory(Path(path,'traj_training.traj'))
if args.r0 is not None:
    r0 = args.r0
else:
    r0 = traj[0].cell.cellpar()[0]/np.sqrt(3)  # works for hexagonal unit cells
structures = []
energies = []
for n, atoms in enumerate(traj):
    del atoms.constraints
    energies.append(atoms.get_potential_energy())
    structures.append(atoms)
    if 1.2*r0>=atoms.cell.cellpar()[2]:
        cellnew = atoms.cell

        cellnew[2][2] = 2*r0

        atoms.set_cell(cellnew)
energies = np.array(energies)




### calc 2d bulk and shear modulus for BAFF: see https://doi.org/10.1021/acs.jpcc.2c06268
areas = [atoms.get_volume()/atoms.cell.cellpar()[2] for atoms in structures]
## bulk
print(polyfit(x=areas[:21], y=energies[:21], deg=3, full=True))
popt, data = polyfit(x=areas[:21], y=energies[:21], deg=3, full=True)
res_B = data[0][0]
# 2D Bulk modulus as Aopt*d^2E/dA^2 at Aopt in eV/Angstrom^2
B = 2*np.sqrt(popt[2]**2 - 3*popt[1]*popt[3]) * areas[0]
from ase import units
print("B2D [N/m]: ", B * units.m**2 / units.J)
## shear
strains = [((structures[0].cell[0][0] - atoms.cell[0][0])/structures[0].cell[0][0]) for atoms in structures]
popt, data = polyfit(strains[21:], energies[21:], 3, full=True)
res_mu = data[0][0]

# shear modulus (dE^2/ds^2)/4Aopt at smin in eV/Angstrom^2
mu = 2*np.sqrt(popt[2]**2 - 3*popt[1]*popt[3])/(4*areas[0])
print("mu2D [N/m]: ", mu * units.m**2 / units.J)

training_model = dict()
training_model['B'] = B
training_model['res_B'] = res_B
training_model['mu'] = mu
training_model['res_mu'] = res_mu
# compare to eq. 1 in paper:
training_model['Kbond'] =  2*np.sqrt(3) * B / 2  # /2 due to beta_r/l0^2=k/4 and in BAFFPotential implementation: 0.5 * Kbond
training_model['Kangle'] = (1/(mu*np.sqrt(3)) - 1/(2*np.sqrt(3) * B))**(-1) /9 * r0**2  # no *2 despite 0.5 * Kangle in BAFFPotential implementation, due to double counting of angles 
training_model['r0'] = r0
training_model['cosT0'] = -0.5
print('----- \n Training model BAFF: \n', training_model)

with open(Path(path,'training_model_BAFF.json'), 'w') as fp:
    json.dump(training_model, fp)

if args.sw:
    from cgf.cgatoms import init_cgatoms
    from cgf.utils.defects import generate_SW_defect
    from cgf.models.baff import BAFFPotential
    from cgf.utils.geometry import geom_optimize

    cg_SW = generate_SW_defect(reference_cell=structures[0].copy().cell, supercell_size=(3,3,1))
    # cg_SW = init_cgatoms(cg_atoms=cg_SW, r0=r0, linkage_length=training_model['linkage_length'])
    calculator = BAFFPotential(r0=r0, Kbond=training_model['Kbond'], Kangle=training_model['Kangle'])
    cg_SW.calc = calculator
    print('BAFF SW energy without optimization of positions of cores: ', cg_SW.get_potential_energy())

    cg_SW_o = geom_optimize(cg_SW, calculator, trajectory=None)
    print('BAFF SW energy with optimization of positions of cores: ', cg_SW_o.get_potential_energy())



### calc training for MikadoRR

with open(Path(path, '../id_groups_cores.json'), 'r') as inp:
    id_groups = json.load(inp)['id_groups']

# get coeffs with all available structures
results = get_optimal_coeffs(r0, structures, energies, id_groups=id_groups, width=4)
training_model = results['opt_training_model']
print('----- \n Optimal training model MikadoRR: \n', training_model)


with open(Path(path,'training_model_MikadoRR.json'), 'w') as fp:
    json.dump(training_model, fp)
with open(Path(path,'all_training_results_MikadoRR.json'), 'w') as fp:
    json.dump(results, fp)

if args.sw:
    from cgf.cgatoms import init_cgatoms
    from cgf.utils.defects import generate_SW_defect
    from cgf.models.surrogate import MikadoRR
    from cgf.utils.geometry import geom_optimize

    cg_SW = generate_SW_defect(reference_cell=structures[0].copy().cell, supercell_size=(3,3,1))
    cg_SW = init_cgatoms(cg_atoms=cg_SW, r0=r0, linkage_length=training_model['linkage_length'])
    calculator = MikadoRR(r0=r0, rr_coeff=np.array(training_model['rr_coeff']), rr_incpt=training_model['rr_incpt'], 
        opt=True, update_linker_sites=True, reevaluate_topology=False)
    cg_SW.calc = calculator
    print('MikadoRR SW energy without optimization of positions of cores: ', cg_SW.get_potential_energy())

    # cg_SW_o = geom_optimize(cg_SW, calculator, trajectory=None)
    cg_SW_o = geom_optimize(cg_SW, calculator, logfile=None, trajectory=None)
    print('MikadoRR SW energy with optimization of positions of cores: ', cg_SW_o.get_potential_energy())