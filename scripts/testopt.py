from ase.io import read
from ase.constraints import FixedPlane
from ase.optimize import BFGS
from ase.visualize import view
import matplotlib.pyplot as plt
import sys
from copy import deepcopy
from timeit import default_timer as timer
from ase.build import bulk

sys.path.insert(0, '../')
from cgf.cgatoms import *
from cgf.surrogate import MikadoRR
from cgf.surrogate_ref import MikadoRR as MikadoRR_V1
from cgf.utils import plot_cgatoms, geom_optimize, geom_optimize_efficient
from cgf.cgatoms import _find_linker_neighbor


################## Unit-cell
r0=30.082756/np.sqrt(3)

# create unit-cell
atoms = bulk('Y', 'hcp', a=30.0827, b=30.0)
atoms.positions[0][2] = 15.0 # vacuum in z-dir
#view(atoms)
cg_atoms = init_cgatoms(atoms.copy(), 2.46, r0=r0, linker_sites='nneighbors')
fig, ax = plot_cgatoms(cg_atoms,plot_neighbor_connections=True)
plt.show()


# Single Point without linker-sites optimization, but with reevaluate_topology=False
# This should give wrong results
calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
        4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, 
        opt=False, update_linker_sites=True, reevaluate_Topology=False)
cg_atoms = init_cgatoms(atoms, 2.46, r0=r0, linker_sites='nneighbors')
cg_atoms.calc = calculator
print('SP unit-cell opt=False, reevaluate_Topology=False', cg_atoms.get_potential_energy())
cg_atoms_o = cg_atoms.calc.get_atoms()
fig, ax = plot_cgatoms(cg_atoms_o)
plt.show()

# Single Point without linker-sites optimization, but with reevaluate_topology=True
calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
        4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, 
        opt=False, update_linker_sites=True, reevaluate_topology=True)
cg_atoms = init_cgatoms(atoms.copy(), 2.46, r0=r0, linker_sites='nneighbors')
cg_atoms.calc = calculator
print('SP unit-cell opt=False, reevalualte_topology=True', cg_atoms.get_potential_energy())
cg_atoms_o = cg_atoms.calc.get_atoms()
fig, ax = plot_cgatoms(cg_atoms_o)
plt.show()




################## SW Defect

r0=35.082756/np.sqrt(3)

calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
        4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=True, update_linker_sites=True)

# SW single point

cg_atoms = read_cgatoms('../test-data/COF-5_opt_SW_cg.json')
calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
        4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=False, update_linker_sites=True)
cg_atoms = init_cgatoms(cg_atoms, 2.46, r0=r0, linker_sites='nneighbors')
cg_atoms.calc = calculator
print('Singe Point SW opt=False', cg_atoms.get_potential_energy())


cg_atoms = read_cgatoms('../test-data/COF-5_opt_SW_cg.json')
calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
        4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=True, update_linker_sites=True)
cg_atoms = init_cgatoms(cg_atoms, 2.46, r0=r0, linker_sites='nneighbors')
cg_atoms.calc = calculator
print('Single Point SW opt=True', cg_atoms.get_potential_energy())


# SW defect optimization


print('SW: nneighbors with geom_optimize_efficient')
calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
        4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=True, update_linker_sites=True)
cg_atoms = read_cgatoms('../test-data/COF-5_opt_SW_cg.json')
start = timer()
cg_atoms = init_cgatoms(cg_atoms, 2.46, r0=r0, linker_sites='nneighbors')
cg_atoms_o = geom_optimize_efficient(cg_atoms, calculator, trajectory='traj.traj')
print("Etot SW: ", cg_atoms_o.get_potential_energy())
end = timer()
print('-- finished (%4.1f s).' % (end-start))

print('SW: best_angles with geom_optimize_efficient')
calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
        4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=True, update_linker_sites=True)
cg_atoms = read_cgatoms('../test-data/COF-5_opt_SW_cg.json')
start = timer()
cg_atoms = init_cgatoms(cg_atoms, 2.46, r0=r0, linker_sites='best_angles')
cg_atoms_o = geom_optimize_efficient(cg_atoms, calculator)
print("Etot SW: ", cg_atoms_o.get_potential_energy())
end = timer()
print('-- finished (%4.1f s).' % (end-start))



fig, ax = plot_cgatoms(cg_atoms_o)
plt.show()

##################### 585 defect


r0=35.082756/np.sqrt(3)

print('585: nneighbors with geom_optimize_efficient')
s = read("../test-data/5-8-5_carbon.gen")
cof585 = read("../test-data/COF-5_585_opt.gen")
s.set_cell(cof585.cell, scale_atoms=True)
cg_atoms = init_cgatoms(s, 2.46, r0=r0)
cg_atoms.pbc[2] = False
calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
        4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=True, update_linker_sites=True)
start = timer()
cg_atoms = init_cgatoms(cg_atoms, 2.46, r0=r0, linker_sites='nneighbors')
cg_atoms_o = geom_optimize_efficient(cg_atoms, calculator, trajectory='traj585.traj')
print("Etot 585: ", cg_atoms_o.get_potential_energy())
end = timer()
print('-- finished (%4.1f s).' % (end-start))


print('585: best_angles with geom_optimize_efficient')
s = read("../test-data/5-8-5_carbon.gen")
cof585 = read("../test-data/COF-5_585_opt.gen")
s.set_cell(cof585.cell, scale_atoms=True)
cg_atoms = init_cgatoms(s, 2.46, r0=r0)
cg_atoms.pbc[2] = False
calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
        4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=True, update_linker_sites=True)
start = timer()
cg_atoms = init_cgatoms(cg_atoms, 2.46, r0=r0, linker_sites='best_angles')
cg_atoms_o = geom_optimize_efficient(cg_atoms, calculator)
print("Etot 585: ", cg_atoms_o.get_potential_energy())
end = timer()
print('-- finished (%4.1f s).' % (end-start))


print('585: nneighbors with geom_optimize')
s = read("../test-data/5-8-5_carbon.gen")
cof585 = read("../test-data/COF-5_585_opt.gen")
s.set_cell(cof585.cell, scale_atoms=True)
cg_atoms = init_cgatoms(s, 2.46, r0=r0)
cg_atoms.pbc[2] = False
calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
        4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=True, update_linker_sites=True)
start = timer()
cg_atoms = init_cgatoms(cg_atoms, 2.46, r0=r0, linker_sites='nneighbors')
cg_atoms_o = geom_optimize(cg_atoms, calculator, trajectory='traj585.taj')
print("Etot 585: ", cg_atoms_o.get_potential_energy())
end = timer()
print('-- finished (%4.1f s).' % (end-start))


print('585: best_angles with geom_optimize')
s = read("../test-data/5-8-5_carbon.gen")
cof585 = read("../test-data/COF-5_585_opt.gen")
s.set_cell(cof585.cell, scale_atoms=True)
cg_atoms = init_cgatoms(s, 2.46, r0=r0)
cg_atoms.pbc[2] = False
calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
        4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=True, update_linker_sites=True)
start = timer()
cg_atoms = init_cgatoms(cg_atoms, 2.46, r0=r0, linker_sites='best_angles')
cg_atoms_o = geom_optimize(cg_atoms, calculator)
print("Etot 585: ", cg_atoms_o.get_potential_energy())
end = timer()
print('-- finished (%4.1f s).' % (end-start))
fig, ax = plot_cgatoms(cg_atoms_o)
plt.show()

