import json

import numpy as np
from ase.io import Trajectory
from cgf.cgatoms import init_cgatoms
from cgf.training_utils import (extract_features, 
                                train_model)
from cgf.utils import remove_hatoms
from cgf.geometry_utils import generate_SW_defect


traj = Trajectory('traj_training.traj')
structures = []
energies = []
for n, atoms in enumerate(traj):
    energies.append(atoms.get_potential_energy())
    remove_hatoms(atoms)
    structures.append(atoms)
energies = np.array(energies)


r0 = structures[0].cell.cellpar()[0]/np.sqrt(3) 


def get_rc_linkersites_neigh(structure):
    from ase.neighborlist import NeighborList, NewPrimitiveNeighborList, natural_cutoffs

    nl = NeighborList(natural_cutoffs(structure), self_interaction=False, bothways=True, 
                          primitive=NewPrimitiveNeighborList,
                          )
    nl.update(structure)
    r_c = []
    core_linker_dir = []
    for i in range(len(structure)):
        neigh_ids, offsets = nl.get_neighbors(i)
        if len(neigh_ids)==3:
            r_c.append(structure[i].position)
            core_linker_dir_tmp = []
            for ni in neigh_ids:
                core_linker_dir_tmp.append(structure[ni].position-structure[i].position)
            core_linker_dir.append(core_linker_dir_tmp)
    
    r_c = np.array(r_c)

    return r_c, core_linker_dir



# get coeffs with all available structures


core_descriptors, bond_descriptors = extract_features(structures=structures, r0=r0, get_rc_linkersites_func=get_rc_linkersites_neigh)
training_model, reg = train_model(core_descriptors, bond_descriptors, energies)

print('reg coreff: ', training_model['rr_coeff'])
print('reg intercept: ', training_model['rr_incpt'])


with open('training_model.json', 'w') as fp:
    json.dump(training_model, fp)


# compare to SW structure

from cgf.surrogate import MikadoRR
from cgf.utils import geom_optimize



cg_SW = generate_SW_defect(reference_cell=structures[0].cell, supercell_size=(3,3,1))
cg_SW = init_cgatoms(cg_atoms=cg_SW, r0=r0, linkage_length=1.4/2)


calculator = MikadoRR(r0=r0, rr_coeff=np.array(training_model['rr_coeff']), rr_incpt=training_model['rr_incpt'], 
        opt=True, update_linker_sites=True, reevaluate_Topology=True)

cg_SW.calc = calculator
print('SW energy without optimization of positions of cores: ', cg_SW.get_potential_energy())

cg_SW_o = geom_optimize(cg_SW, calculator, trajectory=None)
print('SW energy with optimization of positions of cores: ', cg_SW_o.get_potential_energy())


