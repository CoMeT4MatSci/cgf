import json
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from ase.build import bulk
from ase.io import Trajectory, read
from cgf.cgatoms import init_cgatoms
from cgf.surrogate import MikadoRR, get_feature_matrix
from cgf.training_utils import (extract_features, get_learning_curve,
                                train_model)
from cgf.utils import remove_hatoms
from cgf.geometry_utils import generate_SW_defect

import matplotlib.pyplot as plt
from cgf.utils import plot_cgatoms

traj = Trajectory('traj_strain.traj')
structures = []
energies = []
for n, atoms in enumerate(traj):
    energies.append(atoms.get_potential_energy())
    remove_hatoms(atoms)
    structures.append(atoms)
energies = np.array(energies)


len_test = int(len(structures) * 0.3)
ids_test = [random.randrange(0, len(structures)) for _ in range(len_test)]
test_structures = [structures[id_test] for id_test in ids_test]
test_energies = [energies[id_test] for id_test in ids_test]
test_energies = np.array(test_energies)

training_structures = [structures[id_training] for id_training in range(len(structures)) if id_training not in ids_test]
training_energies = [energies[id_training] for id_training in range(len(energies)) if id_training not in ids_test]
training_energies = np.array(training_energies)


r0 = 11.531/np.sqrt(3) 

def extract_features(atoms_list, r0=r0):
    from ase.neighborlist import NeighborList, NewPrimitiveNeighborList, natural_cutoffs
    from ase import Atoms
    from cgf.cgatoms import find_topology, find_neighbor_distances, find_linker_neighbors
    from cgf.bnff import _get_bonds
    from cgf.surrogate import _get_bond_descriptors, _get_core_descriptors

    core_descriptors = []
    bond_descriptors = []
    for s0 in atoms_list:
        nl = NeighborList(natural_cutoffs(s0), self_interaction=False, bothways=True, 
                          primitive=NewPrimitiveNeighborList,
                          #primitive=PrimitiveNeighborList,
                          )
        nl.update(s0)
        r_c = []
        core_linker_dir = []
        for i in range(len(s0)):
            #print(i)
            neigh_ids, offsets = nl.get_neighbors(i)
            if len(neigh_ids)==3:
                #print(neigh_ids)
                r_c.append(s0[i].position)
                core_linker_dir_tmp = []
                for ni in neigh_ids:
                    core_linker_dir_tmp.append(s0[ni].position-s0[i].position)
                core_linker_dir.append(core_linker_dir_tmp)
        
        r_c = np.array(r_c)


        cg_atoms = Atoms(['Y'] * len(r_c), positions=r_c, cell=s0.cell, pbc=True) # create coarse-grained representation based on core centers
        cg_atoms.new_array('linker_sites', np.array(core_linker_dir)) # add positions of linker sites relative to core center

        cg_atoms = find_topology(cg_atoms, r0)
        cg_atoms = find_linker_neighbors(cg_atoms)

        #print(cg_atoms.get_array('linker_sites'))
        bonds = _get_bonds(cg_atoms)
        bond_desc, bond_params, bond_ref = _get_bond_descriptors(cg_atoms, bonds)
        bond_descriptors.append(bond_desc)

        core_desc = _get_core_descriptors(cg_atoms)
        core_descriptors.append(core_desc)
        
    return core_descriptors, bond_descriptors

# n_training_structures, MSE_training, MSE_test = get_learning_curve(training_structures=training_structures,
#                                                                     training_energies=training_energies,
#                                                                     test_structures=test_structures,
#                                                                     test_energies=test_energies,
#                                                                     motif=motif,
#                                                                     r0=r0)

# plt.scatter(n_training_structures, MSE_training, label='Training MSE')
# plt.scatter(n_training_structures, MSE_test, label='Test MSE')
# plt.ylabel('MSE')
# plt.xlabel('Number of Training data')
# plt.legend()
# plt.show()

# get coeffs with all available structures


core_descriptors, bond_descriptors = extract_features(atoms_list=structures, r0=r0)
training_model, reg = train_model(core_descriptors, bond_descriptors, energies)

print('reg coreff: ', training_model['rr_coeff'])
print('reg intercept: ', training_model['rr_incpt'])


with open('training_model.json', 'w') as fp:
    json.dump(training_model, fp)


# # compare to SW structure

from cgf.surrogate import MikadoRR
from cgf.utils import geom_optimize
import matplotlib.pyplot as plt
from cgf.utils import plot_cgatoms


# create unit-cell
cg_atoms = bulk('Y', 'hcp', a=3, b=3)
cg_atoms.positions[0][2] = 1.5 # vacuum in z-dir
cg_atoms.set_cell(structures[0].cell, scale_atoms=True)
cg_atoms = init_cgatoms(cg_atoms=cg_atoms, r0=r0, linkage_length=1.4)
fig, ax = plot_cgatoms(cg_atoms)
plt.show()


# SW = read('SW_defect/sw_relaxed.json')
cg_SW = generate_SW_defect(reference_cell=structures[0].cell, supercell_size=(3,3,1))

# print('Reference defect energy: ', SW.get_potential_energy() - 3*3*traj[0].get_potential_energy())

cg_SW = init_cgatoms(cg_atoms=cg_SW, r0=r0, linkage_length=1.4/2)


calculator = MikadoRR(r0=r0, rr_coeff=np.array(training_model['rr_coeff']), rr_incpt=training_model['rr_incpt'], 
        opt=True, update_linker_sites=True, reevaluate_Topology=True)

cg_SW.calc = calculator
print('SW energy without optimization of positions of cores: ', cg_SW.get_potential_energy())

fig, ax = plot_cgatoms(cg_SW)
plt.show()
cg_SW_o = geom_optimize(cg_SW, calculator, trajectory='trajoptcg.traj')
print('SW energy with optimization of positions of cores: ', cg_SW_o.get_potential_energy())
fig, ax = plot_cgatoms(cg_SW_o)
plt.show()

