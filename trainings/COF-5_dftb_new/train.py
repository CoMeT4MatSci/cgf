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
from cgf.cycles import cycle_graph, find_cycles
from cgf.motifs import find_unique_motifs

traj = Trajectory('traj_training.traj')
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

def get_rc_linkersites_graphmotives(structure, mfs, cy):
    G_cy = cycle_graph(cy, structure.positions)
    r_c = np.array([G_cy.nodes[m['B']]['pos'].mean(axis=0) for m in mfs]) # compute core centers
    core_linker_dir = [[G_cy.nodes[m[ls]]['pos'].mean(axis=0)-G_cy.nodes[m['B']]['pos'].mean(axis=0) for ls in ['A', 'C', 'D']] for m in mfs]

    return r_c, core_linker_dir


# Tp Core Motif
motif = nx.Graph()
motif.add_edge("A", "B")
motif.add_edge("C", "B")
motif.add_edge("D", "B")

# all hexagons
motif.nodes['A']['cl'] = 6
motif.nodes['B']['cl'] = 6
motif.nodes['C']['cl'] = 6
motif.nodes['D']['cl'] = 6

# get coeffs with all available structures

r0 = structures[0].cell.cellpar()[0]/np.sqrt(3)

cy = find_cycles(structures[0])
G_cy = cycle_graph(cy, structures[0].positions)

# annotate cycles with cycle length
for n in G_cy.nodes:
    G_cy.nodes[n]['cl'] = len(G_cy.nodes[n]['cycle'])

mfs = find_unique_motifs(motif, G_cy)

n_training_structures, MSE_training, MSE_test = get_learning_curve(training_structures=training_structures,
                                                                    training_energies=training_energies,
                                                                    test_structures=test_structures,
                                                                    test_energies=test_energies,
                                                                    r0=r0,
                                                                    get_rc_linkersites_func=get_rc_linkersites_graphmotives,
                                                                    **{'mfs': mfs,
                                                                        'cy': cy})

plt.scatter(n_training_structures, MSE_training, label='Training MSE')
plt.scatter(n_training_structures, MSE_test, label='Test MSE')
plt.ylabel('MSE')
plt.xlabel('Number of Training data')
plt.legend()
plt.show()



core_descriptors, bond_descriptors = extract_features(structures, r0, 
                                                      get_rc_linkersites_func=get_rc_linkersites_graphmotives,
                                                       **{'mfs': mfs,
                                                          'cy': cy})
training_model, reg = train_model(core_descriptors, bond_descriptors, energies)


print('reg coreff: ', training_model['rr_coeff'])
print('reg intercept: ', training_model['rr_incpt'])


with open('training_model.json', 'w') as fp:
    json.dump(training_model, fp)


# compare to SW structure

from cgf.surrogate import MikadoRR
from cgf.utils import geom_optimize
SW = read('SW_defect/sw_relaxed.json')

cg_SW = generate_SW_defect(reference_cell=structures[0].copy().cell, supercell_size=(3,3,1))

print('Reference defect energy: ', SW.get_potential_energy() - 3*3*energies[0])

cg_SW = init_cgatoms(cg_atoms=cg_SW, r0=r0, linkage_length=2.5)


calculator = MikadoRR(r0=r0, rr_coeff=np.array(training_model['rr_coeff']), rr_incpt=training_model['rr_incpt'], 
        opt=True, update_linker_sites=True, reevaluate_Topology=True)

cg_SW.calc = calculator
print('SW energy without optimization of positions of cores: ', cg_SW.get_potential_energy())

cg_SW_o = geom_optimize(cg_SW, calculator, trajectory='trajoptcg.traj')
print('SW energy with optimization of positions of cores: ', cg_SW_o.get_potential_energy())



# r0 = 30.082756/np.sqrt(3) 
# core_descriptors, bond_descriptors = extract_features(motif=motif, atoms_list=[SW], r0=r0)
# training_model, reg = train_model(core_descriptors, bond_descriptors, np.array([SW.get_potential_energy()]))



# SW_energies = []
# n_training_structures = []
# for n in range(2, len(training_structures)+1, 1):

#     ids_train_tmp = [random.randrange(0, len(training_structures)) for _ in range(n)]
#     training_structures_tmp = [training_structures[id_training] for id_training in ids_train_tmp]
#     training_energies_tmp = [training_energies[id_training] for id_training in ids_train_tmp]
#     training_energies_tmp = np.array(training_energies_tmp)


#     core_descriptors, bond_descriptors = extract_features(motif=motif, atoms_list=training_structures_tmp, r0=r0)
#     training_model, reg = train_model(core_descriptors, bond_descriptors, training_energies_tmp)

#     cg_SW = generate_SW_defect(reference_cell=SW.cell, supercell_size=(3,3,1))
#     cg_SW = init_cgatoms(cg_atoms=cg_SW, r0=r0, linkage_length=2.5)
#     calculator = MikadoRR(r0=r0, rr_coeff=np.array(training_model['rr_coeff']), rr_incpt=training_model['rr_incpt'], 
#             opt=True, update_linker_sites=True, reevaluate_Topology=True)
#     cg_SW.calc = calculator
#     cg_SW_o = geom_optimize(cg_SW, calculator, trajectory='trajoptcg.traj')
#     print("SW energy test", cg_SW_o.get_potential_energy())
#     SW_energies.append(cg_SW_o.get_potential_energy())


#     n_training_structures.append(n)

# plt.scatter(n_training_structures, SW_energies, label='Training SW energies')
# plt.ylabel('SW energies')
# plt.xlabel('Number of Training data')
# plt.legend()
# plt.show()