import json
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from ase.build import bulk
from ase.io import Trajectory
from cgf.cgatoms import init_cgatoms
from cgf.surrogate import MikadoRR, get_feature_matrix
from cgf.training_utils import (extract_features, get_learning_curve,
                                train_model)
from cgf.utils import remove_hatoms

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

r0 = 30.082756/np.sqrt(3) 


n_training_structures, MSE_training, MSE_test = get_learning_curve(training_structures=training_structures,
                                                                   training_energies=training_energies,
                                                                   test_structures=test_structures,
                                                                   test_energies=test_energies,
                                                                   motif=motif,
                                                                   r0=r0)

plt.scatter(n_training_structures, MSE_training, label='Training MSE')
plt.scatter(n_training_structures, MSE_test, label='Test MSE')
plt.ylabel('MSE')
plt.xlabel('Number of Training data')
plt.legend()
plt.show()

# get coeffs with all available structures

r0 = 30.082756/np.sqrt(3) 
core_descriptors, bond_descriptors = extract_features(motif=motif, atoms_list=structures, r0=r0)
training_model, reg = train_model(core_descriptors, bond_descriptors, energies)

print('reg coreff: ', training_model['rr_coeff'])
print('reg intercept: ', training_model['rr_incpt'])


with open('training_model.json', 'w') as fp:
    json.dump(training_model, fp)