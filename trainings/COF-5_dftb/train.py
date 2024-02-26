import json
import os

import networkx as nx
import numpy as np
from ase.io import read
from cgf.training_utils import extract_features, train_model
from cgf.utils import remove_hatoms

structures = []
energies = []
cells = []
bulk_or_shear = []
for fname in [f.path for f in os.scandir('Triphenylene-DB_1phenyl/') if f.is_dir()]:
    s0 = read(fname + '/result.json')
    energies.append(s0.get_potential_energy())
    cells.append(s0.cell.array.copy())
    bulk_or_shear.append('bulk' in fname)
    remove_hatoms(s0)
    structures.append(s0)
    
energies = np.array(energies)

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
core_descriptors, bond_descriptors = extract_features(motif=motif, atoms_list=structures, r0=r0)
training_model, reg = train_model(core_descriptors, bond_descriptors, energies)

print('reg coreff: ', training_model['rr_coeff'])
print('reg intercept: ', training_model['rr_incpt'])


with open('training_model.json', 'w') as fp:
    json.dump(training_model, fp)

