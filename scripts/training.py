import sys, os
sys.path.insert(0, '../')

import matplotlib.pyplot as plt
import numpy as np

import networkx as nx

from ase import Atoms
from ase.io import read
from ase.neighborlist import NeighborList

from cgf.cgatoms import *
from cgf.cgatoms import _find_linker_neighbor
from cgf.cycles import find_cycles, cycle_graph
from cgf.motifs import find_unique_motifs
from cgf.utils import remove_hatoms, plot_cgatoms, geom_optimize
from cgf.bnff import _get_bonds, _get_phi0
from cgf.surrogate import collect_descriptors, get_feature_matrix, MikadoRR, _get_core_descriptors, _get_bond_descriptors

def collect_descriptors_o(structures, cy, mfs, r0):
    core_descriptors = []
    bond_descriptors = []
    for s0 in structures:
        G_cy = cycle_graph(cy, s0.positions)
        
        r_c = np.array([G_cy.nodes[m['B']]['pos'].mean(axis=0) for m in mfs]) # compute core centers
        #print(r_c)
        #core_linker_dir = [[G_cy.nodes[m[ls]]['pos'].mean(axis=0)-G_cy.nodes[m['B']]['pos'].mean(axis=0) for ls in ['A', 'C', 'D']] for m in mfs]
        core_linker_dir = [[G_cy.nodes[m[ls]]['pos'].mean(axis=0)-r_c[n] for ls in ls_nodes] for n, m in enumerate(mfs)]
        #print(core_linker_dir)
        cg_atoms = Atoms(['Y'] * len(r_c), positions=r_c, cell=s0.cell, pbc=True) # create coarse-grained representation based on core centers
        cg_atoms.new_array('linker_sites', np.array(core_linker_dir)) # add positions of linker sites relative to core center

        cg_atoms = find_topology(cg_atoms, r0)
        cg_atoms = find_neighbor_distances(cg_atoms)
        cg_atoms = find_linker_neighbors(cg_atoms)

        #nl = NeighborList( [1.2*r0/2] * len(cg_atoms), self_interaction=False, bothways=True)
        #nl.update(cg_atoms)        
        
        #_find_linker_neighbor(cg_atoms, r0, neighborlist=nl)

        bonds = _get_bonds(cg_atoms, r0)
        bond_desc, bond_params, bond_ref = _get_bond_descriptors(cg_atoms, bonds)
        bond_descriptors.append(bond_desc)

        core_desc = _get_core_descriptors(cg_atoms)
        core_descriptors.append(core_desc)
        
    return core_descriptors, bond_descriptors

#from cgf.surrogate import (_find_linker_neighbor, _get_core_descriptors, _get_bond_descriptors, 
#                        collect_descriptors, get_feature_matrix, _energy_internal, _num_internal_gradient, _internal_gradient)


## Star and DB2 COF
#motif = nx.Graph()
#motif.add_edge("A", "D1")
#motif.add_edge("A", "E1")
#motif.add_edge("B", "D3")
#motif.add_edge("B", "F1")
#motif.add_edge("C", "F3")
#motif.add_edge("C", "E3")
#
#motif.add_edge("D1", "D2")
#motif.add_edge("D2", "D3")
#motif.add_edge("E1", "E2")
#motif.add_edge("E2", "E3")
#motif.add_edge("F1", "F2")
#motif.add_edge("F2", "F3")
#
## DB2 COF
#for n in motif.nodes:
#    motif.nodes[n]['cl'] = 2
#motif.nodes['A']['cl'] = 6
#motif.nodes['B']['cl'] = 6
#motif.nodes['C']['cl'] = 6
#
#ls_nodes = ['A', 'B', 'C']

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
ls_nodes = ['A', 'C', 'D']


# the individual structures are stored in separate directories as json files

#r0 = 35.33945946/np.sqrt(3)
r0 = 30.082756/np.sqrt(3)   

structures = []
energies = []
cells = []
bulk_or_shear = []
#for fname in [f.path for f in os.scandir('../test-data/DBA2-1phenyl/') if f.is_dir()]:
for fname in [f.path for f in os.scandir('../test-data/Triphenylene-DB_1phenyl/') if f.is_dir()]:
    s0 = read(fname + '/result.json')
    energies.append(s0.get_potential_energy())
    #s0 *= (2,2,1)
    cells.append(s0.cell.array.copy())
    bulk_or_shear.append('bulk' in fname)
    remove_hatoms(s0)
    structures.append(s0)
    
energies = np.array(energies)


# find the cycles in the first structure
# it is assumed that the topology does not change and we can reuse this information

cy = find_cycles(structures[0])

G_cy = cycle_graph(cy, structures[0].positions)

# annotate cycles with cycle length
for n in G_cy.nodes:
    G_cy.nodes[n]['cl'] = len(G_cy.nodes[n]['cycle'])


mfs = find_unique_motifs(motif, G_cy)

#core_descriptors, bond_descriptors = collect_descriptors_o(structures, cy, mfs, r0)
core_descriptors, bond_descriptors = collect_descriptors(structures, cy, mfs, r0)

print('number of samples: %d' % (len(bond_descriptors)))
print('number of linkers: %d' % (len(bond_descriptors[0])))
print('number of descriptors per linker: %d' % (len(bond_descriptors[0][0])))
print('number of cores: %d' % (len(core_descriptors[0])))
print('number of descriptors per core: %d' % (len(core_descriptors[0][0])))



from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error

# feature matrix
X = get_feature_matrix(core_descriptors, bond_descriptors)

# target values
y = energies-energies.min()

# this somehow does not work as I expect, see below
reg = RidgeCV(alphas=[1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12]).fit(X, y)
reg.score(X, y)
print('reg alpha: ', reg.alpha_)
print('reg score: ', reg.score(X, y))

#reg = Ridge(alpha=reg.alpha_).fit(X, y)
reg = Ridge(alpha=1e-11).fit(X, y)

print('reg score with alpha: ', reg.score(X, y))

print('mean squared error: ', mean_squared_error(reg.predict(X), y))
print('reg coreff: ', reg.coef_)
print('reg intercept: ', reg.intercept_)

print('reg coeff reference: ',  np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586, 4.45880373, 27.369685]))
print('reg intercept ref :', 2315.3320266790165)

plt.scatter(reg.predict(X), y)
plt.plot([0., y.max()], [0., y.max()], color='grey')
plt.xlabel('predicted energy')
plt.ylabel('actual energy')
plt.show()

