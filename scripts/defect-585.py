import sys
sys.path.insert(0, '../')

import matplotlib.pyplot as plt

from cgf.cgatoms import *
from cgf.models.surrogate import MikadoRR, _find_linker_neighbor
from cgf.utils.cycles import find_cycles, cycle_graph
from cgf.utils.motifs import find_unique_motifs
from cgf.utils import remove_hatoms
from cgf.models.bnff import _get_bonds

from ase.io import read
from ase.constraints import FixedPlane
from ase.optimize import BFGS
from ase.neighborlist import NeighborList

import networkx as nx

from timeit import default_timer as timer
import cProfile as profile

import pickle


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

r0 = 35.082756/np.sqrt(3)
l0 = (1.42/2)*np.tan(np.pi/3)


print('-- read COF from gen file ...')
cof585 = read('../test-data/COF-5_585_opt.gen', index=-1)
remove_hatoms(cof585)


print('-- find cycles in structure ...')
#cy = find_cycles(cof585)

#with open('../test-data/cycles.data', 'wb') as filehandle:
    # Store the data as a binary data stream
#    pickle.dump(cy, filehandle)

with open('../test-data/cycles.data', 'rb') as filehandle:
    # Read the data as a binary data stream
    cy = pickle.load(filehandle)

print('-- construct cycle graph ...')
G_cy = cycle_graph(cy, cof585.positions)

for n in G_cy.nodes:
    G_cy.nodes[n]['cl'] = len(G_cy.nodes[n]['cycle'])

print('-- match motifs ...')
mfs = find_unique_motifs(motif, G_cy)
print(len(mfs))

print('-- construct CG Atoms ...')
r_c = np.array([G_cy.nodes[m['B']]['pos'].mean(axis=0) for m in mfs]) # compute core centers
core_linker_dir = [[G_cy.nodes[m[ls]]['pos'].mean(axis=0)-G_cy.nodes[m['B']]['pos'].mean(axis=0) for ls in ['A', 'C', 'D']] for m in mfs]

#cg_atoms = Atoms(['Y'] * len(r_c), positions=r_c, cell=cof585.cell, pbc=True) # create coarse-grained representation based on core centers
#cg_atoms.new_array('linker_sites', np.array(core_linker_dir)) # add positions of linker sites relative to core center
#_find_linker_neighbor(cg_atoms, r0)

s = read('../test-data/5-8-5_carbon.gen', index=-1)
s.set_cell(cof585.cell, scale_atoms=True)
cg_atoms = init_cgatoms(s, l0, r0=35.082756/np.sqrt(3))

## plot cg atoms ##
bonds = _get_bonds(cg_atoms, r0)

natoms = len(cg_atoms)
cell = cg_atoms.cell
positions = cg_atoms.positions
core_linker_dir = cg_atoms.get_array('linker_sites')
core_linker_neigh = cg_atoms.get_array('linker_neighbors')

nl = NeighborList( [1.2*r0/2] * natoms, self_interaction=False, bothways=True)
nl.update(cg_atoms)

fig = plt.figure(figsize=(10,10)) 
ax = fig.add_subplot(111, aspect='equal')

ax.scatter(positions[:,0],positions[:,1], color='coral', marker='o', s=75, zorder=10)
for i in range(core_linker_dir.shape[0]):
    for li in range(core_linker_dir.shape[1]):
        ax.arrow(positions[i,0],positions[i,1], core_linker_dir[i,li,0], core_linker_dir[i,li,1], color='coral', head_width=0.9, head_length=0.5, zorder=10)

for ib,b in enumerate(bonds):
    ii, nii, jj, njj = b[0], b[1], b[2], b[3]

    # get angle for site ii
    neighbors, offsets = nl.get_neighbors(ii)
    cells = np.dot(offsets, cell)
    distance_vectors = positions[neighbors] + cells - positions[ii]

    v2 = core_linker_dir[ii][core_linker_neigh[ii,nii]] # vector to linkage site
    v1 = distance_vectors[nii] # vector from ii to neighbor nii        
    
    ax.plot([positions[ii][0], positions[ii][0]+v1[0]/2], [positions[ii][1], positions[ii][1]+v1[1]/2], 'b--')
    ax.annotate(str(ib), [positions[ii][0]+v1[0]/2, positions[ii][1]+v1[1]/2])

plt.scatter(s.positions[:,0], s.positions[:,1], marker='x', color='black', zorder=2)

plt.savefig('../test-data/COF-5_585_opt_cg.pdf')
####


write_cgatoms(cg_atoms, '../test-data/COF-5_585_opt_cg.json')


calc = MikadoRR(r0=r0, 
                rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586, 4.45880373, 27.369685]), 
                rr_incpt=2315.3320266790165/6, opt=True)

cg_atoms.calc = calc


print('-- start energy calculation with MikadoRR_V2 ...')
start = timer()
E_V2 = cg_atoms.get_potential_energy()
end = timer()
print('-- finished (%4.1f s).' % (end-start))
print('-- energy = %5.3f\n' % (E_V2))

# for 2D optimization. Works only with ASE version directly from gitlab
#c = FixedPlane(
#    indices=[atom.index for atom in cg_atoms],
#    direction=[0, 0, 1],
#)

for n in range(len(cg_atoms)):
    ls = cg_atoms.get_array('linker_sites')[n]
    p = cg_atoms.positions[n][:2]
    for l in ls:
        v = p+l[:2]
        plt.scatter(v[0], v[1], color='red')
    plt.scatter(p[0], p[1], color='black')

plt.show()

#cg_atoms.set_constraint(c)

dyn = BFGS(cg_atoms, trajectory='585opt.traj')
dyn.run(fmax=0.01)
