import json
import networkx as nx
import numpy as np
from ase.io import Trajectory

from cgf.training_utils import (extract_features, 
                                train_model)
from cgf.utils import remove_hatoms
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
from cgf.geometry_utils import generate_SW_defect
from cgf.cgatoms import init_cgatoms

cg_SW = generate_SW_defect(reference_cell=structures[0].copy().cell, supercell_size=(3,3,1))

cg_SW = init_cgatoms(cg_atoms=cg_SW, r0=r0, linkage_length=2.5)


calculator = MikadoRR(r0=r0, rr_coeff=np.array(training_model['rr_coeff']), rr_incpt=training_model['rr_incpt'], 
        opt=True, update_linker_sites=True, reevaluate_Topology=True)

cg_SW.calc = calculator
print('SW energy without optimization of positions of cores: ', cg_SW.get_potential_energy())

cg_SW_o = geom_optimize(cg_SW, calculator, trajectory=None)
print('SW energy with optimization of positions of cores: ', cg_SW_o.get_potential_energy())