import numpy as np

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.neighborlist import NeighborList

from .cycles import find_cycles, cycle_graph
from .bnff import _get_bonds, _get_phi0


def collect_descriptors(structures, cy, mfs, r0):
    core_descriptors = []
    bond_descriptors = []
    for s0 in structures:
        G_cy = cycle_graph(cy, s0.positions)

        r_c = np.array([G_cy.nodes[m['B']]['pos'].mean(axis=0) for m in mfs]) # compute core centers
        core_linker_dir = [[G_cy.nodes[m[ls]]['pos'].mean(axis=0)-G_cy.nodes[m['B']]['pos'].mean(axis=0) for ls in ['A', 'C', 'D']] for m in mfs]

        cg_atoms = Atoms(['Y'] * len(r_c), positions=r_c, cell=s0.cell, pbc=True) # create coarse-grained representation based on core centers
        cg_atoms.new_array('linker_sites', np.array(core_linker_dir)) # add positions of linker sites relative to core center

        nl = NeighborList( [1.2*r0/2] * len(cg_atoms), self_interaction=False, bothways=True)
        nl.update(cg_atoms)        
        
        _find_linker_neighbor(cg_atoms, r0, neighborlist=nl)

        bonds = _get_bonds(cg_atoms, r0, neighborlist=nl)
        bond_desc = _get_bond_descriptors(cg_atoms, r0, bonds, neighborlist=nl)
        bond_descriptors.append(bond_desc)

        core_desc = _get_core_descriptors(cg_atoms, r0, neighborlist=nl)
        core_descriptors.append(core_desc)
        
    return core_descriptors, bond_descriptors


def get_feature_matrix(core_descriptors, bond_descriptors):

    # bonds
    bond_desc = np.array(bond_descriptors)    
    lenghts = bond_desc[:,:,0]
    psi0 = bond_desc[:,:,1]
    psi1 = bond_desc[:,:,2]

    # cores
    core_desc = np.array(core_descriptors)
    dphi0 = core_desc[:,:,0] - 2*np.pi/core_desc.shape[2]
    dphi1 = core_desc[:,:,1] - 2*np.pi/core_desc.shape[2]
    dphi2 = core_desc[:,:,2] - 2*np.pi/core_desc.shape[2]

    # feature matrix
    X = np.array([lenghts.sum(axis=1), (lenghts**2).sum(axis=1), 
                  (psi0**2).sum(axis=1) , (psi1**2).sum(axis=1), (psi0*psi1).sum(axis=1), 
                  (dphi0**2).sum(axis=1) + (dphi1**2).sum(axis=1) + (dphi2**2).sum(axis=1)]).T

    return X


def _find_linker_neighbor(cg_atoms, r0, neighborlist=None):
    
    natoms = len(cg_atoms)
    cell = cg_atoms.cell
    positions = cg_atoms.positions

    nl = neighborlist
    if nl==None:
        nl = NeighborList( [1.2*r0/2] * natoms, self_interaction=False, bothways=True)
        nl.update(cg_atoms)

    core_linker_dir = cg_atoms.get_array('linker_sites')
    phi0 = 2*np.pi/core_linker_dir.shape[1]

    core_linker_neigh = []
    # iterate over atoms
    for ii in range(natoms):
        neighbors, offsets = nl.get_neighbors(ii)
        cells = np.dot(offsets, cell)
        distance_vectors = positions[neighbors] + cells - positions[ii]

        linker_neigh = []
        # iterate over neighbors of ii
        for jj in range(len(neighbors)):
            v1 = distance_vectors[jj] # vector from ii to jj
            r1 = np.linalg.norm(v1)

#             print(v1)
            for li, v2 in enumerate(core_linker_dir[ii]):
                dot = np.dot(v1,v2)
                det = np.cross(v1,v2)[2]
                angle = np.arctan2(det, dot)

                if np.abs(angle) < phi0/2:
#                     print(ii, jj, li, angle)
                    linker_neigh.append(li)
                    break
        core_linker_neigh.append(linker_neigh)

    cg_atoms.new_array('linker_neighbors', np.array(core_linker_neigh)) # add linker site id for each neighbor 
    
def _get_core_descriptors(cg_atoms, r0, neighborlist=None):
    
    natoms = len(cg_atoms)
    cell = cg_atoms.cell
    positions = cg_atoms.positions

    nl = neighborlist
    if nl==None:
        nl = NeighborList( [1.2*r0/2] * natoms, self_interaction=False, bothways=True)
        nl.update(cg_atoms)

    core_linker_dir = cg_atoms.get_array('linker_sites')
    phi0 = 2*np.pi/core_linker_dir.shape[1]

    core_desc = []    
    # iterate over atoms
    for ii in range(natoms):
        neighbors, offsets = nl.get_neighbors(ii)
        cells = np.dot(offsets, cell)
        distance_vectors = positions[neighbors] + cells - positions[ii]

        angles = []
        for li in range(core_linker_dir.shape[1]):
            v1 = core_linker_dir[ii, li]
            
            for lj in range(li+1,core_linker_dir.shape[1]):

                v2 = core_linker_dir[ii, lj]

                dot = np.dot(v1,v2)
                det = np.cross(v1,v2)[2]
                angle = np.arctan2(det, dot)

#                 print(ii, li, lj, angle)
                angles.append(np.abs(angle))
        core_desc.append(angles)
    return core_desc

def _get_bond_descriptors(cg_atoms, r0, bonds, neighborlist=None):
    natoms = len(cg_atoms)
    cell = cg_atoms.cell
    positions = cg_atoms.positions
    core_linker_dir = cg_atoms.get_array('linker_sites')
    core_linker_neigh = cg_atoms.get_array('linker_neighbors')

    nl = neighborlist
    if nl==None:
        nl = NeighborList( [1.2*r0/2] * natoms, self_interaction=False, bothways=True)
        nl.update(cg_atoms)

    bond_desc = []
    for b in bonds:
        ii, nii, jj, njj = b[0], b[1], b[2], b[3]

        # get angle for site ii
        neighbors, offsets = nl.get_neighbors(ii)
        cells = np.dot(offsets, cell)
        distance_vectors = positions[neighbors] + cells - positions[ii]

        v2 = core_linker_dir[ii][core_linker_neigh[ii,nii]] # vector to linkage site
        v1 = distance_vectors[nii] # vector from ii to neighbor nii

        r1 = np.linalg.norm(v1)    
        r2 = np.linalg.norm(v2)

        dot = np.dot(v1,v2)
        det = np.cross(v1,v2)[2]
        phi_i = np.arctan2(det, dot)

        # get angle for site jj
        neighbors, offsets = nl.get_neighbors(jj)
        cells = np.dot(offsets, cell)
        distance_vectors = positions[neighbors] + cells - positions[jj]

        v2 = core_linker_dir[jj][core_linker_neigh[jj,njj]] # vector to linkage site
        v1 = distance_vectors[njj] # vector from jj to neighbor njj

        r1 = np.linalg.norm(v1)    
        r2 = np.linalg.norm(v2)

        dot = np.dot(v1,v2)
        det = np.cross(v1,v2)[2]
        phi_j = np.arctan2(det, dot)

        bond_desc.append([r1, phi_i, phi_j])

    return bond_desc


# CALCULATOR

class MikadoRR(Calculator):
    """FF based on the Mikado Model and Ridge Regression.

    """

    implemented_properties = ['energy']
    default_parameters = {'rr_coeff': [0., 0., 0., 0., 0., 0.,],
                          'rr_incpt': 0.0,
                          'r0': 1.0}
    nolabel = True

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        rr_coeff: array of floats
            RR coefficients
        rr_incpt: float
            intercept of RR model [Energy]
        r0: float
            equilibrium distance in units of [length]
        """
        Calculator.__init__(self, **kwargs)

        self.nl = None

    def calculate(self, atoms=None, properties=implemented_properties,
                  system_changes=['positions', 'numbers', 'cell',
                                  'pbc']):

        Calculator.calculate(self, atoms, properties, system_changes)

        natoms = len(self.atoms)
        cell = self.atoms.cell
        positions = self.atoms.positions

        # FF parameters
        rr_coeff = self.parameters.rr_coeff
        rr_incpt = self.parameters.rr_incpt
        r0 = self.parameters.r0

        if self.nl == None:
            self.nl = NeighborList( [1.2*r0/2] * natoms, self_interaction=False, bothways=True)
        self.nl.update(self.atoms)

        # get descriptors
        bonds = _get_bonds(self.atoms, r0, neighborlist=self.nl)
        bond_desc = _get_bond_descriptors(self.atoms, r0, bonds, neighborlist=self.nl)
        bond_descriptors = [bond_desc,]

        core_desc = _get_core_descriptors(self.atoms, r0, neighborlist=self.nl)
        core_descriptors = [core_desc,]
        
        # get features
        X = get_feature_matrix(core_descriptors, bond_descriptors)

        # predict energy
        energy = np.dot(rr_coeff, X.reshape(-1)) + rr_incpt * len(bond_descriptors[0])

        self.results['energy'] = energy