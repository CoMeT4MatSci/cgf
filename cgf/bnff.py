import numpy as np

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.neighborlist import NeighborList

class MikadoPotential(Calculator):
    """Effective force field based on the Mikado Model for flexible polymer fibers.

    """

    implemented_properties = ['energy']
    default_parameters = {'Kbond': 1.0,
                          'Kangle': 1.0,
                          'r0': 1.0}
    nolabel = True

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        Kbond: float
            force constant for bond-stretching in units of [Energy]/[length]^2
        Kangle: float
            force constant for angle-bending in units of [Energy]
        r0: float
            equilibrium distance in units of [length]
        """
        Calculator.__init__(self, **kwargs)

        self.nl = None

    def calculate(self, atoms=None, properties=implemented_properties,
                  system_changes=['positions', 'numbers', 'cell',
                                  'pbc', 'charges']):

        Calculator.calculate(self, atoms, properties, system_changes)

        natoms = len(self.atoms)
        cell = self.atoms.cell
        positions = self.atoms.positions
        phis = self.atoms.get_initial_charges()

        # FF parameters
        Kbond = self.parameters.Kbond
        Kangle = self.parameters.Kangle
        r0 = self.parameters.r0

        if self.nl == None:
            self.nl = NeighborList( [1.2*r0/2] * natoms, self_interaction=False, bothways=True)
        self.nl.update(self.atoms)

        bonds = _get_bonds(atoms, r0, neighborlist=self.nl)
        phi_energy = _get_phi_energy(phis, atoms, r0, bonds, neighborlist=self.nl)        
        
        energy = 0.5 * 3 * Kangle * phi_energy
        for ii in np.arange(natoms):
            neighbors, offsets = self.nl.get_neighbors(ii)
            cells = np.dot(offsets, cell)
            distance_vectors = positions[neighbors] + cells - positions[ii]

            # iterate over neighbors of ii
            for jj in np.arange(len(neighbors)):
                v1 = distance_vectors[jj] # vector from ii to jj
                r1 = np.linalg.norm(v1)

                # bond stretching contribution to energy and forces
                energy += 0.5 * Kbond * (r1 - r0)**2

        self.results['energy'] = energy

def _get_bonds(atoms):
    natoms = len(atoms)
    neigh_ids = atoms.get_array('neighbor_ids')

    bonds = []
    # iterate over atoms
    for ii in np.arange(natoms):
        neighbors = neigh_ids[ii]

        # iterate over neighbors of ii
        for jj in np.arange(len(neighbors)):
            neighbors_jj = neigh_ids[neighbors[jj]]

            # iterate over neighbors of jj
            for kk in np.arange(len(neighbors_jj)):

                if neighbors_jj[kk]==ii:                    
                    bonds.append([ii, jj, neighbors[jj], kk])

    return bonds


def _get_phi_energy(phis, atoms, r0, bonds, neighborlist=None):
    natoms = len(atoms)
    cell = atoms.cell
    positions = atoms.positions

    nl = neighborlist
    if nl==None:
        nl = NeighborList( [1.2*r0/2] * natoms, self_interaction=False, bothways=True)
        nl.update(atoms)

    energy = 0.0
    for b in bonds:
        ii, nii, jj, njj = b[0], b[1], b[2], b[3]

        # get angle for site ii
        neighbors, offsets = nl.get_neighbors(ii)
        cells = np.dot(offsets, cell)
        distance_vectors = positions[neighbors] + cells - positions[ii]

        # vector to first neighbor
        v1 = distance_vectors[0]
        r1 = np.linalg.norm(v1)

        v2 = distance_vectors[nii] # vector from ii to neighbor nii
        r2 = np.linalg.norm(v2)

        dot = np.dot(v1,v2)
        det = np.cross(v1,v2)[2]
        angle = np.arctan2(det, dot)
        angle_2pi = ((1-np.sign(angle))*np.pi + angle)

        if angle > 0.0:
            phi_i = 2*np.pi/3 + phis[ii] - angle_2pi
        else:
            phi_i = 4*np.pi/3 + phis[ii] - angle_2pi

        if nii == 0:
            phi_i = phis[ii]

        # get angle for site jj
        neighbors, offsets = nl.get_neighbors(jj)
        cells = np.dot(offsets, cell)
        distance_vectors = positions[neighbors] + cells - positions[jj]

        # vector to first neighbor
        v1 = distance_vectors[0]
        r1 = np.linalg.norm(v1)

        v2 = distance_vectors[njj] # vector from ii to neighbor nii
        r2 = np.linalg.norm(v2)

        dot = np.dot(v1,v2)
        det = np.cross(v1,v2)[2]
        angle = np.arctan2(det, dot)
        angle_2pi = ((1-np.sign(angle))*np.pi + angle)

        if angle > 0.0:
            phi_j = 2*np.pi/3 + phis[jj] - angle_2pi
        else:
            phi_j = 4*np.pi/3 + phis[jj] - angle_2pi

        if njj == 0:
            phi_j = phis[jj]

        energy += phi_i**2 + phi_i*phi_j + phi_j**2

    return energy


def _get_phi0(atoms, r0, neighborlist=None):
    natoms = len(atoms)
    cell = atoms.cell
    positions = atoms.positions

    nl = neighborlist
    if nl==None:
        nl = NeighborList( [1.2*r0/2] * natoms, self_interaction=False, bothways=True)
        nl.update(atoms)

    phi0 = np.zeros(natoms)
    for ii in np.arange(natoms):
        neighbors, offsets = nl.get_neighbors(ii)
        cells = np.dot(offsets, cell)
        distance_vectors = positions[neighbors] + cells - positions[ii]

        # vector to first neighbor
        v1 = distance_vectors[0]

        angle = np.arctan2(v1[1], v1[0])
        angle_2pi = ((1-np.sign(angle))*np.pi + angle)

        phi0[ii] = angle_2pi

    return phi0


def _energy(pos, atoms):
    atoms.set_positions(np.reshape(pos, (-1,3)))
    return atoms.get_potential_energy()

def _energy_ch(ch, atoms):
    natoms = len(atoms)

    atoms.set_initial_charges(ch[0:natoms])
    return atoms.get_potential_energy()

def _energy_total(pos_ch, atoms):
    natoms = len(atoms)

    atoms.set_initial_charges(pos_ch[0:natoms])
    atoms.set_positions(np.reshape(pos_ch[natoms:], (-1,3)))
    return atoms.get_potential_energy()
