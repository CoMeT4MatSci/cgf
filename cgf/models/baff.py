import numpy as np
from ase.calculators.calculator import Calculator
from ase.neighborlist import NeighborList

from cgf.utils.numeric_stress import numeric_stress_2D


class BAFFPotential(Calculator):
    """Valence force field with bond-stretching and angle-bending terms.

        The energy is given as

            E = sum_{ij} 0.5 * Kbond * (r_{ij} - r0)**2 + sum_{ijk} 0.5 * Kangle * (cosT_{jik} - cosT0)**2,

        where the first sum is over all bonds (counting each bond twice) and the second one
        over all angles with atom i at the center (also counting each angle twice).

    """

    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']  # free_energy==energy (just added for numerical stress)
    default_parameters = {'Kbond': 1.0,
                          'Kangle': 1.0,
                          'r0': 1.0,
                          'cosT0': -0.5}
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
        cosT0: float
            cosine of the equilibrium angle
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

        Kbond = self.parameters.Kbond
        Kangle = self.parameters.Kangle
        r0 = self.parameters.r0
        cosT0 = self.parameters.cosT0

        if self.nl == None:
            self.nl = NeighborList( [1.2*r0/2] * natoms, self_interaction=False, bothways=True)
        self.nl.update(self.atoms)

        forces = np.zeros((natoms, 3))
        energy = 0.0
        # iterate over all atoms
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
                forces[ii,:] += 4 * Kbond * v1/r1 * (r1 - r0)

                neighbors_jj, offsets_jj = self.nl.get_neighbors(neighbors[jj])
                cells_jj = np.dot(offsets_jj, cell)
                distance_vectors_jj = positions[neighbors_jj] + cells_jj - positions[neighbors[jj]]

                # iterate over neighbors of jj
                for kk in np.arange(len(neighbors_jj)):
                    v2 = distance_vectors_jj[kk] # vector from jj to kk
                    r2 = np.linalg.norm(v2)
                    cosT = -np.dot(v1,v2)/(r1*r2)

                    if np.isclose(cosT, 1.0):
                        continue

                    # angle bending contribution to forces
                    forces[ii,:] -= 4 * Kangle * (cosT - cosT0) * (v2)/(r1*r2)
                    forces[ii,:] += 4 * Kangle * (cosT - cosT0) * cosT * (-v1/r1**2)

                # iterate over neighbors of ii
                for kk in np.arange(len(neighbors)):
                    if jj == kk:
                        continue
                    v2 = distance_vectors[kk] # vector from ii to kk
                    r2 = np.linalg.norm(v2)
                    cosT = np.dot(v1,v2)/(r1*r2)

                    # angle bending contribution to energy and forces
                    energy += 0.5 * Kangle * (cosT - cosT0)**2
                    forces[ii,:] += 2 * Kangle * (cosT - cosT0) * (v1 + v2)/(r1*r2)
                    forces[ii,:] -= 2 * Kangle * (cosT - cosT0) * cosT * (v1/r1**2 + v2/r2**2)

        self.results['energy'] = energy
        self.results['free_energy'] = energy
        self.results['forces'] = forces
        if 'stress' in properties:
            self.results['stress'] = self.calculate_numerical_stress_2D(atoms)

    def calculate_numerical_stress_2D(self, atoms, d=1e-6, voigt=True):
        """Calculate numerical stress using finite difference."""

        return numeric_stress_2D(atoms, d=d, voigt=voigt)
