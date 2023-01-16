import numpy as np

import itertools

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.neighborlist import NeighborList

from .cgatoms import _find_linker_neighbor
from .cycles import find_cycles, cycle_graph
from .bnff import _get_bonds, _get_phi0

from scipy.optimize import minimize


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
    dphi0 = core_desc[:,:,0] - np.sign(core_desc[:,:,0])*2*np.pi/core_desc.shape[2]
    dphi1 = core_desc[:,:,1] - np.sign(core_desc[:,:,1])*2*np.pi/core_desc.shape[2]
    dphi2 = core_desc[:,:,2] - np.sign(core_desc[:,:,2])*2*np.pi/core_desc.shape[2]

    # feature matrix
    X = np.array([lenghts.sum(axis=1), (lenghts**2).sum(axis=1), 
                  (psi0**2).sum(axis=1) , (psi1**2).sum(axis=1), (psi0*psi1).sum(axis=1), 
                  (dphi0**2).sum(axis=1) + (dphi1**2).sum(axis=1) + (dphi2**2).sum(axis=1)]).T

    return X

def get_feature_gradient(core_descriptors, bond_descriptors, core_descriptors_grad, bond_descriptors_grad):

    # bonds
    bond_desc = np.array(bond_descriptors)
    bond_desc_grad = np.array(bond_descriptors_grad)
    
    lenghts = bond_desc[:,0]
    psi0 = bond_desc[:,1]
    psi1 = bond_desc[:,2]

    dlenghts_dRk = bond_desc_grad[:,0,:]
    dpsi0_dRk = bond_desc_grad[:,1,:]
    dpsi1_dRk = bond_desc_grad[:,2,:]
    
    # cores
    core_desc = np.array(core_descriptors)
#    dphi0 = core_desc[:,0] - 2*np.pi/core_desc.shape[1]
#    dphi1 = core_desc[:,1] - 2*np.pi/core_desc.shape[1]
#    dphi2 = core_desc[:,2] - 2*np.pi/core_desc.shape[1]
    
    # the core descriptors do not depend on the COM position of the core
    # hence there is no gradient and core_descriptors_grad = None

    # feature gradients matrix
    dXdR = np.array([dlenghts_dRk.sum(axis=0), 2*(lenghts @ dlenghts_dRk), 
                  2*(psi0 @ dpsi0_dRk) , (psi0 @ dpsi1_dRk + psi1 @ dpsi0_dRk), 2*(psi1 @ dpsi1_dRk), 
                  np.zeros((3,))]).T

    return dXdR

def get_feature_internal_gradient(core_descriptors, bond_descriptors, core_descriptors_int_grad, bond_descriptors_int_grad):

    # bonds
    bond_desc = np.array(bond_descriptors)
    bond_desc_grad = np.array(bond_descriptors_int_grad)
    
    lenghts = bond_desc[:,0]
    psi0 = bond_desc[:,1]
    psi1 = bond_desc[:,2]

#    dlenghts_dRk = bond_desc_grad[:,0,:]
    dpsi0_dL = bond_desc_grad[:,1,:]
    dpsi1_dL = bond_desc_grad[:,2,:]
    
    # cores
    core_desc = np.array(core_descriptors)
    core_desc_grad = np.array(core_descriptors_int_grad)    
    dphi0 = core_desc[:,0] - np.sign(core_desc[:,0])*2*np.pi/core_desc.shape[1]
    dphi1 = core_desc[:,1] - np.sign(core_desc[:,1])*2*np.pi/core_desc.shape[1]
    dphi2 = core_desc[:,2] - np.sign(core_desc[:,2])*2*np.pi/core_desc.shape[1]
    dphi0_dL = core_desc_grad[:,0,:]
    dphi1_dL = core_desc_grad[:,1,:]
    dphi2_dL = core_desc_grad[:,2,:]

    # feature gradients matrix
    X = np.array([np.zeros((3,)), np.zeros((3,)), 
                  2*(psi0 @ dpsi0_dL) , (psi0 @ dpsi1_dL + psi1 @ dpsi0_dL), 2*(psi1 @ dpsi1_dL), 
                  2*(dphi0 @ dphi0_dL + dphi1 @ dphi1_dL + dphi2 @ dphi2_dL)]).T

    return X

    
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
                angles.append(angle)
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

def _get_bond_descriptors_gradient(cg_atoms, r0, bonds, at_k, neighborlist=None):
    natoms = len(cg_atoms)
    cell = cg_atoms.cell
    positions = cg_atoms.positions
    core_linker_dir = cg_atoms.get_array('linker_sites')
    core_linker_neigh = cg_atoms.get_array('linker_neighbors')

    nl = neighborlist
    if nl==None:
        nl = NeighborList( [1.2*r0/2] * natoms, self_interaction=False, bothways=True)
        nl.update(cg_atoms)

    bond_desc_grad = []
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

        # r1 = sqrt(v1.v1), hence grad r1 = v1/r1. Here, v1 = R_jj - R_ii
        dr1_dR_k = np.zeros((3,))
        if ii == at_k:
            dr1_dR_k[:] = -1/r1 * v1[:]
        elif jj == at_k:
            dr1_dR_k[:] = 1/r1 * v1[:]
            
            
        dot = np.dot(v1,v2)
        det = np.cross(v1,v2)[2]
        phi_i = np.arctan2(det, dot)

        # the derivative of  arctan x = 1/(1+x^2)
        # dot = v1 . v1 and thus grad dot = v1
        # det = v1[0] * v2[1] - v1[1] * v2[0] and thus grad det = (v2[1], - v2[0], 0) 
        dphi_i_dR_k = np.zeros((3,))
        if ii == at_k:
            dphi_i_dR_k[0] = -1/(1+(det/dot)**2) * (  v2[1]/dot - (det/dot**2) * v2[0]  )
            dphi_i_dR_k[1] = -1/(1+(det/dot)**2) * ( -v2[0]/dot - (det/dot**2) * v2[1]  )
            dphi_i_dR_k[2] = -1/(1+(det/dot)**2) * ( - (det/dot**2) * v2[2] )
        elif jj == at_k:
            dphi_i_dR_k[0] =  1/(1+(det/dot)**2) * (  v2[1]/dot - (det/dot**2) * v2[0]  )
            dphi_i_dR_k[1] =  1/(1+(det/dot)**2) * ( -v2[0]/dot - (det/dot**2) * v2[1]  )
            dphi_i_dR_k[2] =  1/(1+(det/dot)**2) * ( - (det/dot**2) * v2[2] )

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

        # the derivative of  arctan x = 1/(1+x^2)
        # dot = v1 . v1 and thus grad dot = v1
        # det = v1[0] * v2[1] - v1[1] * v2[0] and thus grad det = (v2[1], - v2[0], 0)        
        dphi_j_dR_k = np.zeros((3,))
        if jj == at_k:
            dphi_j_dR_k[0] = -1/(1+(det/dot)**2) * (  v2[1]/dot - (det/dot**2) * v2[0]  )
            dphi_j_dR_k[1] = -1/(1+(det/dot)**2) * ( -v2[0]/dot - (det/dot**2) * v2[1]  )
            dphi_j_dR_k[2] = -1/(1+(det/dot)**2) * ( - (det/dot**2) * v2[2] )
        elif ii == at_k:
            dphi_j_dR_k[0] =  1/(1+(det/dot)**2) * (  v2[1]/dot - (det/dot**2) * v2[0]  )
            dphi_j_dR_k[1] =  1/(1+(det/dot)**2) * ( -v2[0]/dot - (det/dot**2) * v2[1]  )
            dphi_j_dR_k[2] =  1/(1+(det/dot)**2) * ( - (det/dot**2) * v2[2] )

        bond_desc_grad.append([dr1_dR_k, dphi_i_dR_k, dphi_j_dR_k])

    return bond_desc_grad

def _get_core_descriptors_internal_gradient(cg_atoms, r0, at_k, li_k, neighborlist=None):
    
    natoms = len(cg_atoms)
    cell = cg_atoms.cell
    positions = cg_atoms.positions

    nl = neighborlist
    if nl==None:
        nl = NeighborList( [1.2*r0/2] * natoms, self_interaction=False, bothways=True)
        nl.update(cg_atoms)

    core_linker_dir = cg_atoms.get_array('linker_sites')
    phi0 = 2*np.pi/core_linker_dir.shape[1]

    core_desc_grad = []    
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

                dangle_dL = np.zeros((3,))
                if (ii==at_k) and (li==li_k):
                    dangle_dL[0] =  1/(1+(det/dot)**2) * (  v2[1]/dot - (det/dot**2) * v2[0]  )
                    dangle_dL[1] =  1/(1+(det/dot)**2) * ( -v2[0]/dot - (det/dot**2) * v2[1]  )
                    dangle_dL[2] =  1/(1+(det/dot)**2) * ( - (det/dot**2) * v2[2] )
                elif (ii==at_k) and (lj==li_k):
                    dangle_dL[0] =  1/(1+(det/dot)**2) * ( -v1[1]/dot - (det/dot**2) * v1[0]  )
                    dangle_dL[1] =  1/(1+(det/dot)**2) * (  v1[0]/dot - (det/dot**2) * v1[1]  )
                    dangle_dL[2] =  1/(1+(det/dot)**2) * ( - (det/dot**2) * v1[2] )
                    
                angles.append(dangle_dL)
        core_desc_grad.append(angles)
    return core_desc_grad

def _get_bond_descriptors_internal_gradient(cg_atoms, r0, bonds, at_k, li_k, neighborlist=None):
    natoms = len(cg_atoms)
    cell = cg_atoms.cell
    positions = cg_atoms.positions
    core_linker_dir = cg_atoms.get_array('linker_sites')
    core_linker_neigh = cg_atoms.get_array('linker_neighbors')

    nl = neighborlist
    if nl==None:
        nl = NeighborList( [1.2*r0/2] * natoms, self_interaction=False, bothways=True)
        nl.update(cg_atoms)

    bond_desc_grad = []
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
        
        dphi_i_dL_ik = np.zeros((3,))
        if (at_k==ii) and (li_k==core_linker_neigh[ii,nii]):
            dphi_i_dL_ik[0] =  1/(1+(det/dot)**2) * ( -v1[1]/dot - (det/dot**2) * v1[0]  )
            dphi_i_dL_ik[1] =  1/(1+(det/dot)**2) * (  v1[0]/dot - (det/dot**2) * v1[1]  )
            dphi_i_dL_ik[2] =  1/(1+(det/dot)**2) * ( - (det/dot**2) * v1[2] )

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

        dphi_j_dL_ik = np.zeros((3,))
        if (at_k==jj) and (li_k==core_linker_neigh[jj,njj]):
            dphi_j_dL_ik[0] =  1/(1+(det/dot)**2) * ( -v1[1]/dot - (det/dot**2) * v1[0]  )
            dphi_j_dL_ik[1] =  1/(1+(det/dot)**2) * (  v1[0]/dot - (det/dot**2) * v1[1]  )
            dphi_j_dL_ik[2] =  1/(1+(det/dot)**2) * ( - (det/dot**2) * v1[2] )
        
        
        bond_desc_grad.append([np.zeros((3,)), dphi_i_dL_ik, dphi_j_dL_ik])

    return bond_desc_grad

###############################################################################
# CALCULATOR

class MikadoRR(Calculator):
    """FF based on the Mikado Model and Ridge Regression.

    """

    implemented_properties = ['energy', 'forces']
    default_parameters = {'rr_coeff': [0., 0., 0., 0., 0., 0.,],
                          'rr_incpt': 0.0,
                          'r0': 1.0,
                          'opt': True}
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
        opt: boolean
            optimize internal DOFs
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
        opt = self.parameters.opt

        if self.nl == None:
            self.nl = NeighborList( [1.2*r0/2] * natoms, self_interaction=False, bothways=True)
        self.nl.update(self.atoms)

        if opt:
            p0 = self.atoms.get_array('linker_sites')
            res = minimize(_energy_gradient_internal, p0.reshape(-1), args=(self.atoms, r0, rr_coeff, rr_incpt), 
                           method='BFGS', 
                           jac=True,
                           options={'gtol': 1e-3, 'disp': True})
            p = res.x.reshape(p0.shape)
            # make sure that the distances to the linkage sites do not change
            for i in range(p.shape[0]):
                for j in range(p.shape[1]):
                    p[i,j,:] = p[i,j,:] * (np.linalg.norm(p0[i,j,:])/np.linalg.norm(p[i,j,:]))
            self.atoms.set_array('linker_sites', p)
        
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
        
        # predict forces
        if 'forces' in properties:
            forces = np.vstack([-rr_coeff @ get_feature_gradient(core_desc, bond_desc,
                                             None, _get_bond_descriptors_gradient(self.atoms, r0, bonds, at, neighborlist=self.nl)).T for at in range(natoms)])        
            self.results['forces'] = forces

        
def _energy_internal(p, cg_atoms, r0, rr_coeff, rr_incpt):
    """Get energy only."""
    p0 = cg_atoms.get_array('linker_sites')
    cg_atoms.set_array('linker_sites', p.reshape(p0.shape))

    nl = NeighborList( [1.2*r0/2] * len(cg_atoms), self_interaction=False, bothways=True)
    nl.update(cg_atoms)
    
    # get descriptors
    bonds = _get_bonds(cg_atoms, r0, neighborlist=nl)
    bond_desc = _get_bond_descriptors(cg_atoms, r0, bonds, neighborlist=nl)
    bond_descriptors = [bond_desc,]

    core_desc = _get_core_descriptors(cg_atoms, r0, neighborlist=nl)
    core_descriptors = [core_desc,]

    # get features
    X = get_feature_matrix(core_descriptors, bond_descriptors)

    # predict energy
    energy = np.dot(rr_coeff, X.reshape(-1)) + rr_incpt * len(bond_descriptors[0])    
    
    cg_atoms.set_array('linker_sites', p0)
    
    return energy

def _internal_gradient(p, cg_atoms, r0, rr_coeff, rr_incpt):
    """Get gradient wrt internal DOFs only."""
    natoms = len(cg_atoms)
    p0 = cg_atoms.get_array('linker_sites')
    cg_atoms.set_array('linker_sites', p.reshape(p0.shape))

    nl = NeighborList( [1.2*r0/2] * len(cg_atoms), self_interaction=False, bothways=True)
    nl.update(cg_atoms)

    bonds = _get_bonds(cg_atoms, r0, neighborlist=nl)
    
    int_gradient = np.vstack([rr_coeff @ get_feature_internal_gradient(_get_core_descriptors(cg_atoms, r0, neighborlist=nl), 
                                                                       _get_bond_descriptors(cg_atoms, r0, bonds, neighborlist=nl), 
                                _get_core_descriptors_internal_gradient(cg_atoms, r0, at, li, neighborlist=nl),
                                _get_bond_descriptors_internal_gradient(cg_atoms, r0, bonds, at, li, neighborlist=nl)).T for (at,li) in itertools.product(range(natoms), range(3))])
    cg_atoms.set_array('linker_sites', p0)
    
    return int_gradient.reshape(-1)

def _energy_gradient_internal(p, cg_atoms, r0, rr_coeff, rr_incpt):
    """Get energy and gradient wrt internal DOFs."""
    natoms = len(cg_atoms)    
    p0 = cg_atoms.get_array('linker_sites')
    cg_atoms.set_array('linker_sites', p.reshape(p0.shape))

    nl = NeighborList( [1.2*r0/2] * len(cg_atoms), self_interaction=False, bothways=True)
    nl.update(cg_atoms)
    
    # get descriptors
    bonds = _get_bonds(cg_atoms, r0, neighborlist=nl)
    bond_desc = _get_bond_descriptors(cg_atoms, r0, bonds, neighborlist=nl)
    bond_descriptors = [bond_desc,]

    core_desc = _get_core_descriptors(cg_atoms, r0, neighborlist=nl)
    core_descriptors = [core_desc,]

    # get features
    X = get_feature_matrix(core_descriptors, bond_descriptors)

    # predict energy
    energy = np.dot(rr_coeff, X.reshape(-1)) + rr_incpt * len(bond_descriptors[0])    

    # predict internal gradient
    int_gradient = np.vstack([rr_coeff @ get_feature_internal_gradient(_get_core_descriptors(cg_atoms, r0, neighborlist=nl), 
                                                                       _get_bond_descriptors(cg_atoms, r0, bonds, neighborlist=nl), 
                                _get_core_descriptors_internal_gradient(cg_atoms, r0, at, li, neighborlist=nl),
                                _get_bond_descriptors_internal_gradient(cg_atoms, r0, bonds, at, li, neighborlist=nl)).T for (at,li) in itertools.product(range(natoms), range(3))])
    
    
    cg_atoms.set_array('linker_sites', p0)
    
    return energy, int_gradient.reshape(-1)

def _num_internal_gradient(cg_atoms, at_k, li, i, r0, rr_coeff, rr_incpt, d=0.001):
    p0 = cg_atoms.get_array('linker_sites')
    p = p0.copy()
    p[at_k, li, i] += d

    energy_plus = _energy_internal(p.reshape(-1), cg_atoms, r0, rr_coeff, rr_incpt)

    p[at_k, li, i] -= 2 * d
    energy_minus = _energy_internal(p.reshape(-1), cg_atoms, r0, rr_coeff, rr_incpt)
        
    return (energy_plus - energy_minus)/(2*d)


###############################################################################
# CALCULATOR V2
class MikadoRR_V2(Calculator):
    """
    FF based on the Mikado Model and Ridge Regression.

    V2 of MikadoRR: a bond_params variable was introduced
                    this means, no more need for relooping for grad calculation

    """

    implemented_properties = ['energy', 'forces']
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
        bond_desc, bond_params = _get_bond_descriptors_V2(self.atoms, r0, bonds, neighborlist=self.nl)
        bond_descriptors = [bond_desc,]

        core_desc = _get_core_descriptors(self.atoms, r0, neighborlist=self.nl)
        core_descriptors = [core_desc,]
        
        # get features
        X = get_feature_matrix(core_descriptors, bond_descriptors)

        # predict energy
        energy = np.dot(rr_coeff, X.reshape(-1)) + rr_incpt * len(bond_descriptors[0])
        
        # predict forces
        forces = np.vstack([-rr_coeff @ get_feature_gradient(core_desc, bond_desc,
                                             None, _get_bond_descriptors_gradient_V2(self.atoms, bond_params, r0, at, neighborlist=self.nl)).T for at in range(natoms)])        
        
        self.results['energy'] = energy
        self.results['forces'] = forces

def _get_bond_descriptors_V2(cg_atoms, r0, bonds, neighborlist=None):
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
    bond_params = []
    for b in bonds:
        ii, nii, jj, njj = b[0], b[1], b[2], b[3]

        # get angle for site ii
        neighbors, offsets = nl.get_neighbors(ii)
        cells = np.dot(offsets, cell)
        distance_vectors = positions[neighbors] + cells - positions[ii]

        v2_ii = core_linker_dir[ii][core_linker_neigh[ii,nii]] # vector to linkage site
        v1_ii = distance_vectors[nii] # vector from ii to neighbor nii

        r1_ii = np.linalg.norm(v1_ii)    
        r2_ii = np.linalg.norm(v2_ii)

        dot_ii = np.dot(v1_ii, v2_ii)
        det_ii = np.cross(v1_ii, v2_ii)[2]
        phi_ii = np.arctan2(det_ii, dot_ii)

        # get angle for site jj
        neighbors, offsets = nl.get_neighbors(jj)
        cells = np.dot(offsets, cell)
        distance_vectors = positions[neighbors] + cells - positions[jj]

        v2_jj = core_linker_dir[jj][core_linker_neigh[jj,njj]] # vector to linkage site
        v1_jj = distance_vectors[njj] # vector from jj to neighbor njj

        r1_jj = np.linalg.norm(v1_jj)    
        r2_jj = np.linalg.norm(v2_jj)

        dot_jj = np.dot(v1_jj, v2_jj)
        det_jj = np.cross(v1_jj,v2_jj)[2]
        phi_jj = np.arctan2(det_jj, dot_jj)
        
        
        bond_desc.append([r1_jj, phi_ii, phi_jj])
        bond_params.append([[ii, v1_ii, v2_ii, r1_ii, r2_ii, dot_ii, det_ii, phi_ii], [jj, v1_jj, v2_jj, r1_jj, r2_jj, det_jj, dot_jj, phi_jj]])

    return bond_desc, bond_params

def _get_bond_descriptors_gradient_V2(cg_atoms, bond_params, r0, at_k, neighborlist=None):
    natoms = len(cg_atoms)


    nl = neighborlist
    if nl==None:
        nl = NeighborList( [1.2*r0/2] * natoms, self_interaction=False, bothways=True)
        nl.update(cg_atoms)

    bond_desc_grad = []
    for bp in bond_params:
        ii, v1_ii, v2_ii, r1_ii, r2_ii, dot_ii, det_ii, phi_ii = bp[0]
        jj, v1_jj, v2_jj, r1_jj, r2_jj, det_jj, dot_jj, phi_jj = bp[1]

        # get angle for site ii

        # r1 = sqrt(v1.v1), hence grad r1 = v1/r1. Here, v1 = R_jj - R_ii
        dr1_dR_k = np.zeros((3,))
        if ii == at_k:
            dr1_dR_k[:] = -1/r1_ii * v1_ii[:]
        elif jj == at_k:
            dr1_dR_k[:] = 1/r1_ii * v1_ii[:]
            

        # the derivative of  arctan x = 1/(1+x^2)
        # dot = v1 . v1 and thus grad dot = v1
        # det = v1[0] * v2[1] - v1[1] * v2[0] and thus grad det = (v2[1], - v2[0], 0) 
        dphi_i_dR_k = np.zeros((3,))
        if ii == at_k:
            dphi_i_dR_k[0] = -1/(1+(det_ii/dot_ii)**2) * (  v2_ii[1]/dot_ii - (det_ii/dot_ii**2) * v2_ii[0]  )
            dphi_i_dR_k[1] = -1/(1+(det_ii/dot_ii)**2) * ( -v2_ii[0]/dot_ii - (det_ii/dot_ii**2) * v2_ii[1]  )
            dphi_i_dR_k[2] = -1/(1+(det_ii/dot_ii)**2) * ( - (det_ii/dot_ii**2) * v2_ii[2] )
        elif jj == at_k:
            dphi_i_dR_k[0] =  1/(1+(det_ii/dot_ii)**2) * (  v2_ii[1]/dot_ii - (det_ii/dot_ii**2) * v2_ii[0]  )
            dphi_i_dR_k[1] =  1/(1+(det_ii/dot_ii)**2) * ( -v2_ii[0]/dot_ii - (det_ii/dot_ii**2) * v2_ii[1]  )
            dphi_i_dR_k[2] =  1/(1+(det_ii/dot_ii)**2) * ( - (det_ii/dot_ii**2) * v2_ii[2] )

        # get angle for site jj
        
        # the derivative of  arctan x = 1/(1+x^2)
        # dot = v1 . v1 and thus grad dot = v1
        # det = v1[0] * v2[1] - v1[1] * v2[0] and thus grad det = (v2[1], - v2[0], 0)        
        dphi_j_dR_k = np.zeros((3,))
        if jj == at_k:
            dphi_j_dR_k[0] = -1/(1+(det_jj/dot_jj)**2) * (  v2_jj[1]/dot_jj - (det_jj/dot_jj**2) * v2_jj[0]  )
            dphi_j_dR_k[1] = -1/(1+(det_jj/dot_jj)**2) * ( -v2_jj[0]/dot_jj - (det_jj/dot_jj**2) * v2_jj[1]  )
            dphi_j_dR_k[2] = -1/(1+(det_jj/dot_jj)**2) * ( - (det_jj/dot_jj**2) * v2_jj[2] )
        elif ii == at_k:
            dphi_j_dR_k[0] =  1/(1+(det_jj/dot_jj)**2) * (  v2_jj[1]/dot_jj - (det_jj/dot_jj**2) * v2_jj[0]  )
            dphi_j_dR_k[1] =  1/(1+(det_jj/dot_jj)**2) * ( -v2_jj[0]/dot_jj - (det_jj/dot_jj**2) * v2_jj[1]  )
            dphi_j_dR_k[2] =  1/(1+(det_jj/dot_jj)**2) * ( - (det_jj/dot_jj**2) * v2_jj[2] )

        bond_desc_grad.append([dr1_dR_k, dphi_i_dR_k, dphi_j_dR_k])

    return bond_desc_grad
