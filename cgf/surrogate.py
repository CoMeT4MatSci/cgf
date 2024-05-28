import numpy as np

import itertools
import warnings

from ase import Atoms
from ase.calculators.calculator import Calculator

from .cgatoms import find_topology, find_neighbor_distances
from .bnff import _get_bonds

from scipy.optimize import minimize



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
#    core_desc = np.array(core_descriptors)
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
    
    #lenghts = bond_desc[:,0]
    psi0 = bond_descriptors[:,1]
    psi1 = bond_descriptors[:,2]

#    dlenghts_dRk = bond_desc_grad[:,0,:]
    dpsi0_dL = bond_descriptors_int_grad[:,1,:]
    dpsi1_dL = bond_descriptors_int_grad[:,2,:]
    
    # cores

    f = 2*np.pi/core_descriptors.shape[1]
    dphi0 = core_descriptors[:,0] - np.sign(core_descriptors[:,0]) * f
    dphi1 = core_descriptors[:,1] - np.sign(core_descriptors[:,1]) * f
    dphi2 = core_descriptors[:,2] - np.sign(core_descriptors[:,2]) * f
    dphi0_dL = core_descriptors_int_grad[:,0,:]
    dphi1_dL = core_descriptors_int_grad[:,1,:]
    dphi2_dL = core_descriptors_int_grad[:,2,:]

    # feature gradients matrix
    fg1 = 2*(psi0 @ dpsi0_dL)
    fg2 = (psi0 @ dpsi1_dL + psi1 @ dpsi0_dL)
    fg3 = 2*(psi1 @ dpsi1_dL)
    fg4 = 2*(dphi0 @ dphi0_dL + dphi1 @ dphi1_dL + dphi2 @ dphi2_dL)

    X = np.array([np.zeros((3,)), np.zeros((3,)), fg1 , fg2, fg3, fg4]).T
    return X


###############################################################################
# CALCULATOR V2
class MikadoRR(Calculator):
    """
    FF based on the Mikado Model and Ridge Regression.

    V2 of MikadoRR: - a bond_params variable was introduced
                    this means, no more need for relooping for grad calculation
                    - a lot of redundant calculations were removed
                    - cross products were replaced by cheaper expression


    """

    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']  # free_energy==energy (just added for numerical stress)
    default_parameters = {'rr_coeff': [0., 0., 0., 0., 0., 0.,],
                          'rr_incpt': 0.0,
                          'r0': 1.0,
                          'opt': True,
                          'update_linker_sites': True,
                          'reevaluate_topology': False}
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
        update_linker_sites: boolean
            updates linker_sites if optimized. Has only an effect if ASE optimizer is used
        reevaluate_topology: boolean
            reevaluates topology at each calculation, meaning 'neighbor_ids' and 'neighbor_distances' are updated
            Only necessary if unit-cell is too small (multiple neighbors with same ID)
        """
        Calculator.__init__(self, **kwargs)

        self._linkersites = None  # variable to save linkersite positions for geometry optimization
        self._warned = False  # checks if warning has already been printed
        self._gtol = kwargs.get('gtol', 1e-2)
        


    def calculate(self, atoms=None, properties=implemented_properties,
                  system_changes=['positions', 'numbers', 'cell',
                                  'pbc']):
        Calculator.calculate(self, atoms, properties, system_changes)

        # Check if reevaluate_topology should be used and give warning
        for a_id, n_ids in enumerate(self.atoms.get_array('neighbor_ids')):
            if len(n_ids)!=len(np.unique(n_ids)) and self.parameters.reevaluate_topology==False:
                if self._warned==False:
                    warnings.warn(f"Warning!!! Site {a_id} has neighbors with the same IDs ({n_ids})! \n \t This can lead to erroneous results if reevaluate_topology=False")
                    self._warned = True  # so that no warning spammed
        
        # FF parameters
        rr_coeff = self.parameters.rr_coeff
        rr_incpt = self.parameters.rr_incpt
        r0 = self.parameters.r0

        if self.parameters.reevaluate_topology==True:
            self.atoms = find_topology(self.atoms, r0)  # new NL is created
        elif self.parameters.reevaluate_topology==False:
            self.atoms = find_neighbor_distances(self.atoms)  # topology stays the same. Can cause issues if unit-cell too small

        if self.parameters.update_linker_sites:
            if self._linkersites is not None:
                self.atoms.set_array('linker_sites', self._linkersites)
            
        natoms = len(self.atoms)

        # Do optimization of linkersites
        if self.parameters.opt:
            p0 = self.atoms.get_array('linker_sites')
            res = minimize(_energy_gradient_internal, p0.reshape(-1), args=(self.atoms, rr_coeff, rr_incpt), 
                           method='BFGS', 
                           jac=True,
                           options={'gtol': self._gtol, 'disp': False})  # 1e-2 might be already enough?
            p = res.x.reshape(p0.shape)

            p = _renormalize_linker_lengths(self.atoms, p, p0)

            self.atoms.set_array('linker_sites', p)
            self._linkersites = p  # save linkersites in class for later use

        # get descriptors
        bonds = _get_bonds(self.atoms)
        bond_desc, bond_params, bond_ref = _get_bond_descriptors(self.atoms, bonds)
        bond_descriptors = [bond_desc,]

        core_desc = _get_core_descriptors(self.atoms)
        core_descriptors = [core_desc,]
        
        # get features
        X = get_feature_matrix(core_descriptors, bond_descriptors)

        # predict energy
        energy = np.dot(rr_coeff, X.reshape(-1)) + rr_incpt * len(bond_descriptors[0])
        self.results['energy'] = energy
        self.results['free_energy'] = energy

        # predict forces
        if 'forces' in properties:
            forces = np.zeros((natoms, 3)) # natoms, ndirections
            for at in range(natoms):
                forces[at,:] = -rr_coeff @ get_feature_gradient(core_desc, bond_desc,
                                                None, _get_bond_descriptors_gradient(bond_params, bond_ref, at)).T
            self.results['forces'] = forces
        if 'stress' in properties:
            self.results['stress'] = Calculator.calculate_numerical_stress(self, atoms)



###############################################################################

def _renormalize_linker_lengths(atoms, p, p0):
    # make sure that the distances to the linkage sites do not change
    dim = np.sum(atoms.pbc)
    if dim==2:  # if 2D
        d = np.where(atoms.pbc==False)[0][0]  # index of out-of-plane dimension
        p[:,:,d] = p0[:,:,d].copy()
    else:
        d = 3
    if dim==2:
        p_tmp = np.delete(p, np.s_[d], 2)
        p0_tmp = np.delete(p0, np.s_[d], 2)
    else:
        p_tmp = p.copy()
        p0_tmp = p0.copy()
        scaling_factors = []
    scaling_factors = np.zeros(p.shape[:2])
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            scaling_factors[i, j] = (np.linalg.norm(p0_tmp[i,j,:])/np.linalg.norm(p_tmp[i,j,:]))

    for k in range(p.shape[2]):
        if k==d:
            continue
        p[:,:,k] *= scaling_factors

    return p

def _energy_gradient_internal(p, cg_atoms, rr_coeff, rr_incpt):
    """Get energy and gradient wrt internal DOFs."""
    natoms = len(cg_atoms)    
    p0 = cg_atoms.get_array('linker_sites')
    p = p.reshape(p0.shape)


    if np.sum(cg_atoms.pbc)==2:  # if 2D
        d = np.where(cg_atoms.pbc==False)[0][0]
        p[:,:,d] = p0[:,:,d].copy()  # make sure that out of plane stays the same

    cg_atoms.set_array('linker_sites', p)

    # get descriptors
    bonds = _get_bonds(cg_atoms)
    bond_desc, bond_params, bond_ref = _get_bond_descriptors(cg_atoms, bonds)
    bond_descriptors = [bond_desc,]

    core_desc = _get_core_descriptors(cg_atoms)
    core_descriptors = [core_desc,]

    # get features
    X = get_feature_matrix(core_descriptors, bond_descriptors)

    # predict energy
    energy = np.dot(rr_coeff, X.reshape(-1)) + rr_incpt * len(bond_descriptors[0])    

    # predict internal gradient
    int_gradient = np.zeros((natoms*3, 3)) # natoms * nlinker_per_core, ndirections
    for (at,li) in itertools.product(range(natoms), range(3)):
        int_gradient[at*3 + li,:] = rr_coeff @ get_feature_internal_gradient(core_desc, 
                                                                bond_desc, 
                                _get_core_descriptors_internal_gradient(cg_atoms, at, li),
                                _get_bond_descriptors_internal_gradient(cg_atoms, bond_params, bond_ref, at, li)).T
    
    return energy, int_gradient.reshape(-1)

def _get_core_descriptors(cg_atoms):
    
    natoms = len(cg_atoms)

    core_linker_dir = cg_atoms.get_array('linker_sites')
    nangles = core_linker_dir.shape[1]

    core_desc = np.zeros((natoms, nangles))
    # iterate over atoms
    for ii in range(natoms):

        iangle = 0
        for li in range(core_linker_dir.shape[1]):
            v1 = core_linker_dir[ii, li]
            
            for lj in range(li+1,core_linker_dir.shape[1]):
                v2 = core_linker_dir[ii, lj]

                dot = np.dot(v1,v2)                
                det = v1[0]*v2[1] - v1[1]*v2[0] # = np.cross(v1,v2)[2]
                angle = np.arctan2(det, dot)
                
                core_desc[ii, iangle] = angle
                iangle += 1
    return core_desc

def calc_rs_dot_det_phi(v1, v2):

    r1 = np.linalg.norm(v1)    
#    r2 = np.linalg.norm(v2) # not needed
    r2 = r1

    dot = np.dot(v1, v2)
    det = v1[0]*v2[1] - v1[1]*v2[0]
    phi = np.arctan2(det, dot)
    return r1, r2, dot, det, phi

def _get_bond_descriptors(cg_atoms, bonds):

    core_linker_dir = cg_atoms.get_array('linker_sites')
    core_linker_neigh = cg_atoms.get_array('linker_neighbors')
    neigh_dist_vec = cg_atoms.get_array('neighbor_distances')

    bond_desc = np.zeros((len(bonds), 3))
    bond_params = []

    bond_ref = dict()  # references which bond_ids (ib) belong to which atoms (ii and jj)
    bond_ref['ii'] = dict()
    bond_ref['jj'] = dict()
    for ii in range(len(cg_atoms)):
        bond_ref['ii'][ii] = []
        bond_ref['jj'][ii] = []


    for ib, b in enumerate(bonds):
        ii, nii, jj, njj = b[0], b[1], b[2], b[3]

        # get angle for site ii
        distance_vectors = neigh_dist_vec[ii]

        v2_ii = core_linker_dir[ii][core_linker_neigh[ii,nii]] # vector to linkage site
        v1_ii = distance_vectors[nii] # vector from ii to neighbor nii

        r1_ii, r2_ii, dot_ii, det_ii, phi_ii = calc_rs_dot_det_phi(v1_ii, v2_ii)

        # get angle for site jj
        distance_vectors = neigh_dist_vec[jj]

        v2_jj = core_linker_dir[jj][core_linker_neigh[jj,njj]] # vector to linkage site
        v1_jj = distance_vectors[njj] # vector from jj to neighbor njj

        r1_jj, r2_jj, dot_jj, det_jj, phi_jj = calc_rs_dot_det_phi(v1_jj, v2_jj)

        bond_desc[ib, 0] = r1_jj 
        bond_desc[ib, 1] = phi_ii
        bond_desc[ib, 2] = phi_jj
        bond_params.append([[ii, v1_ii, v2_ii, r1_ii, r2_ii, dot_ii, det_ii, phi_ii, nii], 
                            [jj, v1_jj, v2_jj, r1_jj, r2_jj, det_jj, dot_jj, phi_jj, njj]])
        bond_ref['ii'][ii].append(ib)
        bond_ref['jj'][jj].append(ib)
    return bond_desc, bond_params, bond_ref

def _get_bond_descriptors_gradient(bond_params, bond_ref, at_k):


    ib1 = bond_ref['ii'][at_k]
    ib2 = bond_ref['jj'][at_k]
    ibs = ib1 + (ib2)
    bond_desc_grad = np.zeros((len(bond_params),3,3))
    for ib in ibs:
        ii, v1_ii, v2_ii, r1_ii, r2_ii, dot_ii, det_ii, phi_ii, nii = bond_params[ib][0]
        jj, v1_jj, v2_jj, r1_jj, r2_jj, det_jj, dot_jj, phi_jj, njj = bond_params[ib][1]

        # r1 = sqrt(v1.v1), hence grad r1 = v1/r1. Here, v1 = R_jj - R_ii

        # get angle for site ii
            
        # the derivative of  arctan x = 1/(1+x^2)
        # dot = v1 . v1 and thus grad dot = v1
        # det = v1[0] * v2[1] - v1[1] * v2[0] and thus grad det = (v2[1], - v2[0], 0) 
        if ii == at_k:
            bond_desc_grad[ib,0,:] = -1/r1_ii * v1_ii[:] # dr1_dR_k[:]            
            
            div_dot_ii = 1/dot_ii
            detsq_dotsq = (det_ii*div_dot_ii)**2
            det_dotsq = (det_ii*div_dot_ii**2)
            k = 1/(1+detsq_dotsq)
            bond_desc_grad[ib,1,0] = -k * (  v2_ii[1]*div_dot_ii - det_dotsq * v2_ii[0]  ) # dphi_i_dR_k[0]
            bond_desc_grad[ib,1,1] = -k * ( -v2_ii[0]*div_dot_ii - det_dotsq * v2_ii[1]  ) # dphi_i_dR_k[1]
            bond_desc_grad[ib,1,2] = -k * ( - det_dotsq * v2_ii[2] ) # dphi_i_dR_k[2]
        elif jj == at_k:
            div_dot_ii = 1/dot_ii
            detsq_dotsq = (det_ii*div_dot_ii)**2
            det_dotsq = (det_ii*div_dot_ii**2)
            k = 1/(1+detsq_dotsq)
            bond_desc_grad[ib,1,0] =  k * (  v2_ii[1]*div_dot_ii - det_dotsq * v2_ii[0]  )
            bond_desc_grad[ib,1,1] =  k * ( -v2_ii[0]*div_dot_ii - det_dotsq * v2_ii[1]  )
            bond_desc_grad[ib,1,2] =  k * ( - det_dotsq * v2_ii[2] )

        # get angle for site jj
        
        # the derivative of  arctan x = 1/(1+x^2)
        # dot = v1 . v1 and thus grad dot = v1
        # det = v1[0] * v2[1] - v1[1] * v2[0] and thus grad det = (v2[1], - v2[0], 0)        
        if jj == at_k:
            bond_desc_grad[ib,0,:] = 1/r1_ii * v1_ii[:] # dr1_dR_k[:]
            
            div_dot_jj = 1/dot_jj
            detsq_dotsq = (det_jj*div_dot_jj)**2
            det_dotsq = (det_jj*div_dot_jj**2)
            k = 1/(1+detsq_dotsq)
            bond_desc_grad[ib,2,0] = -k * (  v2_jj[1]*div_dot_jj - det_dotsq * v2_jj[0]  ) # dphi_j_dR_k[0]
            bond_desc_grad[ib,2,1] = -k * ( -v2_jj[0]*div_dot_jj - det_dotsq * v2_jj[1]  ) # dphi_j_dR_k[1]
            bond_desc_grad[ib,2,2] = -k * ( - det_dotsq * v2_jj[2] ) # dphi_j_dR_k[2]
        elif ii == at_k:
            div_dot_jj = 1/dot_jj
            detsq_dotsq = (det_jj*div_dot_jj)**2
            det_dotsq = (det_jj*div_dot_jj**2)
            k = 1/(1+detsq_dotsq)
            bond_desc_grad[ib,2,0] =  k * (  v2_jj[1]*div_dot_jj - det_dotsq * v2_jj[0]  )
            bond_desc_grad[ib,2,1] =  k * ( -v2_jj[0]*div_dot_jj - det_dotsq * v2_jj[1]  )
            bond_desc_grad[ib,2,2] =  k * ( - det_dotsq * v2_jj[2] )

    return bond_desc_grad

def _get_bond_descriptors_internal_gradient(cg_atoms, bond_params, bond_ref, at_k, li_k):

    core_linker_neigh = cg_atoms.get_array('linker_neighbors')

    ib1 = bond_ref['ii'][at_k]
    ib2 = bond_ref['jj'][at_k]
    ibs = ib1 + (ib2)
    bond_desc_grad = np.zeros((len(bond_params),3,3))
    #for ib, bp in enumerate(bond_params):
    for ib in ibs:
        ii, v1_ii, v2_ii, r1_ii, r2_ii, dot_ii, det_ii, phi_ii, nii = bond_params[ib][0]
        jj, v1_jj, v2_jj, r1_jj, r2_jj, det_jj, dot_jj, phi_jj, njj = bond_params[ib][1]

        # get angle for site ii
        if (at_k==ii) and (li_k==core_linker_neigh[ii,nii]):
            div_dot_ii = 1/dot_ii
            detsq_dotsq = (det_ii*div_dot_ii)**2
            det_dotsq = (det_ii*div_dot_ii**2)
            k = 1/(1+detsq_dotsq)
            bond_desc_grad[ib,1,0] =  k * ( -v1_ii[1]*div_dot_ii - det_dotsq * v1_ii[0]  ) # dphi_i_dL_ik[0]
            bond_desc_grad[ib,1,1] =  k * (  v1_ii[0]*div_dot_ii - det_dotsq * v1_ii[1]  ) # dphi_i_dL_ik[1]
            bond_desc_grad[ib,1,2] =  k * ( - det_dotsq * v1_ii[2] ) # dphi_i_dL_ik[2]

        # get angle for site jj
        if (at_k==jj) and (li_k==core_linker_neigh[jj,njj]):
            div_dot_jj = 1/dot_jj
            detsq_dotsq = (det_jj*div_dot_jj)**2
            det_dotsq = (det_jj*div_dot_jj**2)
            k = 1/(1+detsq_dotsq)
            bond_desc_grad[ib,2,0] =  k * ( -v1_jj[1]*div_dot_jj - det_dotsq * v1_jj[0]  ) # dphi_j_dL_ik[0]
            bond_desc_grad[ib,2,1] =  k * (  v1_jj[0]*div_dot_jj - det_dotsq * v1_jj[1]  ) # dphi_j_dL_ik[1]
            bond_desc_grad[ib,2,2] =  k * ( - det_dotsq * v1_jj[2] ) # dphi_j_dL_ik[2]
                
    return bond_desc_grad

def _get_core_descriptors_internal_gradient(cg_atoms, at_k, li_k):
    natoms = len(cg_atoms)
    core_linker_dir = cg_atoms.get_array('linker_sites')
    nangles = core_linker_dir.shape[1]

    core_desc_grad = np.zeros((natoms, nangles, 3))

    # iterate over atoms not necessary, since ii==at_k
    ii = at_k
    iangle = 0
    for li in range(core_linker_dir.shape[1]):
        v1 = core_linker_dir[ii, li]

        for lj in range(li+1,core_linker_dir.shape[1]):
            v2 = core_linker_dir[ii, lj]

            if (ii==at_k) and (li==li_k):
                dot = np.dot(v1,v2)
                det = v1[0]*v2[1] - v1[1]*v2[0] #= np.cross(v1,v2)[2]

                div_dot = 1/dot
                detsq_dotsq = (det * div_dot)**2
                det_dotsq = (det*div_dot**2)
                k = 1/(1+detsq_dotsq)

                core_desc_grad[ii, iangle, 0] =  k * (  v2[1]*div_dot - det_dotsq * v2[0]  ) # dangle_dL[0]
                core_desc_grad[ii, iangle, 1] =  k * ( -v2[0]*div_dot - det_dotsq * v2[1]  ) # dangle_dL[1]
                core_desc_grad[ii, iangle, 2] =  k * ( - det_dotsq * v2[2] ) # dangle_dL[2]

            elif (ii==at_k) and (lj==li_k):
                dot = np.dot(v1,v2)
                det = v1[0]*v2[1] - v1[1]*v2[0] #= np.cross(v1,v2)[2]

                div_dot = 1/dot
                detsq_dotsq = (det * div_dot)**2
                det_dotsq = (det*div_dot**2)
                k = 1/(1+detsq_dotsq)

                core_desc_grad[ii, iangle, 0] =  k * ( -v1[1]*div_dot - det_dotsq * v1[0]  )
                core_desc_grad[ii, iangle, 1] =  k * (  v1[0]*div_dot - det_dotsq * v1[1]  )
                core_desc_grad[ii, iangle, 2] =  k * ( - det_dotsq * v1[2] )

            iangle += 1
    
    return core_desc_grad