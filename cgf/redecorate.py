from ase import Atoms
import numpy as np
from cgf.bnff import _get_phi0


def get_left_right(atoms, natoms, ignoreHs=False):
    '''
    Get natoms left/rightmost atoms.
    
    Input:
    atoms: ase atoms object
    natoms [int]: number of atoms to get each from furthest left/right
    ignoreHs [bool]: do not take H-atoms into account
    
    Returns:
    indices of left and right atoms
    ''' 
    srt_indx = np.argsort(atoms.positions[:,0])
    left, right = list(srt_indx[0:natoms]), list(srt_indx[-natoms:])
    
    if ignoreHs:
        new_left = []
        for i in left:
            if atoms.numbers[i] > 1:
                new_left.append(i)
        new_right = []                
        for i in right:
            if atoms.numbers[i] > 1:
                new_right.append(i)
        left = new_left
        right = new_right
        
    return left, right


def rot_ar_z(radi):
    # rotation matrix: rotation around z
    return  np.array([[np.cos(radi), -np.sin(radi), 0],
                      [np.sin(radi), np.cos(radi), 0],
                      [0, 0, 1]], dtype=np.double)


def w(x, phi0=0., phi1=0.):
    '''
    Vertical displacement of elastic beam.
    
    Inputs:
    x: normalized point(s) along beam [0,1]
    phi0: angle at core 0
    phi1: angle at core 1
    
    Returns:
    Vertical displacements
    '''
    return (x**2 - 2*x + 1)*x*phi0 + (x - 1)*x**2*phi1

def dwdx(x, phi0=0., phi1=0.):
    '''
    Derivative of vertical displacement of elastic beam dw/dx
    '''
    
    return (3*x**2 - 4*x + 1)*phi0 + (3*x**2 - 2*x)*phi1


def redecorate(cg_atoms, linker_atoms, r0):
    '''
    Redecorates coarse-grained model with real atoms.
    
    Input:
    cg_atoms: ase atoms object of coarse-grained model that should be redecorated
    linker_atoms: ase atoms object of linker molecule
    r0: distance between cores
    
    Returns:
    ase atoms object
    '''
    r_c = cg_atoms.positions  # positions of cg atoms
    phi0 = _get_phi0(cg_atoms, r0)  # get absolute angle
    phi_c = cg_atoms.get_initial_charges() + phi0  # calculate angle
    
    
    redecorated_atoms = Atoms(cell=cg_atoms.cell, pbc=True)
    
    ### find nearest neighbors (incl. in neighboring cells)
    nn_c = []
    for i in range(len(r_c)):
        for j in range(len(r_c)):
            if i != j:
                # s1 and s2 indicate if neighbor in same unit cell or in neighboring one
                for s1 in [-1,0,1]:
                    for s2 in [-1,0,1]:
                        Rvec = s1 * cg_atoms.cell.array[0,:] + s2 * cg_atoms.cell.array[1,:]
                        if np.linalg.norm(r_c[i]-r_c[j] - Rvec) < 1.2*r0:
                            nn_c.append([i,j,s1,s2])
    ###
    
    
    # linkers as flexible beams
    for i in range(len(nn_c)):
        if nn_c[i][0] > nn_c[i][1]:  # no double evaluation
            continue
        Rvec = nn_c[i][2] * cg_atoms.cell.array[0,:] + nn_c[i][3] * cg_atoms.cell.array[1,:]

        r_ij = r_c[nn_c[i][1]] - r_c[nn_c[i][0]] + Rvec
        l_ij = np.linalg.norm(r_ij) # distance between centers of neighboring cores
        r_ij = r_ij/l_ij  # unit vector
        r_ij_perp = np.cross(np.array([0,0,1.]), r_ij)  # vector perpendicular to beam direction
        phi_ij = np.mod(np.arctan2(r_ij[1], r_ij[0]), 2*np.pi)  # orientation of vector between neighbors

        phi0s = np.array([(phi_c[nn_c[i][0]]-phi_ij),
                       np.mod(phi_c[nn_c[i][0]]+2*np.pi/3, 2*np.pi)-phi_ij, 
                       np.mod(phi_c[nn_c[i][0]]+4*np.pi/3, 2*np.pi)-phi_ij])
        phi0 = phi0s[0]
        if phi0 > np.pi:
            phi0 = phi0 - 2*np.pi

        for p in phi0s:    
            if p > np.pi:
                p = p - 2*np.pi
            if p < -np.pi:
                p = p + 2*np.pi

            if np.abs(p) < np.abs(phi0):
                phi0 = p

        phi1s = np.array([np.mod(phi_c[nn_c[i][1]]+np.pi, 2*np.pi)-phi_ij, 
                       np.mod(phi_c[nn_c[i][1]]+2*np.pi/3+np.pi, 2*np.pi)-phi_ij, 
                       np.mod(phi_c[nn_c[i][1]]+4*np.pi/3+np.pi, 2*np.pi)-phi_ij])
        phi1 = phi1s[0]
        if phi1 > np.pi:
            phi1 = phi1 - 2*np.pi

        for p in phi1s:    
            if p > np.pi:
                p = p - 2*np.pi
            if p < -np.pi:
                p = p + 2*np.pi

            if np.abs(p) < np.abs(phi1):
                phi1 = p

        # find the linking sites for the linker per core
        # this need to be generalized for other cores
        linker_0 = rot_ar_z(phi0) @ ((1.42/2)*np.tan(np.pi/3)*r_ij) + r_c[nn_c[i][0]]
        linker_1 = rot_ar_z(phi1) @ (-(1.42/2)*np.tan(np.pi/3)*r_ij) + r_c[nn_c[i][1]] + Rvec
        linker_01 = linker_1 - linker_0
        phi_01 = np.mod(np.arctan2(linker_01[1], linker_01[0]), 2*np.pi)

        ellij = np.linalg.norm(linker_01) # the linker is shorter than the distance between cores


        ### transform linker according to fiber shape
        s = linker_atoms.copy()
        x0 = s.positions[:, 0].min()
        ell0 = ( s.positions[:,0].max()- s.positions[:,0].min() )
        for ati in range(len(s)):

            x = s.positions[ati, 0]
            y = s.positions[ati, 1]

            nx = -dwdx( (x-x0)/ell0, phi0, phi1)
            ny = 1.
            nn = np.sqrt(nx**2 + ny**2)
            nx = nx / nn
            ny = ny / nn

            # direction perpendicular to curve w(x) is [nx, ny]
            s.positions[ati, 0] = x0 + (x-x0)*(ellij/ell0) + y*nx
            s.positions[ati, 1] = ell0*w((x-x0)/ell0, phi0, phi1) + y*ny

        ### place linker at the center of vector connecting two cores
        ### and with correct orientation
        s.rotate(phi_01 * 180/np.pi, 'z')
        s.translate(linker_0 + linker_01/2)
        redecorated_atoms += s

    return redecorated_atoms


def redecorate_cg_atoms(cg_atoms, linker_atoms, r0):
    '''
    Redecorates coarse-grained model with real atoms.
    
    Input:
    cg_atoms: ase atoms object of coarse-grained model that should be redecorated
    linker_atoms: ase atoms object of linker molecule
    r0: distance between cores
    
    Returns:
    ase atoms object
    '''
    r_c = cg_atoms.positions  # positions of cg atoms
    phi0 = _get_phi0(cg_atoms, r0)  # get absolute angle
    phi_c = cg_atoms.get_initial_charges() + phi0  # calculate angle
    
    redecorated_atoms = Atoms(cell=cg_atoms.cell, pbc=True)
    
    ### find nearest neighbors (incl. in neighboring cells)
    nn_c = []
    for i in range(len(r_c)):
        for j in range(len(r_c)):
            if i != j:
                # s1 and s2 indicate if neighbor in same unit cell or in neighboring one
                for s1 in [-1,0,1]:
                    for s2 in [-1,0,1]:
                        Rvec = s1 * cg_atoms.cell.array[0,:] + s2 * cg_atoms.cell.array[1,:]
                        if np.linalg.norm(r_c[i]-r_c[j] - Rvec) < 1.2*r0:
                            nn_c.append([i,j,s1,s2])
    ###
    
    
    # linkers as flexible beams
    for i in range(len(nn_c)):
        if nn_c[i][0] > nn_c[i][1]:  # no double evaluation
            continue
        Rvec = nn_c[i][2] * cg_atoms.cell.array[0,:] + nn_c[i][3] * cg_atoms.cell.array[1,:]

        r_ij = r_c[nn_c[i][1]] - r_c[nn_c[i][0]] + Rvec
        l_ij = np.linalg.norm(r_ij) # distance between centers of neighboring cores
        r_ij = r_ij/l_ij  # unit vector
        r_ij_perp = np.cross(np.array([0,0,1.]), r_ij)  # vector perpendicular to beam direction
        phi_ij = np.mod(np.arctan2(r_ij[1], r_ij[0]), 2*np.pi)  # orientation of vector between neighbors

        phi0s = np.array([(phi_c[nn_c[i][0]]-phi_ij),
                       np.mod(phi_c[nn_c[i][0]]+2*np.pi/3, 2*np.pi)-phi_ij, 
                       np.mod(phi_c[nn_c[i][0]]+4*np.pi/3, 2*np.pi)-phi_ij])
        phi0 = phi0s[0]
        if phi0 > np.pi:
            phi0 = phi0 - 2*np.pi

        for p in phi0s:    
            if p > np.pi:
                p = p - 2*np.pi
            if p < -np.pi:
                p = p + 2*np.pi

            if np.abs(p) < np.abs(phi0):
                phi0 = p

        phi1s = np.array([np.mod(phi_c[nn_c[i][1]]+np.pi, 2*np.pi)-phi_ij, 
                       np.mod(phi_c[nn_c[i][1]]+2*np.pi/3+np.pi, 2*np.pi)-phi_ij, 
                       np.mod(phi_c[nn_c[i][1]]+4*np.pi/3+np.pi, 2*np.pi)-phi_ij])
        phi1 = phi1s[0]
        if phi1 > np.pi:
            phi1 = phi1 - 2*np.pi

        for p in phi1s:    
            if p > np.pi:
                p = p - 2*np.pi
            if p < -np.pi:
                p = p + 2*np.pi

            if np.abs(p) < np.abs(phi1):
                phi1 = p

        # find the linking sites for the linker per core
        # this need to be generalized for other cores
        linker_0 = rot_ar_z(phi0) @ ((1.42/2)*np.tan(np.pi/3)*r_ij) + r_c[nn_c[i][0]]
        linker_1 = rot_ar_z(phi1) @ (-(1.42/2)*np.tan(np.pi/3)*r_ij) + r_c[nn_c[i][1]] + Rvec
        linker_01 = linker_1 - linker_0
        phi_01 = np.mod(np.arctan2(linker_01[1], linker_01[0]), 2*np.pi)

        ellij = np.linalg.norm(linker_01) # the linker is shorter than the distance between cores


        ### transform linker according to fiber shape
        s = linker_atoms.copy()
        x0 = s.positions[:, 0].min()
        ell0 = ( s.positions[:,0].max()- s.positions[:,0].min() )
        for ati in range(len(s)):

            x = s.positions[ati, 0]
            y = s.positions[ati, 1]

            nx = -dwdx( (x-x0)/ell0, phi0, phi1)
            ny = 1.
            nn = np.sqrt(nx**2 + ny**2)
            nx = nx / nn
            ny = ny / nn

            # direction perpendicular to curve w(x) is [nx, ny]
            s.positions[ati, 0] = x0 + (x-x0)*(ellij/ell0) + y*nx
            s.positions[ati, 1] = ell0*w((x-x0)/ell0, phi0, phi1) + y*ny

        ### place linker at the center of vector connecting two cores
        ### and with correct orientation
        s.rotate(phi_01 * 180/np.pi, 'z')
        s.translate(linker_0 + linker_01/2)
        redecorated_atoms += s

    return redecorated_atoms



def redecorate_cg_atoms_new(cg_atoms, core_atoms, linker_atoms, linkage_length=None):
    '''
    Redecorates coarse-grained model with real atoms.
    
    Input:
    cg_atoms: ase atoms object of coarse-grained model that should be redecorated
    linker_atoms: ase atoms object of linker molecule. Must be aligned along x. Must have origin at 0.
    core_atoms: ase atoms of core molecule
    linkage_length: optional. If None, linkage_length from linker_neighbors taken. Otherwise scaled accordingly
    
    DOES NOT YET WORK FOR SMALL UNIT CELLS!
    Returns:
    ase atoms object
    '''
    

    core_linker_dir = cg_atoms.get_array('linker_sites')
    core_linker_neigh = cg_atoms.get_array('linker_neighbors')
    neigh_dist_vec = cg_atoms.get_array('neighbor_distances')
    neigh_ids = cg_atoms.get_array('neighbor_ids')
    positions = cg_atoms.get_positions()


    redecorated_atoms = Atoms(cell=cg_atoms.cell, pbc=True)
    if core_atoms:
        core_atoms.center()
        for p in positions:
            core_atoms_tmp = core_atoms.copy()
            core_atoms_tmp.translate(p)
            redecorated_atoms += core_atoms_tmp

    used_pairs = []
    for ii in range(len(cg_atoms)):
        neighbors = neigh_ids[ii]
        for jj in range(len(neighbors)):
            # to not double count:
            # ! DOES NOT WORK FOR SMALL CELLS!
            if ((ii, neighbors[jj]) in used_pairs) or ((neighbors[jj], ii) in used_pairs):
                continue
            used_pairs.append((ii, neighbors[jj]))

            # v1 refers to the vector connecting two cores
            # v2 refers to the linker_sites vector

            # calculate the angle between the vector between core ii and its neighbor
            # and the linker_site vector of core ii in that direction
            cln = core_linker_neigh[ii][jj]
            v1_ii = neigh_dist_vec[ii][jj]  # vec from ii to neighbor nii
            v2_ii = core_linker_dir[ii][cln]  # linker_site vec from ii in direction of nii
            dot = np.dot(v1_ii,v2_ii)
            det = np.cross(v1_ii,v2_ii)[2]
            phi_ii = np.arctan2(det, dot)

            # calculate the angle between the vector between the neighbor and core ii
            # and the linker_site vector of the neighbor in that direction
            n_ii = neigh_ids[ii][jj]  # core index of neighbor
            neighbors_nii = neigh_ids[n_ii]  # neighbors of this atom
            for kk in range(len(neighbors_nii)):
                v1_nii = neigh_dist_vec[n_ii][kk] # vector from nii to kk

                if np.allclose(v1_ii+v1_nii, np.zeros(3)):  # check if two vectors opposing each other (if v1_nii=-v1_ii)
                    cln_nii = core_linker_neigh[n_ii][kk]  # linker_site vec from nii in direction of ii
                    break
                        
            v2_nii = core_linker_dir[n_ii][cln_nii]  # linker_site vec from nii in direction of ii
            
            if linkage_length:
                v2_nii = linkage_length * v2_nii/np.linalg.norm(v2_nii)
                v2_ii = linkage_length * v2_ii/np.linalg.norm(v2_ii)
            
            dot = np.dot(v1_nii,v2_nii)
            det = np.cross(v1_nii,v2_nii)[2]
            phi_nii = np.arctan2(det, dot)  # angle between core_ii-neighbor vec and linker-site_ii vec



            linkervec_ii_nii = -v2_ii + v1_ii + v2_nii  # vector from one linkersite to the connected linkersite
            ell_ii_nii = np.linalg.norm(linkervec_ii_nii)  # norm
            norm = np.linalg.norm(v1_ii)  # norm of vector between cg sites
            normal = np.cross(np.array([0,0,1.]), v1_ii)  # normal vector
            # # bending the beam accordingly

            s = linker_atoms.copy()
            len_x_linker = s.positions[:,0].max()-s.positions[:,0].min()  # linker len along x

            s.positions[:,0] *= ell_ii_nii/len_x_linker   # scaling linker to correct length along x

            # rotating linker to v2_ii
            dot = np.dot(v1_ii, np.array([1, 0, 0]))  # assumes initial linker orientation along x
            det = np.cross(v1_ii,  np.array([1, 0, 0]))[2]
            phi_linker = np.arctan2(det, dot)
            s.rotate(-phi_linker * 180/np.pi, 'z')

            # translating pos of core + linkage_length along vector to neigh core
            s.translate(positions[ii] + np.linalg.norm(v2_ii) * v1_ii/np.linalg.norm(v1_ii))  
            
            # apply bending
            posnew = []
            for x, y, z in s.get_positions():
                 lens = np.sqrt((positions[ii][0]-x)**2 + (positions[ii][1]-y)**2)
                 disp_vec = normal * w(lens/norm, phi_ii, phi_nii)
                 posnew.append([x+disp_vec[0], y+disp_vec[1], z])
            s.set_positions(posnew)

            redecorated_atoms += s


    return redecorated_atoms