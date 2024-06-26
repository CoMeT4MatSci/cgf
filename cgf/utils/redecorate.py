from ase import Atoms
from ase.cell import Cell
import numpy as np
from cgf.models.bnff import _get_phi0
from cgf.utils.geometry import rot_ar_z, get_left_right


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


def redecorate_cg_atoms_depricated(cg_atoms, linker_atoms, r0):
    '''
    DEPCRICATED. Should be removed soon.
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



def redecorate_cg_atoms(cg_atoms, linker_atoms, core_atoms=None, linkage_length=None):
    '''
    Redecorates coarse-grained model with real atoms.
    DOES NOT YET WORK FOR SMALL UNIT CELLS!

    
    Input:
    cg_atoms: ase atoms object of coarse-grained model that should be redecorated
    linker_atoms: ase atoms object of linker molecule. Must be aligned along x.
    core_atoms: ase atoms of core molecule (optional)
    linkage_length: optional. If None, linkage_length from linker_sites taken. Otherwise scaled accordingly
    
    Returns:
    ase atoms object
    '''
    

    core_linker_dir = cg_atoms.get_array('linker_sites')
    core_linker_neigh = cg_atoms.get_array('linker_neighbors')
    neigh_dist_vec = cg_atoms.get_array('neighbor_distances')
    neigh_ids = cg_atoms.get_array('neighbor_ids')
    positions = cg_atoms.get_positions()

    
    redecorated_atoms = Atoms(cell=cg_atoms.cell, pbc=True)

    # put core atoms on node positions (if core atoms present)
    if core_atoms:
        if isinstance(core_atoms.cell, Cell):
            # remove cell if present
            core_atoms.cell=None
        #core_atoms.center()
        for n, p in enumerate(positions):
            core_atoms_tmp = core_atoms.copy()

            # get average absolute orientation of node based on linker_sites
            phi_0 = 0
            n_linkersites = len(core_linker_dir[n])
            seen_ids = []

            for m in range(n_linkersites):
                phi_0s = []
                ref_vec = np.array([0, 1, 0])
                ref_vec = rot_ar_z(2*np.pi/n_linkersites*m) @ ref_vec
                for v2 in core_linker_dir[n]:
                    dot = np.dot(ref_vec, v2)
                    det = np.cross(ref_vec, v2)[2]
                    phi_0s.append(np.arctan2(det, dot))

                sorted_phi_0s, sorted_ids = zip(*sorted(zip(list(np.abs(phi_0s)), list(range(len(phi_0s))))))
                
                for sid in sorted_ids:
                    if sid in seen_ids:
                        continue
                    phi_0 += phi_0s[sid]
                    break

                seen_ids.append(sid)

            core_atoms_tmp.rotate('z', np.rad2deg(phi_0/n_linkersites))

            core_atoms_tmp.translate(p)
            redecorated_atoms += core_atoms_tmp

    if isinstance(linker_atoms.cell, Cell):
        # remove cell if present
        linker_atoms.cell=None
    linker_atoms.center()
    left, right = get_left_right(linker_atoms, natoms=1)
    linker_atoms.translate([-linker_atoms.get_positions()[left[0]][0], 0, 0])  # shift so that linker_atoms start at x=0

    # go through all beams and place the linker atoms accordingly there
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
            phi_nii = np.arctan2(det, dot)  # angle between neighbor_nii-core_ii vec and linker-site_nii vec

            
            s = linker_atoms.copy()

            # generate positions of the linkage sites based on phi and linkage_length
            linkage_site1 = positions[ii] + v2_ii
            linkage_site2 = positions[ii] + v1_ii + v2_nii

            linkage_vec = linkage_site2 - linkage_site1

            len_x_linker = s.positions[:,0].max()-s.positions[:,0].min()  # linker len along x
            len_linkage_vec = np.linalg.norm(linkage_vec)

            s.positions[:,0] *= len_linkage_vec/len_x_linker  # scaling linker to correct length along x
            
            # bending the linker accordingly (assumes linker is along x)

            posnew = []
            for n, sat in enumerate(s):

                lens = sat.position[0]

                # calculate the normal along the bent beam. To properly shift atoms that are not at y=0
                m = dwdx(lens/len_linkage_vec, phi_ii, phi_nii)
                nomal_w = np.array([-m, 1, 0])
                nomal_w /= np.linalg.norm(nomal_w)
                nomal_w *= sat.position[1]

                # displacement vector. Displaces atoms vertically along y and then shifts along normal_w
                disp_vec = w(lens/len_linkage_vec, phi_ii, phi_nii) * np.array([0, 1, 0]) * len_linkage_vec  + nomal_w
                disp_vec[1] -= sat.position[1]  # to correct if atoms are not y=0

                posnew.append([sat.position[0]+disp_vec[0], sat.position[1]+disp_vec[1], sat.position[2]])
            s.set_positions(posnew)


            # rotating linker to v1_ii
            dot = np.dot(v1_ii, np.array([1, 0, 0]))  # assumes initial linker orientation along x
            det = np.cross(v1_ii,  np.array([1, 0, 0]))[2]
            phi_v1_ii = np.arctan2(det, dot)
            # rotate to linkage_vec
            dot = np.dot(v1_ii, linkage_vec)  
            det = np.cross(v1_ii,  linkage_vec)[2]
            phi_linkage_vec = np.arctan2(det, dot)
            
            s.rotate(-(phi_v1_ii-phi_linkage_vec) * 180/np.pi, 'z')

            # translating pos of core along vector to neigh core
            s.translate(positions[ii]+v2_ii)  
            
            redecorated_atoms += s

    redecorated_atoms.wrap()
    return redecorated_atoms