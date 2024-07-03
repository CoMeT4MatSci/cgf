import numpy as np
import matplotlib.pyplot as plt
from cgf.utils.redecorate import w

def plot_cgatoms(cg_atoms, fig=None, ax=None,
                plot_beam=True, 
                plot_linker_sites=True,
                plot_neighbor_connections=False,
                plot_cell=True,
                savefigname=None):
    """Plots the positions of the cg_atoms, their orientation and the bonds

    Args:
        cg_atoms: cg_atoms with decorated new arrays
        fig (_type_, optional): fig object for subplots. Defaults to None. If None, new one is inilialized.
        ax (_type_, optional): ax object for subplots. Defaults to None. If None, new one is initialized.
        savefigname (str, optional): Name of file to save image to. Defaults to None.

    Returns:
        fig, ax
    """


    natoms = len(cg_atoms)
    cell = cg_atoms.cell
    positions = cg_atoms.positions
    core_linker_dir = cg_atoms.get_array('linker_sites')
    core_linker_neigh = cg_atoms.get_array('linker_neighbors')
    neigh_ids = cg_atoms.get_array('neighbor_ids')
    neigh_dist_vec = cg_atoms.get_array('neighbor_distances')

    if not fig:
        fig = plt.figure(figsize=(10,10))
    if not ax:
        ax = fig.add_subplot(111, aspect='equal')

    ax.scatter(positions[:,0],positions[:,1], color='darkblue', marker='o', s=150/np.sqrt(len(cg_atoms)), zorder=10)

    for ii in range(natoms):
        neighbors = neigh_ids[ii]
        for jj in range(len(neighbors)):

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
            dot = np.dot(v1_nii,v2_nii)
            det = np.cross(v1_nii,v2_nii)[2]
            phi_nii = np.arctan2(det, dot)

            # generate positions of the linkage sites based on phi and linkage_length
            linkage_site1 = positions[ii] + v2_ii
            linkage_site2 = positions[ii] + v1_ii + v2_nii
            linkage_vec = linkage_site2 - linkage_site1


            xs = np.linspace(linkage_site1[0],
                            (linkage_site2[0])   ,
                                30)
            ys = np.linspace(linkage_site1[1],
                            (linkage_site2[1]),
                                30)

            # bending the beam accordingly
            norm = np.linalg.norm(linkage_vec)  # norm of vector between cg sites
            normal = np.cross(np.array([0,0,1.]), linkage_vec)  # normal vector
            xnew = []; ynew = []
            for x, y in zip(xs, ys):
                lens = np.sqrt((xs[0]-x)**2 + (ys[0]-y)**2)
                disp_vec = normal * w(lens/norm, phi_ii, phi_nii)
                xnew.append(x+disp_vec[0])
                ynew.append(y+disp_vec[1])
            if plot_beam:
                ax.plot(xnew, ynew,
                        color='lightsteelblue', linewidth=50/np.sqrt(len(cg_atoms)), zorder=-1)

            if plot_neighbor_connections:
                ax.plot([positions[ii][0], positions[ii][0] + v1_ii[0]],
                        [positions[ii][1], positions[ii][1] + v1_ii[1]], 
                        color='blue', zorder=5)
            if plot_linker_sites:
                ax.arrow(positions[ii,0],positions[ii,1], core_linker_dir[ii,cln,0], core_linker_dir[ii,cln,1], 
            color='darkred', head_width=0.5, head_length=0.5, lw=3, zorder=10, alpha=0.8)

    if plot_cell:  # simulation cell
        ax.plot([0., cell.array[0,0]], [0., cell.array[0,1]], color='grey')
        ax.plot([0., cell.array[1,0]], [0., cell.array[1,1]], color='grey')
        ax.plot([cell.array[1,0], cell.array[1,0] + cell.array[0,0]], [cell.array[1,1], cell.array[1,1] + cell.array[0,1]], color='grey')
        ax.plot([cell.array[0,0], cell.array[1,0] + cell.array[0,0]], [cell.array[0,1], cell.array[1,1] + cell.array[0,1]], color='grey')

    plt.tight_layout()
    if savefigname:
        plt.savefig(savefigname, dpi=300)     

    return fig, ax