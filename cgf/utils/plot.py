import numpy as np
import matplotlib.pyplot as plt
from cgf.utils.redecorate import w

def plot_cgatoms(cg_atoms, fig=None, ax=None,
                plot_beam=True, 
                plot_linker_sites=True,
                linkage_length='auto',
                beam_from_linkage=False,
                plot_neighbor_connections=False,
                plot_cell=True,
                savefigname=None):
    """Plots the positions of the cg_atoms, their orientation and the bonds

    Args:
        cg_atoms: cg_atoms with decorated new arrays
        fig (_type_, optional): fig object for subplots. Defaults to None. If None, new one is inilialized.
        ax (_type_, optional): ax object for subplots. Defaults to None. If None, new one is initialized.
        plot_beam (bool, optional): plots the elastic beam based on the linker-sites.
        linkage_length(str or float, optional): if 'auto' then linkage length determined by cg_atoms. Otherwise set to float value.
        beam_from_linkage(bool, optional): if True, the beam is calculated and plotted from the linkage to linkage, not cores to core.
        plot_neighbor_connections(bool, optional): if True, plots direct connections between neighbors. Good for bugfixing.
        plot_cell(bool, optional): if True, unit cell is plotted 
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
    scaling_factor = 100/np.sqrt(cg_atoms.get_volume()/cg_atoms.cell[2][2])

    if not fig:
        fig = plt.figure(figsize=(10,10))
    if not ax:
        ax = fig.add_subplot(111, aspect='equal')

    ax.scatter(positions[:,0],positions[:,1], color='#00305d', marker='o', s=50*scaling_factor, zorder=10)

    for ii in range(natoms):
        neighbors = neigh_ids[ii]
        for jj in range(len(neighbors)):

            # v1 refers to the vector connecting two cores
            # v2 refers to the linker_sites vector

            # calculate the angle between the vector between core ii and its neighbor
            # and the linker_site vector of core ii in that direction
            cln = core_linker_neigh[ii][jj]
            v1_ii = neigh_dist_vec[ii][jj]  # vec from ii to neighbor nii
            if linkage_length=='auto':
                v2_ii = core_linker_dir[ii][cln]  # linker_site vec from ii in direction of nii from cg_atoms object
            else:
                v2_ii = core_linker_dir[ii][cln]/np.linalg.norm(core_linker_dir[ii][cln]) * linkage_length  # linker_site vec from ii in direction of nii scaled

            dot = np.dot(v1_ii,v2_ii)
            det = np.cross(v1_ii,v2_ii)[2]
            psi_ii = np.arctan2(det, dot)

            # calculate the angle between the vector between the neighbor and core ii
            # and the linker_site vector of the neighbor in that direction
            n_ii = neigh_ids[ii][jj]  # core index of neighbor
            neighbors_nii = neigh_ids[n_ii]  # neighbors of this atom
            for kk in range(len(neighbors_nii)):
                v1_nii = neigh_dist_vec[n_ii][kk] # vector from nii to kk

                if np.allclose(v1_ii+v1_nii, np.zeros(3)):  # check if two vectors opposing each other (if v1_nii=-v1_ii)
                    cln_nii = core_linker_neigh[n_ii][kk]  # linker_site vec from nii in direction of ii
                    break

            if linkage_length=='auto':    
                v2_nii = core_linker_dir[n_ii][cln_nii]  # linker_site vec from nii in direction of ii from cg_atoms object
            else:
                v2_nii = core_linker_dir[n_ii][cln_nii]/np.linalg.norm(core_linker_dir[n_ii][cln_nii]) * linkage_length  # linker_site vec from nii in direction of ii scaled
                
            dot = np.dot(v1_nii,v2_nii)
            det = np.cross(v1_nii,v2_nii)[2]
            psi_nii = np.arctan2(det, dot)

            # generate positions of the linkage sites based on psi and linkage_length
            if beam_from_linkage:
                linkage_site1 = positions[ii] + v2_ii
                linkage_site2 = positions[ii] + v1_ii + v2_nii
                linkage_vec = linkage_site2 - linkage_site1
            else:
                linkage_site1 = positions[ii]
                linkage_site2 = positions[ii] + v1_ii 
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
                disp_vec = normal * w(lens/norm, psi_ii, psi_nii)
                xnew.append(x+disp_vec[0])
                ynew.append(y+disp_vec[1])
            if plot_beam:
                ax.plot(xnew, ynew,
                        color='lightsteelblue', linewidth=6*scaling_factor, zorder=-1)

            if plot_neighbor_connections:
                ax.plot([positions[ii][0], positions[ii][0] + v1_ii[0]],
                        [positions[ii][1], positions[ii][1] + v1_ii[1]], 
                        color='#00305d', zorder=5, linewidth=2*scaling_factor)
            if plot_linker_sites:
                ax.arrow(positions[ii,0],positions[ii,1], v2_ii[0], v2_ii[1], 
            color='darkred', head_width=1*scaling_factor, head_length=1*scaling_factor, lw=5*scaling_factor, zorder=10, alpha=0.8)

    if plot_cell:  # simulation cell
        ax.plot([0., cell.array[0,0]], [0., cell.array[0,1]], color='grey')
        ax.plot([0., cell.array[1,0]], [0., cell.array[1,1]], color='grey')
        ax.plot([cell.array[1,0], cell.array[1,0] + cell.array[0,0]], [cell.array[1,1], cell.array[1,1] + cell.array[0,1]], color='grey')
        ax.plot([cell.array[0,0], cell.array[1,0] + cell.array[0,0]], [cell.array[0,1], cell.array[1,1] + cell.array[0,1]], color='grey')

    plt.tight_layout()
    if savefigname:
        plt.savefig(savefigname, dpi=300)     

    return fig, ax