from ase import Atoms
import numpy as np
import time


def remove_hatoms(s):
    del s[[atom.symbol == 'H' for atom in s]]
    return s


def mol2Atoms(mol, repr='2D'):
    '''
    Construct ase Atoms object from rdkit mol object.
    Input:
    mol: rdkit mol object
    repr: '2D' or '3D'. '3D' additionally performs an UFF geometry optimization.
    
    Returns:
    ASE atoms object
    '''

    try:
        from rdkit.Chem import AllChem
        from rdkit import Chem
        from rdkit.Chem.rdMolTransforms import CanonicalizeMol, TransformConformer
    except ImportError:
        print('Not possible to import rdkit. Please install rdkit if you want to use mol2Atoms')

    AllChem.Compute2DCoords(mol)
    m2=Chem.AddHs(mol)
    
    if repr=='2D':
        # 2D coordinates
        AllChem.Compute2DCoords(m2)
    else:
        # or 3D
         AllChem.EmbedMolecule(m2)
         AllChem.UFFOptimizeMolecule(m2)
         CanonicalizeMol(m2, ignoreHs=False)
    
    c = m2.GetConformer(0)
    
    pos = c.GetPositions()

    nums = []
    for atom in c.GetOwningMol().GetAtoms():
        nums.append(atom.GetAtomicNum())

    return Atoms(positions=pos, numbers=nums)

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
    import matplotlib.pyplot as plt
    from cgf.redecorate import w

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



def geom_optimize(cg_atoms, calculator, trajectory=None, logfile=None, max_steps=500, fmax=0.01):
    from ase.constraints import FixedPlane
    from ase.optimize import BFGS
    
    starttime = time.time()

    cg_atoms.calc = calculator

    # for 2D optimization. Works only with ASE version directly from gitlab
    c = FixedPlane(
        indices=[atom.index for atom in cg_atoms],
        direction=[0, 0, 1],  # only move in xy plane
    )

    cg_atoms.set_constraint(c)

    dyn = BFGS(cg_atoms, trajectory=trajectory, logfile=logfile)
    dyn.run(fmax=fmax, steps=max_steps)
    cg_atoms_o = cg_atoms.calc.get_atoms()
    print(f"Relaxation time: {(time.time() - starttime):.2f} s")

    return cg_atoms_o

def cell_optimize(cg_atoms, calculator, isotropic=False, trajectory=None, logfile=None, max_steps=1000, fmax=0.01):
    from ase.constraints import FixedPlane
    from ase.optimize import BFGS
    from ase.filters import FrechetCellFilter
    starttime = time.time()

    cg_atoms.calc = calculator

    # for 2D optimization. Works only with ASE version directly from gitlab
    c = FixedPlane(
        indices=[atom.index for atom in cg_atoms],
        direction=[0, 0, 1],  # only move in xy plane
    )

    cg_atoms.set_constraint(c)
    ecf = FrechetCellFilter(cg_atoms, mask=[True, True, False, False, False, True], hydrostatic_strain=isotropic)

    dyn = BFGS(ecf, trajectory=trajectory, logfile=logfile)
    dyn.run(fmax=fmax, steps=max_steps)
    cg_atoms_o = cg_atoms.calc.get_atoms()
    print(f"Relaxation time: {(time.time() - starttime):.2f} s")

    return cg_atoms_o

def geom_optimize_efficient(cg_atoms, calculator, trajectory=None, logfile=None, max_steps=500, fmax=0.01):
    from ase.constraints import FixedPlane
    from ase.optimize import BFGS
    from cgf.surrogate import MikadoRR


    calc = MikadoRR(**calculator.todict())
    cg_atoms.calc = calc

    ### first: only optimize linker_sites
    print('Optimizing linker sites only...')
    cg_atoms.calc.parameters.opt = True
    cg_atoms.get_potential_energy()
    cg_atoms_o_ls = cg_atoms.calc.get_atoms()
    

    ### second: optimize geometry without optimizing linker sites
    print('Optimizing geometry without optimizing linker sites...')
    calc = MikadoRR(**calculator.todict())
    cg_atoms_o_ls.calc = calc
    cg_atoms_o_ls.calc.parameters.opt = False
    

    # for 2D optimization. Works only with ASE version directly from gitlab
    c = FixedPlane(
        indices=[atom.index for atom in cg_atoms_o_ls],
        direction=[0, 0, 1],  # only move in xy plane
    )

    cg_atoms_o_ls.set_constraint(c)

    dyn = BFGS(cg_atoms_o_ls, trajectory=trajectory, logfile=logfile)
    dyn.run(fmax=fmax, steps=max_steps)
    cg_atoms_o_pos = cg_atoms_o_ls.calc.get_atoms()

    ### thrid: optimize geometry and optimize linker sites
    print('Optimizing geometry and optimizing linker sites...')
    calc = MikadoRR(**calculator.todict())
    cg_atoms_o_pos.calc = calc
    cg_atoms_o_pos.calc.parameters.opt = True

    # for 2D optimization. Works only with ASE version directly from gitlab
    c = FixedPlane(
        indices=[atom.index for atom in cg_atoms_o_pos],
        direction=[0, 0, 1],  # only move in xy plane
    )

    cg_atoms_o_pos.set_constraint(c)

    dyn = BFGS(cg_atoms_o_pos, trajectory=trajectory)
    dyn.run(fmax=fmax, steps=max_steps)
    cg_atoms_o = cg_atoms_o_pos.calc.get_atoms()

    return cg_atoms_o





def numeric_stress_2D(atoms, d=1e-6, voigt=True):
    """Based on ase numeric_stress calculation"""
    stress = np.zeros((3, 3), dtype=float)

    cell = atoms.cell.copy()
    V = atoms.get_volume()
    for i in range(2):
        x = np.eye(3)
        x[i, i] += d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        eplus = atoms.get_potential_energy(force_consistent=True)

        x[i, i] -= 2 * d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        eminus = atoms.get_potential_energy(force_consistent=True)

        stress[i, i] = (eplus - eminus) / (2 * d * V)
        x[i, i] += d

        j = i - 2
        x[i, j] = d
        x[j, i] = d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        eplus = atoms.get_potential_energy(force_consistent=True)

        x[i, j] = -d
        x[j, i] = -d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        eminus = atoms.get_potential_energy(force_consistent=True)

        stress[i, j] = (eplus - eminus) / (4 * d * V)
        stress[j, i] = stress[i, j]
    atoms.set_cell(cell, scale_atoms=True)

    if voigt:
        return stress.flat[[0, 4, 8, 5, 2, 1]]
    else:
        return stress