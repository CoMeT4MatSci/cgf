from ase import Atoms
import numpy as np

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

    ax.scatter(positions[:,0],positions[:,1], color='darkblue', marker='o', s=75, zorder=10)

    for ii in range(natoms):
        neighbors = neigh_ids[ii]
        for jj in range(len(neighbors)):

            cln = core_linker_neigh[ii][jj]
            v1_ii = neigh_dist_vec[ii][jj]  # vec from ii to neighbor nii
            v2_ii = core_linker_dir[ii][cln]  # linker_site vec
            dot = np.dot(v1_ii,v2_ii)
            det = np.cross(v1_ii,v2_ii)[2]
            phi_ii = np.arctan2(det, dot)

            n_ii = neigh_ids[ii][jj]
            neighbors_nii = neigh_ids[n_ii]
            for kk in range(len(neighbors_nii)):
                if neighbors_nii[kk]==ii:
                    cln_nii = core_linker_neigh[n_ii][kk]
                        
            v1_nii = -1.*v1_ii  # vec from neighbor nii to ii
            v2_nii = core_linker_dir[n_ii][cln_nii]  # linker_site vec
            dot = np.dot(v1_nii,v2_nii)
            det = np.cross(v1_nii,v2_nii)[2]
            phi_nii = np.arctan2(det, dot)


            xs = np.linspace(positions[ii][0],
                            (positions[ii][0] + v1_ii[0])   ,
                                30)
            ys = np.linspace(positions[ii][1],
                            (positions[ii][1] + v1_ii[1]),
                                30)

            norm = np.linalg.norm(v1_ii)  # norm of vector between cg sites
            normal = np.cross(np.array([0,0,1.]), v1_ii)  # normal vector
            xnew = []; ynew = []
            for x, y in zip(xs, ys):
                lens = np.sqrt((xs[0]-x)**2 + (ys[0]-y)**2)
                disp_vec = normal * w(lens/norm, phi_ii, phi_nii)
                xnew.append(x+disp_vec[0])
                ynew.append(y+disp_vec[1])
            if plot_beam:
                ax.plot(xnew, ynew,
                        color='lightsteelblue', linewidth=10, zorder=-1)

            if plot_neighbor_connections:
                ax.plot(xs,ys, color='blue', zorder=5)
            if plot_linker_sites:
                ax.arrow(positions[ii,0],positions[ii,1], core_linker_dir[ii,cln,0], core_linker_dir[ii,cln,1], 
            color='darkblue', head_width=0.9, head_length=0.5, lw=2, zorder=10)

    if plot_cell:  # simulation cell
        ax.plot([0., cell.array[0,0]], [0., cell.array[0,1]], color='grey')
        ax.plot([0., cell.array[1,0]], [0., cell.array[1,1]], color='grey')
        ax.plot([cell.array[1,0], cell.array[1,0] + cell.array[0,0]], [cell.array[1,1], cell.array[1,1] + cell.array[0,1]], color='grey')
        ax.plot([cell.array[0,0], cell.array[1,0] + cell.array[0,0]], [cell.array[0,1], cell.array[1,1] + cell.array[0,1]], color='grey')

    plt.tight_layout()
    if savefigname:
        plt.savefig(savefigname, dpi=300)     

    return fig, ax



def geom_optimize(cg_atoms, calculator, trajectory=None):
    from ase.constraints import FixedPlane
    from ase.optimize import BFGS
    r0=35.082756/np.sqrt(3)

    cg_atoms.calc = calculator

    # for 2D optimization. Works only with ASE version directly from gitlab
    c = FixedPlane(
        indices=[atom.index for atom in cg_atoms],
        direction=[0, 0, 1],  # only move in xy plane
    )

    cg_atoms.set_constraint(c)

    dyn = BFGS(cg_atoms, trajectory=trajectory)
    dyn.run(fmax=0.01)
    cg_atoms_o = cg_atoms.calc.get_atoms()

    return cg_atoms_o

def geom_optimize_efficient(cg_atoms, calculator, trajectory=None):
    from ase.constraints import FixedPlane
    from ase.optimize import BFGS
    from cgf.surrogate import MikadoRR


    calc = MikadoRR(**calculator.todict())
    cg_atoms.calc = calc

    ### first: only optimize linker_sites
    cg_atoms.calc.parameters.opt = True
    cg_atoms.get_potential_energy()
    cg_atoms_o_ls = cg_atoms.calc.get_atoms()
    

    ### second: optimize geometry without optimizing linker sites
    calc = MikadoRR(**calculator.todict())
    cg_atoms_o_ls.calc = calc
    cg_atoms_o_ls.calc.parameters.opt = False
    

    # for 2D optimization. Works only with ASE version directly from gitlab
    c = FixedPlane(
        indices=[atom.index for atom in cg_atoms_o_ls],
        direction=[0, 0, 1],  # only move in xy plane
    )

    cg_atoms_o_ls.set_constraint(c)

    dyn = BFGS(cg_atoms_o_ls, trajectory=trajectory)
    dyn.run(fmax=0.01)
    cg_atoms_o_pos = cg_atoms_o_ls.calc.get_atoms()

    ### thrid: optimize geometry and optimize linker sites
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
    dyn.run(fmax=0.01)
    cg_atoms_o = cg_atoms_o_pos.calc.get_atoms()

    return cg_atoms_o