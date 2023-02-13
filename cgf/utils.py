from ase import Atoms

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

def plot_cgatoms(cg_atoms, fig=None, ax=None, savefigname=None):
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

    ax.scatter(positions[:,0],positions[:,1], color='coral', marker='o', s=75, zorder=10)

    colors = ['red', 'green', 'purple']
    #for i in range(core_linker_dir.shape[0]):
    #    for li in range(core_linker_dir.shape[1]):
    #        ax.arrow(positions[i,0],positions[i,1], core_linker_dir[i,li,0], core_linker_dir[i,li,1], 
    #        color=colors[core_linker_neigh[i][li]], head_width=0.9, head_length=0.5, zorder=10)
    ns = []
    for ii in range(natoms):
        neighbors = neigh_ids[ii]
        ns.append(ii)
        for jj in range(len(neighbors)):
            if neighbors[jj] in ns:
                linestyle = ':'
            else:
                linestyle='--'

            cln = core_linker_neigh[ii][jj]
            #ax.plot([positions[ii][0]+core_linker_dir[ii,cln,0], positions[neighbors[jj]][0]],
            #        [positions[ii][1]+core_linker_dir[ii,cln,1], positions[neighbors[jj]][1]],
            #        color=colors[jj], linestyle=linestyle, linewidth=3, alpha=0.5)
            ax.arrow(positions[ii][0]+core_linker_dir[ii,cln,0], positions[ii][1]+core_linker_dir[ii,cln,1],
                    neigh_dist_vec[ii][jj][0]-core_linker_dir[ii,cln,0], neigh_dist_vec[ii][jj][1]-core_linker_dir[ii,cln,1],
                    color=colors[jj], linestyle=linestyle, linewidth=3, alpha=0.5,
                    head_width=0.9, head_length=0.5, zorder=10)

            ax.arrow(positions[ii,0],positions[ii,1], core_linker_dir[ii,cln,0], core_linker_dir[ii,cln,1], 
            color=colors[jj], head_width=0.9, head_length=0.5, zorder=10)
            
            #ax.annotate(str(neighbors[jj])+"-"+str(jj), [positions[ii][0]+neigh_dist_vec[ii][jj][0]*2/3, positions[ii][1]+neigh_dist_vec[ii][jj][1]*2/3])
            #ax.annotate(str(jj), [positions[ii][0]+core_linker_dir[ii,cln,0], positions[ii][1]+core_linker_dir[ii,cln,1]])

    plt.tight_layout()
    if savefigname:
        plt.savefig(savefigname, dpi=300)     

    return fig, ax
