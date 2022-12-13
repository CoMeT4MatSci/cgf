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