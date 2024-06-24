import time

from ase import Atoms
from ase.constraints import FixedPlane
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS
import numpy as np


def geom_optimize(cg_atoms, calculator, trajectory=None, logfile=None, max_steps=500, fmax=0.01):

    
    starttime = time.time()

    cg_atoms.calc = calculator

    # for 2D optimization
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

    starttime = time.time()

    cg_atoms.calc = calculator

    # for 2D optimization
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

    from cgf.models.surrogate import MikadoRR


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
    

    # for 2D optimization
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


def remove_hatoms(s):
    del s[[atom.symbol == 'H' for atom in s]]
    return s


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
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.rdMolTransforms import (CanonicalizeMol,
                                                TransformConformer)
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