from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem.rdMolTransforms import CanonicalizeMol, TransformConformer
from ase import Atoms
import numpy as np
from cgf.bnff import _get_phi0


def mol2Atoms(mol, repr='2D'):
    '''
    Construct ase Atoms object from rdkit mol object.
    Input:
    mol: rdkit mol object
    repr: '2D' or '3D'. '3D' additionally performs an UFF geometry optimization.
    
    Returns:
    ASE atoms object
    '''
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
    x: point(s) along beam
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