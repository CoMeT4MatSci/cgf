from ase.build import bulk, make_supercell
from ase.neighborlist import NeighborList
from ase.io import read
from ase import Atom, Atoms
import numpy as np

def generate_SW_defect(reference_cell, supercell_size=(3,3,1)):
    """Gemerates Stone-Wales defect based on a honeycomb lattice by rotating two neighboring atoms by 90 deg


    """
    if isinstance(supercell_size, int):
        supercell_size = (supercell_size, supercell_size, 1)
    if supercell_size[0]<3 or  supercell_size[1]<3:
        raise "Supercell must be at least 3x3 for a SW defect"

    # create unit-cell
    cg_SW = bulk('Y', 'hcp', a=3, b=3)
    cg_SW.positions[0][2] = 1.5 # vacuum in z-dir
    cg_SW.positions[1][2] = 1.5 # vacuum in z-dir
    cg_SW *= supercell_size

    rot_atoms = cg_SW[[0,1]]
    vec01 = rot_atoms.positions[1] - rot_atoms.positions[0]
    rot_atoms.rotate(90, v='z', center=vec01/2 + rot_atoms.positions[0], rotate_cell=False)


    del cg_SW[[0, 1]]
    cg_SW = cg_SW + rot_atoms
    reference_cell[0] *= supercell_size[0]
    reference_cell[1] *= supercell_size[1]

    cg_SW.set_cell(reference_cell, scale_atoms=True)

    # shift SW to center for nicer visuals
    shift_vec = reference_cell[0]/2+reference_cell[1]/2 - (vec01 - rot_atoms.positions[0])
    cg_SW.translate(shift_vec)
    cg_SW.wrap()

    return cg_SW

def generate_585_defect(reference_cell, supercell_size=(3,3,1)):
    """Gemerates 585 defect based on a honeycomb lattice by 
    deleting two neighboring atoms and shifting some other atoms a bit
    """
    if isinstance(supercell_size, int):
        supercell_size = (supercell_size, supercell_size, 1)
    if supercell_size[0]<3 or  supercell_size[1]<3:
        raise "Supercell must be at least 3x3 for a 585 defect"

    # create unit-cell
    cg_585 = bulk('Y', 'hcp', a=3, b=3)
    cg_585.positions[0][2] = 1.5 # vacuum in z-dir
    cg_585.positions[1][2] = 1.5 # vacuum in z-dir
    cg_585 *= supercell_size


    nl = NeighborList([0.8]*len(cg_585), self_interaction=False, bothways=True)
    nl.update(cg_585)

    # atoms 0 and 1 will be deleted eventually

    neigh_atoms0, offsets0 = nl.get_neighbors(0)
    neigh_atoms1, offsets1 = nl.get_neighbors(1)
    neigh_atoms0 = neigh_atoms0[neigh_atoms0!=1]
    neigh_atoms1 = neigh_atoms1[neigh_atoms1!=0]


    # shift 585 to center for nicer visuals and to avoid pbc issues
    vec01 = cg_585[1].position - cg_585[0].position
    pos0 = cg_585[0].position
    shift_vec = cg_585.cell[0]/2+cg_585.cell[1]/2 - (vec01 - pos0)
    cg_585.translate(shift_vec)
    cg_585.wrap()

    # shifting neighboring atoms to 0 and 1 a bit towards each other
    vec0 = cg_585[neigh_atoms0[0]].position  - cg_585[neigh_atoms0[1]].position 
    vec1 = cg_585[neigh_atoms1[0]].position  - cg_585[neigh_atoms1[1]].position

    cg_585[neigh_atoms0[0]].position -= vec0*0.15
    cg_585[neigh_atoms0[1]].position += vec0*0.15
    cg_585[neigh_atoms1[0]].position -= vec1*0.15
    cg_585[neigh_atoms1[1]].position += vec1*0.15

    del cg_585[[0, 1]]  # delete 0 and 1

    # scale cell    
    reference_cell[0] *= supercell_size[0]
    reference_cell[1] *= supercell_size[1]
    cg_585.set_cell(reference_cell, scale_atoms=True)

    return cg_585

def generate_558_GB1(reference_cell, y_super=1):
    """See: doi: 10.1038/srep11744 , GB D1"""
    x = np.array([0.50000000,
                0.71513960,
                0.58755514,
                0.68834372,
                1.50246679,
                2.00532665,
                2.02095542,
                2.90256167,
                3.12682013,
                3.34108498,
                4.09857685,
                4.13611474,
                4.45952833,
                4.50103834,
                5.43975332,
                5.51233397,
                5.71044013,
                5.72747357,
                6.46641366,
                6.93872971,
                7.00725129,
                7.85956958,
                8.10714286,
                8.34291372,
                9.07348993,
                9.13510436,
                9.42412578,
                9.47186229])
    y = np.array([2.57943530,
                6.37503517,
                4.84474297,
                1.09156987,
                3.54098793,
                0.40000000,
                7.21206958,
                2.80323955,
                1.40488457,
                6.51827048,
                5.21487707,
                3.68209348,
                0.40840607,
                7.17581915,
                5.41473299,
                3.05081501,
                1.50826705,
                6.78338598,
                4.35714335,
                7.30269173,
                0.44633272,
                5.03050668,
                6.50081497,
                1.46808714,
                2.83688139,
                4.37021260,
                7.20219809,
                0.50825443])
    pos = np.array([x, y, np.zeros(len(x))]).T
    # d1 = read('D1_gb.xyz')
    d1 = Atoms(['Y'] * len(pos), positions=pos, cell=np.array([[9.97186229, 0.0, 0.0], 
                                                               [0.0, 7.75, 0.0],
                                                               [0.0, 0.0, 30.0]]), pbc=True)
    
    # d1.set_chemical_symbols('Y'*len(d1))
    P = np.array([[4, 0, 0], [2, 4, 0], [0, 0, 1]])
    d1.set_cell(P@reference_cell, scale_atoms=True)
    d1.positions[:,2] = 1.5

    if y_super>0:
        # create unit-cell
        p = bulk('Y', 'hcp', a=3, b=3)
        p.positions[0][2] = 1.5 # vacuum in z-dir
        p.positions[1][2] = 1.5 # vacuum in z-dir
        P = np.array([[4, 0, 0], [1, 2, 0], [0, 0, 1]])

        p = make_supercell(p, P)
        p.translate([-2/3, -p.cell[1][1]/6, 0])
        p.wrap()
        p.set_cell(P@reference_cell, scale_atoms=True)

        p *= (1, y_super, 1)

        
        p.translate(d1.cell[1])

        d1_p = d1 + p
        d1_p_cell = np.array([d1.cell[0], d1.cell[1]+p.cell[1], d1.cell[2]])
        d1_p.set_cell(d1_p_cell)

        d1_p.positions[:,2] = 1.5
        return d1_p
    else:
        return d1
    
def generate_558_GB2(reference_cell, y_super=1):
    """See: https://doi.org/10.1016/j.carbon.2018.08.045 , GB(2,0)|(2,0)"""
    gb = bulk('Y', 'hcp', a=10, b=10)
    gb.positions[0][2] = 1.5 # vacuum in z-dir
    gb.positions[1][2] = 1.5 # vacuum in z-dir
    P = np.array([[2, 0, 0], [1, 2, 0], [0, 0, 1]])
    gb = make_supercell(gb, P)
    cell = gb.cell
    cell[1][1] += cell[1][1]/4
    inds = [a.index for a in gb if np.isclose(a.position[1], np.max(gb.positions[:,1]))]
    gb.positions[inds] += cell[1]/4 - np.array([0, 0.5, 0])
    gb.set_cell(cell)
    gb += Atom('Y', position=[3.4, 2*cell[1][1]/3 - 0.7, 1.5])
    gb += Atom('Y', position=[16.4, 2*cell[1][1]/3 - 0.7, 1.5])
    gb.set_cell(P@reference_cell, scale_atoms=True)
    gb.positions[:,2] = 1.5

    if y_super>0:
        # create unit-cell
        p = bulk('Y', 'hcp', a=3, b=3)
        p.positions[0][2] = 1.5 # vacuum in z-dir
        p.positions[1][2] = 1.5 # vacuum in z-dir
        P = np.array([[2, 0, 0], [1, 2, 0], [0, 0, 1]])

        p = make_supercell(p, P)

        p.set_cell(P@reference_cell, scale_atoms=True)

        p *= (1, y_super, 1)   
        p.translate(gb.cell[1])

        gb_p = gb + p
        gb_p_cell = np.array([gb.cell[0], gb.cell[1]+p.cell[1], gb.cell[2]])
        gb_p.set_cell(gb_p_cell)
        gb_p.positions[:,2] = 1.5

        return gb_p
    else:
        return gb