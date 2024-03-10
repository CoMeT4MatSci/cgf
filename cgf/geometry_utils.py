from ase.build import bulk
from ase.neighborlist import NeighborList

def generate_SW_defect(reference_cell, supercell_size=(3,3,1)):
    """Gemerates Stone-Wales defect based on a honeycomb lattice by rotating two neighboring atoms by 90 deg


    """

    if supercell_size[0]<3 or  supercell_size[1]<3:
        raise "Supercell must be at least 3x3 for a SW defect"

    # create unit-cell
    cg_SW = bulk('Y', 'hcp', a=3, b=3)
    cg_SW.positions[0][2] = 1.5 # vacuum in z-dir
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
    if supercell_size[0]<3 or  supercell_size[1]<3:
        raise "Supercell must be at least 3x3 for a 585 defect"

    # create unit-cell
    cg_585 = bulk('Y', 'hcp', a=3, b=3)
    cg_585.positions[0][2] = 1.5 # vacuum in z-dir
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