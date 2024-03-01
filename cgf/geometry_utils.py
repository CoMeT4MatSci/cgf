from ase.build import bulk

def generate_SW_defect(reference_cell, supercell_size=(3,3,1)):

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