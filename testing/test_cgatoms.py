import pytest
import numpy as np
from ase.build import bulk
from cgf.cgatoms import init_cgatoms

def test_init_cgatoms():

    r0=30.082756/np.sqrt(3)

    # create unit-cell
    atoms = bulk('Y', 'hcp', a=30.0827, b=30.0)
    atoms.positions[0][2] = 15.0 # vacuum in z-dir

    cg_atoms = init_cgatoms(atoms.copy(), 2.46, r0=r0, linker_sites='nneighbors')

    assert cg_atoms.get_array('neighbor_ids').shape==(2, 3)  # two sites, each with three neighbors
    assert cg_atoms.get_array('neighbor_ids')[0]==pytest.approx(np.array([1, 1, 1]))  # site 0 should be connected three times to 1
    assert cg_atoms.get_array('neighbor_ids')[1]==pytest.approx(np.array([0, 0, 0]))  # site 1 should be connected three times to 0
    
    assert cg_atoms.get_array('neighbor_distances').shape==(2, 3, 3)  # two sites, each with three neighbors. Each described via a vector towards it
    for i in cg_atoms.get_array('neighbor_distances'):
        for j in i:
            assert np.linalg.norm(j)==pytest.approx(r0, abs=1e-4)  # distance to each neighbor should be r0
            assert j[2]==pytest.approx(0.)  # z-pos should be 0

    assert cg_atoms.get_array('linker_sites').shape==(2, 3, 3)  # two sites, with three linker-sites each, each linkersite having x,y,z coord

    assert cg_atoms.get_array('linker_neighbors').shape(2, 3)
    