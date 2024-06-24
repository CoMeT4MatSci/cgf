import os
from pathlib import Path

import numpy as np
import pytest
from ase.io import Trajectory

from cgf.utils.training import (extract_features, get_rc_linkersites_beamfit,
                                train_model)

test_data_path = Path('testing/test-data').resolve()


def test_training_COF5():
    traj = Trajectory(Path(test_data_path, 'traj_training_COF5.traj'))
    structures = []
    energies = []
    for n, atoms in enumerate(traj):
        energies.append(atoms.get_potential_energy())
        structures.append(atoms.copy())

    energies = np.array(energies)
    r0 = structures[0].cell.cellpar()[0]/np.sqrt(3)
    core_descriptors, bond_descriptors = extract_features(structures, r0, 
                                                        get_rc_linkersites_func=get_rc_linkersites_beamfit,
                                                        **{'id_groups': [[30, 40, 50, 25, 45, 35], [10, 60, 20, 65, 15, 55]],
                                                            'r0_beamfit': r0,
                                                            'linkage_length': 3.5})
    assert len(core_descriptors[0])==2
    assert len(core_descriptors)==21
    assert len(bond_descriptors[0])==6
    assert len(bond_descriptors)==21

    training_model, reg = train_model(core_descriptors, bond_descriptors, energies)

    assert np.allclose(np.array(training_model['rr_coeff']), np.array([-44.44196423154224, 1.2805478350206887, 4.228979309801054, 4.228979309830416, 4.228979926706899, 27.101000946607556]))
    assert training_model['rr_incpt']==pytest.approx(385.59437001007376)