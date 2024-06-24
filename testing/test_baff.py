import os
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

from cgf.cgatoms import init_cgatoms, read_cgatoms
from cgf.models.baff import BAFFPotential
from cgf.utils.geometry import (cell_optimize, geom_optimize)

test_data_path = Path('testing/test-data').resolve()
r0=17.352109548422437
Kbond=2.56038257535498
Kangle=5.1094517403813695

def test_BAFFPotential_calc_SP_unit_cell():

    # create unit-cell
    atoms = bulk('Y', 'hcp', a=r0*np.sqrt(3), b=r0*np.sqrt(3))
    atoms.positions[0][2] = 15.0 # vacuum in z-dir

    # Single Point 
    calculator = BAFFPotential(r0=r0, Kbond=Kbond, Kangle=Kangle)
    cg_atoms = init_cgatoms(atoms.copy(), 2.46, r0=r0, linker_sites='nneighbors')
    cg_atoms.calc = calculator
    assert cg_atoms.get_potential_energy()==pytest.approx(0, abs=1e-5)

def test_BAFFPotential_calc_stress_unit_cell():

    # create unit-cell and strain
    atoms = bulk('Y', 'hcp', a=r0*np.sqrt(3), b=r0*np.sqrt(3))
    atoms.positions[0][2] = 15.0 # vacuum in z-dir
    cell = atoms.cell
    cell[:,0] *= 1.1
    cell[:,1] *= 1.1
    atoms.set_cell(cell, scale_atoms=True)

    # Single Point 
    calculator = BAFFPotential(r0=r0, Kbond=Kbond, Kangle=Kangle)
    cg_atoms = init_cgatoms(atoms.copy(), 2.46, r0=r0, linker_sites='nneighbors')
    cg_atoms.calc = calculator

    assert cg_atoms.get_stress()[0]>1e-3
    assert cg_atoms.get_stress()[1]>1e-3
    assert cg_atoms.get_stress()[2:]==pytest.approx(np.zeros(cg_atoms.get_stress()[2:].shape), abs=1e-4)

def test_BAFFPotential_cell_optimize_unit_cell():

    # isotropic cell-relax
    # create unit-cell and strain
    atoms = bulk('Y', 'hcp', a=r0*np.sqrt(3), b=r0*np.sqrt(3))
    atoms.positions[0][2] = 15.0 # vacuum in z-dir
    cell = atoms.cell
    cell[0] *= 1.01
    cell[1] *= 1.01
    atoms.set_cell(cell, scale_atoms=True)

    calculator = BAFFPotential(r0=r0, Kbond=Kbond, Kangle=Kangle)
    cg_atoms = init_cgatoms(atoms.copy(), 2.46, r0=r0, linker_sites='nneighbors')
    cg_atoms.calc = calculator
    cg_atoms_o = cell_optimize(cg_atoms, calculator, isotropic=True)

    assert cg_atoms_o.cell.cellpar()[0]==pytest.approx(cg_atoms_o.cell.cellpar()[1])
    assert cg_atoms_o.cell.cellpar()[-1]==pytest.approx(120.)

    # full cell-relax
    # create unit-cell and strain
    atoms = bulk('Y', 'hcp', a=r0*np.sqrt(3), b=r0*np.sqrt(3))
    atoms.positions[0][2] = 15.0 # vacuum in z-dir
    cell = atoms.cell
    cell[0] *= 1.01
    cell[1] *= 1.02
    atoms.set_cell(cell, scale_atoms=True)

    calculator = BAFFPotential(r0=r0, Kbond=Kbond, Kangle=Kangle)
    cg_atoms = init_cgatoms(atoms.copy(), 2.46, r0=r0, linker_sites='nneighbors')
    cg_atoms.calc = calculator
    cg_atoms_o = cell_optimize(cg_atoms_o, calculator, isotropic=False)  

    assert cg_atoms_o.cell.cellpar()[0]<np.linalg.norm(cell[0])
    assert cg_atoms_o.cell.cellpar()[1]<np.linalg.norm(cell[1])
    assert cg_atoms_o.get_stress()==pytest.approx(np.zeros(cg_atoms_o.get_stress().shape), abs=1e-6)

def test_BAFFPotential_calc_SP_super_cell():

    # create unit-cell
    atoms = bulk('Y', 'hcp', a=r0*np.sqrt(3), b=r0*np.sqrt(3))
    atoms.positions[0][2] = 15.0 # vacuum in z-dir

    # create supercell
    atoms_supercell = atoms * (3, 3, 1)

    calculator = BAFFPotential(r0=r0, Kbond=Kbond, Kangle=Kangle)
    cg_atoms = init_cgatoms(atoms_supercell.copy(), 2.46, r0=r0, linker_sites='nneighbors')
    cg_atoms.calc = calculator
    E = cg_atoms.get_potential_energy()

    assert E==pytest.approx(0, abs=1e-4)  # compare with and without reevalulate

def test_BAFFPotential_calc_SW(tmp_path):

    os.chdir(tmp_path)

    # SW optimization of sites

    cg_atoms = read_cgatoms(Path(test_data_path/'COF-5_opt_SW_cg.json'))
    calculator = BAFFPotential(r0=r0, Kbond=Kbond, Kangle=Kangle)
    cg_atoms_o = geom_optimize(cg_atoms, calculator)

    assert cg_atoms_o.get_potential_energy()==pytest.approx(12.883830010199036)

