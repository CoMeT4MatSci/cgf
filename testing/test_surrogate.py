import pytest
import os
import numpy as np
from pathlib import Path
from ase.build import bulk
from cgf.cgatoms import init_cgatoms, read_cgatoms
from cgf.utils import geom_optimize, geom_optimize_efficient, cell_optimize
from cgf.surrogate import MikadoRR

test_data_path = Path('testing/test-data').resolve()


def test_MikadoRR_calc_SP_unit_cell():
    r0=30.082756/np.sqrt(3)

    # create unit-cell
    atoms = bulk('Y', 'hcp', a=30.082756, b=30.082756)
    atoms.positions[0][2] = 15.0 # vacuum in z-dir


    # Single Point without linker-sites optimization, but with reevaluate_topology=False
    # This should give wrong results
    calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
            4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, 
            opt=False, update_linker_sites=True, reevaluate_topology=False)
    cg_atoms = init_cgatoms(atoms.copy(), 2.46, r0=r0, linker_sites='nneighbors')
    cg_atoms.calc = calculator
    assert cg_atoms.get_potential_energy()!=pytest.approx(0)  # this should give the wrong result for such a small unit cell

    # Single Point without linker-sites optimization, but with reevaluate_topology=True
    calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
            4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, 
            opt=False, update_linker_sites=True, reevaluate_topology=True)
    cg_atoms = init_cgatoms(atoms.copy(), 2.46, r0=r0, linker_sites='nneighbors')
    cg_atoms.calc = calculator
    assert cg_atoms.get_potential_energy()==pytest.approx(0, abs=1e-5)

def test_MikadoRR_calc_stress_unit_cell():
    r0=30.082756/np.sqrt(3)

    # create unit-cell and strain
    atoms = bulk('Y', 'hcp', a=30.082756, b=30.082756)
    atoms.positions[0][2] = 15.0 # vacuum in z-dir
    cell = atoms.cell
    cell[:,0] *= 1.1
    cell[:,1] *= 1.1
    atoms.set_cell(cell, scale_atoms=True)

    # Single Point without linker-sites optimization, but with reevaluate_topology=True
    calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
            4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, 
            opt=False, update_linker_sites=True, reevaluate_topology=True)
    cg_atoms = init_cgatoms(atoms.copy(), 2.46, r0=r0, linker_sites='nneighbors')
    cg_atoms.calc = calculator

    assert cg_atoms.get_stress()[0]>1e-3
    assert cg_atoms.get_stress()[1]>1e-3
    assert cg_atoms.get_stress()[2:]==pytest.approx(np.zeros(cg_atoms.get_stress()[2:].shape), abs=1e-4)

def test_MikadoRR_cell_optimize_unit_cell():
    r0=30.082756/np.sqrt(3)

    # isotropic cell-relax
    # create unit-cell and strain
    atoms = bulk('Y', 'hcp', a=30.082756, b=30.082756)
    atoms.positions[0][2] = 15.0 # vacuum in z-dir
    cell = atoms.cell
    cell[0] *= 1.01
    cell[1] *= 1.01
    atoms.set_cell(cell, scale_atoms=True)

    # Single Point without linker-sites optimization, but with reevaluate_topology=True
    calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
            4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, 
            opt=False, update_linker_sites=True, reevaluate_topology=True)
    cg_atoms = init_cgatoms(atoms.copy(), 2.46, r0=r0, linker_sites='nneighbors')
    cg_atoms.calc = calculator
    cg_atoms_o = cell_optimize(cg_atoms, calculator, isotropic=True)

    assert cg_atoms_o.cell.cellpar()[0]==pytest.approx(cg_atoms_o.cell.cellpar()[1])
    assert cg_atoms_o.cell.cellpar()[-1]==pytest.approx(120.)

    # full cell-relax
    # create unit-cell and strain
    atoms = bulk('Y', 'hcp', a=30.082756, b=30.082756)
    atoms.positions[0][2] = 15.0 # vacuum in z-dir
    cell = atoms.cell
    cell[0] *= 1.01
    cell[1] *= 1.02
    atoms.set_cell(cell, scale_atoms=True)

    # Single Point without linker-sites optimization, but with reevaluate_topology=True
    calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
            4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, 
            opt=False, update_linker_sites=True, reevaluate_topology=True)
    cg_atoms = init_cgatoms(atoms.copy(), 2.46, r0=r0, linker_sites='nneighbors')
    cg_atoms.calc = calculator
    cg_atoms_o = cell_optimize(cg_atoms, calculator, isotropic=True)
#     cg_atoms_o = cell_optimize(cg_atoms_o, calculator, isotropic=False)  # somehow cell collapes weirdly with non-isotropic relaxation and causes problems?

    assert cg_atoms_o.cell.cellpar()[0]<np.linalg.norm(cell[0])
    assert cg_atoms_o.cell.cellpar()[1]<np.linalg.norm(cell[1])
#     assert cg_atoms_o.get_stress()==pytest.approx(np.zeros(cg_atoms_o.get_stress().shape), abs=1e-6)

def test_MikadoRR_calc_SP_super_cell():
    r0=30.082756/np.sqrt(3)

    # create unit-cell
    atoms = bulk('Y', 'hcp', a=30.082756, b=30.082756)
    atoms.positions[0][2] = 15.0 # vacuum in z-dir

    # create supercell
    atoms_supercell = atoms * (3, 3, 1)

    # Single Point without linker-sites optimization, but with reevaluate_topology=False
    # This should give the correct result because cell is big enough
    calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
            4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, 
            opt=False, update_linker_sites=True, reevaluate_topology=False)
    cg_atoms = init_cgatoms(atoms_supercell.copy(), 2.46, r0=r0, linker_sites='nneighbors')
    cg_atoms.calc = calculator
    E_no_reevaluate = cg_atoms.get_potential_energy()

    # Single Point without linker-sites optimization, but with reevaluate_topology=True
    calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
            4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, 
            opt=False, update_linker_sites=True, reevaluate_topology=True)
    cg_atoms = init_cgatoms(atoms_supercell.copy(), 2.46, r0=r0, linker_sites='nneighbors')
    cg_atoms.calc = calculator
    E_with_reevaluate = cg_atoms.get_potential_energy()
    assert E_with_reevaluate==pytest.approx(E_no_reevaluate, abs=1e-4)==pytest.approx(0, abs=1e-4)  # compare with and without reevalulate

def test_MikadoRR_calc_SW(tmp_path):

    os.chdir(tmp_path)

    r0=35.082756/np.sqrt(3)

    calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
            4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=True, update_linker_sites=True)

    # SW single point

    cg_atoms = read_cgatoms(Path(test_data_path/'COF-5_opt_SW_cg.json'))
    calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
            4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=False, update_linker_sites=True)
    cg_atoms.calc = calculator
    assert cg_atoms.get_potential_energy()==pytest.approx(12.753976338633947)

    # SW optimization of linker_sites

    cg_atoms = read_cgatoms(Path(test_data_path/'COF-5_opt_SW_cg.json'))
    calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
            4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=True, update_linker_sites=True, **{'gtol': 1e-3})
    cg_atoms.calc = calculator
    assert cg_atoms.get_potential_energy()==pytest.approx(11.982403)

    # SW optimization of sites

    cg_atoms = read_cgatoms(Path(test_data_path/'COF-5_opt_SW_cg.json'))
    calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
            4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=True, update_linker_sites=True)
    cg_atoms_o = geom_optimize(cg_atoms, calculator, trajectory='traj.traj')

    cg_atoms = read_cgatoms(Path(test_data_path/'COF-5_opt_SW_cg.json'))
    calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
            4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=True, update_linker_sites=True)
    cg_atoms_o_eff = geom_optimize_efficient(cg_atoms, calculator, trajectory='traj_eff.traj')
    
    # check if the two geometry optimization methods yield same result
    assert cg_atoms_o_eff.get_potential_energy()==pytest.approx(cg_atoms_o.get_potential_energy(), abs=1e-4)==pytest.approx(11.530553279106243, abs=1e-4)
