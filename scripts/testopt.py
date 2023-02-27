from ase.io import read
from ase.constraints import FixedPlane
from ase.optimize import BFGS
import matplotlib.pyplot as plt
import sys
from copy import deepcopy
from timeit import default_timer as timer

sys.path.insert(0, '../')
from cgf.cgatoms import *
from cgf.surrogate import MikadoRR
from cgf.surrogate_ref import MikadoRR as MikadoRR_V1
from cgf.utils import plot_cgatoms, geom_optimize, geom_optimize_efficient
from cgf.cgatoms import _find_linker_neighbor

r0=35.082756/np.sqrt(3)
calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
        4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=True, update_linker_sites=True)


###### SW defect


print('SW: nneighbors with geom_optimize_efficient')
calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
        4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=True, update_linker_sites=True)
cg_atoms = read_cgatoms('../test-data/COF-5_opt_SW_cg.json')
start = timer()
cg_atoms = init_cgatoms(cg_atoms, 2.46, r0=r0, linker_sites='nneighbors')
cg_atoms_o = geom_optimize_efficient(cg_atoms, calculator)
print("Etot SW: ", cg_atoms_o.get_potential_energy())
end = timer()
print('-- finished (%4.1f s).' % (end-start))

print('SW: best_angles with geom_optimize_efficient')
calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
        4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=True, update_linker_sites=True)
cg_atoms = read_cgatoms('../test-data/COF-5_opt_SW_cg.json')
start = timer()
cg_atoms = init_cgatoms(cg_atoms, 2.46, r0=r0, linker_sites='best_angles')
cg_atoms_o = geom_optimize_efficient(cg_atoms, calculator)
print("Etot SW: ", cg_atoms_o.get_potential_energy())
end = timer()
print('-- finished (%4.1f s).' % (end-start))



#fig, ax = plot_cgatoms(cg_atoms_o)
#plt.show()

###### 585 defect


print('585: nneighbors with geom_optimize_efficient')
s = read("../test-data/5-8-5_carbon.gen")
cof585 = read("../test-data/COF-5_585_opt.gen")
s.set_cell(cof585.cell, scale_atoms=True)
cg_atoms = init_cgatoms(s, 2.46, r0=r0)
cg_atoms.pbc[2] = False
calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
        4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=True, update_linker_sites=True)
start = timer()
cg_atoms = init_cgatoms(cg_atoms, 2.46, r0=r0, linker_sites='nneighbors')
cg_atoms_o = geom_optimize_efficient(cg_atoms, calculator)
print("Etot 585: ", cg_atoms_o.get_potential_energy())
end = timer()
print('-- finished (%4.1f s).' % (end-start))

print('585: best_angles with geom_optimize_efficient')
s = read("../test-data/5-8-5_carbon.gen")
cof585 = read("../test-data/COF-5_585_opt.gen")
s.set_cell(cof585.cell, scale_atoms=True)
cg_atoms = init_cgatoms(s, 2.46, r0=r0)
cg_atoms.pbc[2] = False
calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
        4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=True, update_linker_sites=True)
start = timer()
cg_atoms = init_cgatoms(cg_atoms, 2.46, r0=r0, linker_sites='best_angles')
cg_atoms_o = geom_optimize_efficient(cg_atoms, calculator)
print("Etot 585: ", cg_atoms_o.get_potential_energy())
end = timer()
print('-- finished (%4.1f s).' % (end-start))

print('585: nneighbors with geom_optimize')
s = read("../test-data/5-8-5_carbon.gen")
cof585 = read("../test-data/COF-5_585_opt.gen")
s.set_cell(cof585.cell, scale_atoms=True)
cg_atoms = init_cgatoms(s, 2.46, r0=r0)
cg_atoms.pbc[2] = False
calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
        4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=True, update_linker_sites=True)
start = timer()
cg_atoms = init_cgatoms(cg_atoms, 2.46, r0=r0, linker_sites='nneighbors')
cg_atoms_o = geom_optimize(cg_atoms, calculator)
print("Etot 585: ", cg_atoms_o.get_potential_energy())
end = timer()
print('-- finished (%4.1f s).' % (end-start))

print('585: best_angles with geom_optimize')
s = read("../test-data/5-8-5_carbon.gen")
cof585 = read("../test-data/COF-5_585_opt.gen")
s.set_cell(cof585.cell, scale_atoms=True)
cg_atoms = init_cgatoms(s, 2.46, r0=r0)
cg_atoms.pbc[2] = False
calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
        4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=True, update_linker_sites=True)
start = timer()
cg_atoms = init_cgatoms(cg_atoms, 2.46, r0=r0, linker_sites='best_angles')
cg_atoms_o = geom_optimize(cg_atoms, calculator)
print("Etot 585: ", cg_atoms_o.get_potential_energy())
end = timer()
print('-- finished (%4.1f s).' % (end-start))

#write_cgatoms(cg_atoms_o, '../test-data/COF-5_585_opt_cg.json')

#fig, ax = plot_cgatoms(cg_atoms_o)
#plt.show()


# ###### 585 defect 2x2 supercell

# s = read("../test-data/5-8-5_carbon.gen")
# cof585 = read("../test-data/COF-5_585_opt.gen")

# s.set_cell(cof585.cell, scale_atoms=True)
# ssuper = s * (2,2,1)

# # rattle in 2D
# rng = np.random.RandomState(42)
# positions = ssuper.arrays['positions']
# ssuper.positions[:,:2] = positions[:,:2] + rng.normal(scale=0.1, size=ssuper.positions[:,:2].shape)

# cg_atoms = init_cgatoms(ssuper, 2.46, r0=35.082756/np.sqrt(3))
# cg_atoms.pbc[2] = False


# cg_atoms_o = optimize(cg_atoms, trajectory="585_2x2supercell.traj")


# print("Etot 585 2x2 supercell: ", cg_atoms.get_potential_energy())
# write_cgatoms(cg_atoms_o, '../test-data/COF-5_585_2x2supercell_opt_cg.json')
# fig, ax = plot_cgatoms(cg_atoms_o)
# plt.show()