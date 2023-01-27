from ase.io import read
from ase.constraints import FixedPlane
from ase.optimize import BFGS
import sys
sys.path.insert(0, '../')
from cgf.cgatoms import *
from cgf.surrogate import MikadoRR

s = read("5-8-5_carbon.gen")
cof585 = read("COF-5_585_opt.gen")

s.set_cell(cof585.cell, scale_atoms=True)
cg_atoms = init_cgatoms(s, 2.46, r0=35.082756/np.sqrt(3))


calc = MikadoRR(r0=30.082756/np.sqrt(3), rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
        4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=True)

cg_atoms.calc = calc

# for 2D optimization. Works only with ASE version directly from gitlab
c = FixedPlane(
    indices=[atom.index for atom in cg_atoms],
    direction=[0, 0, 1],
)

import matplotlib.pyplot as plt

for n in range(len(cg_atoms)):
    ls = cg_atoms.get_array('linker_sites')[n]
    p = cg_atoms.positions[n][:2]
    for l in ls:
        v = p+l[:2]
        plt.scatter(v[0], v[1], color='red')
    plt.scatter(p[0], p[1], color='black')

plt.show()

cg_atoms.set_constraint(c)


dyn = BFGS(cg_atoms, trajectory='585opt.traj')
dyn.run(fmax=0.01)
