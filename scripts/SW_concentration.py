from ase.io import read
from ase.constraints import FixedPlane
from ase.optimize import BFGS
from ase.visualize import view
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

r0=30.082756/np.sqrt(3)
calculator = MikadoRR(r0=r0, rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
        4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=True, update_linker_sites=True)

#sups = [3, 4, 5, 6, 8, 10, 15, 20]
sups = [3, 4, 5, 6, 8, 10]

Es = []
f = open("SW_concentration.dat", 'w')
for sup in sups:

    print(f'Calc {sup}x{sup}')
    r0=30.082756/np.sqrt(3)

    s = read(f"../test-data/{sup}x{sup}_1SW_graphene.cif")  # reading in graphene with SW defect

    # scaling s to appropriate Cell-Size of reference COF
    cof5 = read('../test-data/Triphenylene-DB_1phenyl/dftb_bulk_000/geo_end.gen')
    supcof5 = cof5 * (sup, sup, 1)

    s.set_cell(supcof5.cell, scale_atoms=True)
    s.set_pbc([True, True, False])
    #view(s)

    # starting calculation
    start = timer()
    cg_atoms = init_cgatoms(s, 2.46, r0=r0, linker_sites='nneighbors')
    cg_atoms_o = geom_optimize_efficient(cg_atoms, calculator)
    print(f"Etot {sup}x{sup} supercell with 1 SW defect: ", cg_atoms_o.get_potential_energy())
    Es.append(cg_atoms_o.get_potential_energy())
    f.write(f"{sup} \t {cg_atoms_o.get_potential_energy()}\n")
    write_cgatoms(cg_atoms_o, f"../test-data/COF-5_opt_{sup}x{sup}_1SW_cg.json")
    end = timer()
    print('-- finished (%4.1f s).' % (end-start))
    fig, ax = plot_cgatoms(cg_atoms_o)
    #fig.savefig(f"../test-data/COF-5_opt_{sup}x{sup}_1SW_cg.png", dpi=400)
    plt.show()
    plt.clf()
f.close()

sups = np.array(sups)
sups_labels = []
for sup in range(sups[0],sups[-1]+1):
    sups_labels.append(f"{sup}x{sup}")


fig, ax = plt.subplots(figsize=(6,4))
ax.plot(sups, Es, color='blue')
ax.scatter(sups, Es, color='blue')
ax.set_xticks(range(sups[0],sups[-1]+1), sups_labels)
ax.set_xlabel('1 SW Defect in nxn supercell')
ax.set_ylabel('SW Defect energy [eV]')
plt.tight_layout()
#plt.savefig('SW_concentration.png', dpi=500)
plt.show()
