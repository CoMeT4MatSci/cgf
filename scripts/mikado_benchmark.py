import sys, os
sys.path.insert(0, '../') # adjust path to the base directory of the package

import cProfile as profile

from cgf.surrogate_ref import MikadoRR
from cgf.surrogate import MikadoRR_V2
from cgf.cgatoms import read_cgatoms

import pickle
import numpy as np
from timeit import default_timer as timer


def load_pickle(filename):
    """Loads pickle of an object from file.
    Args:
        filename (str): Filename load from.
    Returns:
        object
    """
    with open(filename, 'rb') as input_file:
        obj = pickle.load(input_file)
    return obj


cg_atoms_V1 = read_cgatoms('../test-data/COF-5_opt_SW_cg.json')

calc = MikadoRR(r0=30.082756/np.sqrt(3), rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
         4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=True)
cg_atoms_V1.calc = calc

print('-- start energy calculation with MikadoRR ...')
start = timer()
E_V1 = cg_atoms_V1.get_potential_energy()

end = timer()
print('-- finished (%4.1f s).' % (end-start))

print('-- start energy calculation with MikadoRR ...')
start = timer()

forces_V1 = cg_atoms_V1.get_forces()

end = timer()
print('-- finished (%4.1f s).' % (end-start))



cg_atoms_V2 = read_cgatoms('../test-data/COF-5_opt_SW_cg.json')

calc = MikadoRR_V2(r0=30.082756/np.sqrt(3), rr_coeff=np.array([-44.4342221, 1.27912546, 4.45880587, 4.45880586,
         4.45880373, 27.369685]), rr_incpt=2315.3320266790165/6, opt=True)
cg_atoms_V2.calc = calc

prof = profile.Profile()
print('-- start energy calculation with MikadoRR_V2 ...')
start = timer()

prof.enable()
E_V2 = cg_atoms_V2.get_potential_energy()
prof.disable()

end = timer()
print('-- finished (%4.1f s).' % (end-start))

prof.dump_stats('bench.stats')

print('-- start force calculation with MikadoRR_V2 ...')
start = timer()

forces_V2 = cg_atoms_V2.get_forces()

end = timer()
print('-- finished (%4.1f s).' % (end-start))

print()
print('-- energy difference ', E_V1 - E_V2)
print('-- force difference ', forces_V1 - forces_V2)