import numpy as np

from ase import Atoms
from ase.neighborlist import NeighborList

###############################################################################
import json

def convert(x):
    if hasattr(x, "tolist"):  # numpy arrays have this
        return x.tolist()
    raise TypeError(x)
###############################################################################

def _find_linker_neighbor(cg_atoms, r0, neighborlist=None):
    
    natoms = len(cg_atoms)
    cell = cg_atoms.cell
    positions = cg_atoms.positions

    nl = neighborlist
    if nl==None:
        nl = NeighborList( [1.2*r0/2] * natoms, self_interaction=False, bothways=True)
        nl.update(cg_atoms)

    core_linker_dir = cg_atoms.get_array('linker_sites')
    phi0 = 2*np.pi/core_linker_dir.shape[1]

    core_linker_neigh = []
    # iterate over atoms
    for ii in range(natoms):
        neighbors, offsets = nl.get_neighbors(ii)
        cells = np.dot(offsets, cell)
        distance_vectors = positions[neighbors] + cells - positions[ii]

        linker_neigh = []
        # iterate over neighbors of ii
        for jj in range(len(neighbors)):
            v1 = distance_vectors[jj] # vector from ii to jj
            r1 = np.linalg.norm(v1)

            for li, v2 in enumerate(core_linker_dir[ii]):
                dot = np.dot(v1,v2)
                det = np.cross(v1,v2)[2]
                angle = np.arctan2(det, dot)

                if np.abs(angle) < phi0/2:
                    linker_neigh.append(li)
                    break
        core_linker_neigh.append(linker_neigh)

    cg_atoms.new_array('linker_neighbors', np.array(core_linker_neigh)) # add linker site id for each neighbor
    
    
def init_cgatoms(cg_atoms, linkage_length, r0):
    """
    Initialize a CG ase Atoms object. 
    
    Input:
    cg_atoms: ASE atoms object with core positions and cell
    linkage_length: distance of linkage site from center of core
    r0: cutoff-radius used for finding nearest neighbor cores
    
    Returns:
    ASE atoms object with additional arrays 'linker_sites' and 'linker_neighbors' which are used in CG Models
    """
    natoms = len(cg_atoms)
    cell = cg_atoms.cell
    positions = cg_atoms.positions

    nl = NeighborList( [1.2*r0/2] * natoms, self_interaction=False, bothways=True)
    nl.update(cg_atoms)

    core_linker_dir = []
    core_linker_neigh = []    
    for ii in range(natoms):
        neighbors, offsets = nl.get_neighbors(ii)
        cells = np.dot(offsets, cell)
        distance_vectors = positions[neighbors] + cells - positions[ii]

        linker_dir = []
        linker_neigh = []
        # iterate over neighbors of ii
        for jj in range(len(neighbors)):
            v1 = distance_vectors[jj] # vector from ii to jj
            r1 = np.linalg.norm(v1)
            linker_dir.append(linkage_length/r1*v1) # point towards neighbor with length=linkage_length
            linker_neigh.append(jj) # linkage id == neighbor id

        core_linker_dir.append(linker_dir)
        core_linker_neigh.append(linker_neigh)        

    cg_atoms.new_array('linker_sites', np.array(core_linker_dir)) # add positions of linker sites relative to core center
    cg_atoms.new_array('linker_neighbors', np.array(core_linker_neigh)) # add linker site id for each neighbor
    
    return cg_atoms

def write_cgatoms(cg_atoms, fname):
    """
    Write a CG ase Atoms object to json.
    
    Input:
    cg_atoms: ASE atoms object with core positions, cell and additional arrays 'linker_sites' and 'linker_neighbors'
    fname: filename
    
    """
    
    cga_dict = {'positions': cg_atoms.positions, 'numbers': cg_atoms.numbers, 'cell': cg_atoms.cell.array,
                'linker_sites': cg_atoms.get_array('linker_sites'), 'linker_neighbors': cg_atoms.get_array('linker_neighbors')}
    
    with open(fname, 'w') as outfile:
        json.dump(json.dumps(cga_dict, default=convert), outfile)

        
def read_cgatoms(fname):
    """
    Write a CG ase Atoms object to json.
    
    Input:
    fname: filename
    
    """    
    with open(fname) as jfile:
        data = json.load(jfile)
    data = json.loads(data)
    
    cg_atoms = Atoms(numbers=data['numbers'], positions=data['positions'], cell=np.array(data['cell']), pbc=True) # create coarse-grained representation based on core centers
    cg_atoms.new_array('linker_sites', np.array(data['linker_sites'])) # add positions of linker sites relative to core center
    cg_atoms.new_array('linker_neighbors', np.array(data['linker_neighbors'])) # add linker site id for each neighbor

    return cg_atoms