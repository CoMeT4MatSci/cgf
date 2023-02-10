import numpy as np

from ase import Atoms
from ase.neighborlist import NeighborList, mic

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

    cg_atoms.set_array('linker_neighbors', np.array(core_linker_neigh)) # add linker site id for each neighbor


def find_topology(cg_atoms, r0):
    """
    finds the topology of cg_atoms, meaning setting the 'neighbor_ids' 
    arrays of cg_atom 
    
    Input:
    cg_atoms: ASE atoms object with core positions and cell
    r0: cutoff-radius used for finding nearest neighbor cores
    
    Returns:
    ASE atoms object with set array 'neighbor_ids'
    """
    natoms = len(cg_atoms)

    nl = NeighborList( [1.2*r0/2] * natoms, self_interaction=False, bothways=True)
    nl.update(cg_atoms)

    neigh_ids = [] 
    for ii in range(natoms):
        neighbors, offsets = nl.get_neighbors(ii)

        neigh_id = []
        # iterate over neighbors of ii
        for jj in range(len(neighbors)):
            neigh_id.append(neighbors[jj])

        neigh_ids.append(neigh_id)     

    cg_atoms.set_array('neighbor_ids', np.array(neigh_ids))

    return cg_atoms


def find_neighbor_distances(cg_atoms):
    """
    Finds the neighbor distances and sets the array 'neighbor_distances' to cg_atoms
    
    cg_atoms must have already 'neighbor_ids'
    Input:
    cg_atoms: ASE atoms object with core positions and cell
    
    Returns:
    ASE atoms object with set array 'neighbor_distances' array
    """
    natoms = len(cg_atoms)
    cell = cg_atoms.cell
    positions = cg_atoms.positions


    neigh_ids = cg_atoms.get_array('neighbor_ids')
    neigh_dist_vec = []    
    for ii in range(natoms):
        neighbors = neigh_ids[ii]
        distance_vectors = mic(positions[neighbors] - positions[ii], cell)  # minimum image convention

        dist_vec = []
        # iterate over neighbors of ii
        for jj in range(len(neighbors)):
            dist_vec.append(distance_vectors[jj])  # vector from ii to jj

        neigh_dist_vec.append(dist_vec)     

    cg_atoms.set_array('neighbor_distances', np.array(neigh_dist_vec)) # distance from each cg_atom to its neighbors

    return cg_atoms

def find_linker_sites_guess(cg_atoms, linkage_length):
    """
    Makes an initial guess for 'linker_sites' 
    based on variable linkage_length and the direction towards the nearest neighbor.

    cg_atoms must have already 'neighbor_ids' and 'neighbor_distances'
    
    Input:
    cg_atoms: ASE atoms object with core positions and cell
    
    Returns:
    ASE atoms object with set array 'neighbor_distances' array
    """

    natoms = len(cg_atoms)
    neigh_ids = cg_atoms.get_array('neighbor_ids')
    neigh_dist_vec = cg_atoms.get_array('neighbor_distances')

    core_linker_dir = []
    for ii in range(natoms):
        neighbors = neigh_ids[ii]

        linker_dir = []
        for jj in range(len(neighbors)):
            v1 = neigh_dist_vec[ii][jj]  # vector from ii to jj
            r1 = np.linalg.norm(v1)
            linker_dir.append(linkage_length/r1*v1) # point towards neighbor with length=linkage_length

        core_linker_dir.append(linker_dir)

    cg_atoms.set_array('linker_sites', np.array(core_linker_dir)) # add positions of linker sites relative to core center

    return cg_atoms

def find_linker_neighbors(cg_atoms):
    """
    Updates the topology of cg_atoms, meaning the 'linker_neighbors' and 
    'neighbor_ids' arrays of cg_atom
    cg_atoms must have already 'linker_sites', 'neighbor_ids' and 'neighbor_distances'
    
    Input:
    cg_atoms: ASE atoms object with core positions and cell
    
    Returns:
    ASE atoms object with updated arrays 'linker_neighbors' and 'neighbor_ids'
    """
    
    natoms = len(cg_atoms)

    core_linker_dir = cg_atoms.get_array('linker_sites')
    neigh_ids = cg_atoms.get_array('neighbor_ids')
    neigh_dist_vec = cg_atoms.get_array('neighbor_distances')

    phi0 = 2*np.pi/core_linker_dir.shape[1]  # angle corridor in which linker site should be located

    core_linker_neigh = []
    # iterate over atoms
    for ii in range(natoms):
        neighbors = neigh_ids[ii]

        linker_neigh = []
        # iterate over neighbors of ii
        for jj in range(len(neighbors)):
            v1 = neigh_dist_vec[ii][jj] # vector from ii to jj

            # assign linker connection points to the respective neighboring sites
            for li, v2 in enumerate(core_linker_dir[ii]):
                dot = np.dot(v1,v2)
                det = np.cross(v1,v2)[2]
                angle = np.arctan2(det, dot)
                
                # if angle between linker site and neighbor small
                # then assign linker site to this neighbor (important for initial guess)
                if np.abs(angle) < phi0/2:  
                    linker_neigh.append(li)
                    break
        core_linker_neigh.append(linker_neigh)

    cg_atoms.set_array('linker_neighbors', np.array(core_linker_neigh)) # add linker site id for each neighbor
    
    return cg_atoms

def init_cgatoms(cg_atoms, linkage_length, r0):
    """
    Initialize a CG ase Atoms object. Makes an initial guess for 'linker_sites' 
    based on variable linkage_length and the direction towards the nearest neighbor
    
    Input:
    cg_atoms: ASE atoms object with core positions and cell
    linkage_length: distance of linkage site from center of core
    r0: cutoff-radius used for finding nearest neighbor cores
    
    Returns:
    ASE atoms object with additional arrays 'linker_sites', 'neighbor_distances', 'neighbor_ids' and 'linker_neighbors' which are used in CG Models
    """

    cg_atoms = find_topology(cg_atoms, r0)  # setting 'neighbor_ids'
    cg_atoms = find_neighbor_distances(cg_atoms)  # setting 'neighbor_distances'
    cg_atoms = find_linker_sites_guess(cg_atoms, linkage_length)  # setting 'linker_sites'
    cg_atoms = find_linker_neighbors(cg_atoms)  # setting 'linker_neighbors'
    
    return cg_atoms

def write_cgatoms(cg_atoms, fname):
    """
    Write a CG ase Atoms object to json.
    
    Input:
    cg_atoms: ASE atoms object with core positions, cell and additional arrays 'linker_sites', 'linker_neighbors', 'neighbor_ids', 'neighbor_distances'.
    fname: filename
    
    """
    
    cga_dict = {'positions': cg_atoms.positions, 'numbers': cg_atoms.numbers, 'cell': cg_atoms.cell.array,
                'linker_sites': cg_atoms.get_array('linker_sites'),
                'linker_neighbors': cg_atoms.get_array('linker_neighbors'),
                'neighbor_ids': cg_atoms.get_array('neighbor_ids'),
                'neighbor_distances': cg_atoms.get_array('neighbor_distances')}
    
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
    
    std_keys = ['numbers', 'positions', 'cell']
    cg_atoms = Atoms(numbers=data['numbers'], positions=data['positions'], cell=np.array(data['cell']), pbc=True) # create coarse-grained representation based on core centers
    
    
    for k in data.keys():
        if k in std_keys:
            continue
        cg_atoms.new_array(k, np.array(data[k]))

    return cg_atoms