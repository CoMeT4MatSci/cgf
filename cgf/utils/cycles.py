from timeit import default_timer as timer

import numpy as np
from ase.data import covalent_radii

try:
    import networkx as nx
    OPTIONAL_PACKAGE_AVAILABLE = True
except ImportError:
    OPTIONAL_PACKAGE_AVAILABLE = False

def get_AC(atoms, covalent_factor=1.3, pbc=False):
    """
    Generate adjacent matrix from atoms and coordinates. (adapted from xyz2mol)

    AC is a (num_atoms, num_atoms) matrix with 1 representing a covalent bond and 0 being none


    covalent_factor - 1.3 is an arbitrary factor

    args:
        atoms - ase Atoms object

    optional
        covalent_factor - increase covalent bond length threshold with factor

    returns:
        AC - adjacent matrix

    """

    # Calculate distance matrix
    dist_matrix = atoms.get_all_distances(mic=pbc)

    # Calculate adjacency matrix
    atoms_num = atoms.get_atomic_numbers()
    num_atoms = len(atoms)
    AC = np.zeros((num_atoms, num_atoms), dtype=int)
    for i in range(num_atoms):
        a_i = atoms_num[i]
        Rcov_i = covalent_radii[a_i] * covalent_factor
        for j in range(i + 1, num_atoms):
            a_j = atoms_num[j]
            Rcov_j = covalent_radii[a_j] * covalent_factor
            if dist_matrix[i, j] <= Rcov_i + Rcov_j:
                AC[i, j] = 1
                AC[j, i] = 1

    return AC

    
def find_cycles(s, max_cycle_len=6):
    if not OPTIONAL_PACKAGE_AVAILABLE:
        raise ImportError("The 'networkx' package is required for this function. Please install it to proceed.")
    print('--- start  graph construction ...')
    start = timer()    
    AC = get_AC(s, covalent_factor=1.3)    
    #G = nx.from_numpy_matrix(AC)
    G = nx.from_numpy_array(AC)  # necessary for networkx>3.0
    end = timer()
    print('--- finished (%4.1f s).' % (end-start))

    print('--- start minimum_cycle_basis ...')
    start = timer()        
    cy = nx.minimum_cycle_basis(G)
    end = timer()
    print('--- finished (%4.1f s).' % (end-start))

    # this should also work
    # G_dir = G.to_directed()

    # cy = []
    # for c in nx.simple_cycles(G_dir):
    #     if len(c) > 2:
    #         cy.append(c)

    print('--- select cycles ...')
    start = timer()            
    
    # find edges not in a cycle of length less than max_cycle_len
    # and treat them as "2 node cycles"
    cy_flat = [c for c in cy if len(c) <= max_cycle_len]
    print(len(cy_flat))
    for e in G.edges:
        in_cycle = False
        for c in cy_flat:
            if (e[0] in c) and (e[1] in c):
                # print('edge (%d, %d) in cycle' % e)
                in_cycle = True
                break
            # else:
            #     print('edge (%d, %d) not in a cycle' % e)
        if not in_cycle:
            # print('edge (%d, %d) not in a cycle' % e)        
            cy.append([e[0], e[1]])
    end = timer()
    print('--- finished (%4.1f s).' % (end-start))
            
    return cy

def cycle_graph(cy, positions, max_cycle_len=6):
    # construct a graph from a list of cycles (cy)
    # - Each node in the new graph is a cycle. 
    # - The respective nodes from the bigger graph are stored in the attribute 'cycle'. 
    # - Also the (real-space) positions are stored in 'pos'.
    if not OPTIONAL_PACKAGE_AVAILABLE:
        raise ImportError("The 'networkx' package is required for this function. Please install it to proceed.")
    G_cy = nx.Graph()
    for i in np.arange(len(cy)):
        if (len(cy[i]) <= max_cycle_len):
            G_cy.add_node(i, cycle=cy[i], pos=positions[cy[i],:])

    # - An edge is added if two cycles share (a) node(s)
    for i in np.arange(len(cy)):
        if (len(cy[i]) <= max_cycle_len):
            for j in np.arange(i+1,len(cy)):
                if (len(cy[j]) <= max_cycle_len) and (len(set(cy[i]) & set(cy[j])) != 0):
                    G_cy.add_edge(i,j)
    return G_cy
