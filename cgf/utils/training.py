import random
import warnings

import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList
from scipy.optimize import minimize
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error

from cgf.cgatoms import find_linker_neighbors, find_topology, init_cgatoms
from cgf.utils.redecorate import w
from cgf.models.surrogate import (_get_bond_descriptors, _get_bonds,
                           _get_core_descriptors, get_feature_matrix)
from cgf.utils.geometry import remove_hatoms, rot_ar_z

## PART A: Feature Extraction

def extract_features(structures, r0, get_rc_linkersites_func, **kwargs):
    """Extraction of features from a list of atoms objects

    Args:
        structures (list): list of ASE atoms objects
        r0 (float): Approximate distance between nodes
        get_rc_linkersites_func (function): function to determine rc and the linkersites


    Returns:
        list, list: core_ descriptors and bond_descriptors
    """
    core_descriptors = []
    bond_descriptors = []
    for s0 in structures:
       
        r_c, core_linker_dir = get_rc_linkersites_func(structure=s0, **kwargs)

        cg_atoms = Atoms(['Y'] * len(r_c), positions=r_c, cell=s0.cell, pbc=True) # create coarse-grained representation based on core centers
        cg_atoms.new_array('linker_sites', np.array(core_linker_dir)) # add positions of linker sites relative to core center

        cg_atoms = find_topology(cg_atoms, r0)
        cg_atoms = find_linker_neighbors(cg_atoms)

        bonds = _get_bonds(cg_atoms)
        bond_desc, bond_params, bond_ref = _get_bond_descriptors(cg_atoms, bonds)
        bond_descriptors.append(bond_desc)

        core_desc = _get_core_descriptors(cg_atoms)
        core_descriptors.append(core_desc)

        
    return core_descriptors, bond_descriptors

def get_rc_linkersites_ids(structure, id_groups):
    """A method to obtain rc and the linkersites based on manually selected ids.
    Each group contains the group of ids of atoms representing the linkersites.
    The linkersite is obtained from the center of the ids of atoms of this linkersites
    The core-position is evaluated based on the center of the linkersites of one core.

    Does not work if linkersites are split-between periodic images.

    Args:
        structure (atoms): ASE atoms object
        id_groups (list): List of Lists of Lists of atomic ids

    Returns:
        np.array, list: core positions and linker-site vectors for each node
    """
    
    r_cs = []   
    core_linker_dir = []

    for id_group in id_groups:
        cld_tmp = []
        cld_tmp_pos = []
        for idg in id_group:
            cld_tmp_pos.append(np.average(structure[idg].get_positions(), axis=0))

        r_c = np.average(cld_tmp_pos, axis=0)
        r_cs.append(r_c)

        for cp in cld_tmp_pos:
            cld_tmp.append(cp-r_c)

        core_linker_dir.append(cld_tmp)

    return np.array(r_cs), core_linker_dir

def get_rc_linkersites_graphmotives(structure, mfs, cy):
    """Extracts features based on graph motifs. Works for example with COF-5:

        from cgf.utils.cycles import cycle_graph, find_cycles
        from cgf.utils.motifs import find_unique_motifs

        # Tp Core Motif
        motif = nx.Graph()
        motif.add_edge("A", "B")
        motif.add_edge("C", "B")
        motif.add_edge("D", "B")

        # all hexagons
        motif.nodes['A']['cl'] = 6
        motif.nodes['B']['cl'] = 6
        motif.nodes['C']['cl'] = 6
        motif.nodes['D']['cl'] = 6

        # get coeffs with all available structures

        r0 = structures[0].cell.cellpar()[0]/np.sqrt(3)

        cy = find_cycles(structures_noH[0])
        G_cy = cycle_graph(cy, structures_noH[0].positions)

        # annotate cycles with cycle length
        for n in G_cy.nodes:
            G_cy.nodes[n]['cl'] = len(G_cy.nodes[n]['cycle'])

        mfs = find_unique_motifs(motif, G_cy)

    Args:
        structure (Atoms): ASE atoms object
        mfs (list): unique motifs
        cy (list): list of cycle graphs

    Returns:
        _type_: _description_
    """
    from cgf.utils.cycles import cycle_graph
    
    G_cy = cycle_graph(cy, structure.positions)
    r_c = np.array([G_cy.nodes[m['B']]['pos'].mean(axis=0) for m in mfs]) # compute core centers
    core_linker_dir = [[G_cy.nodes[m[ls]]['pos'].mean(axis=0)-G_cy.nodes[m['B']]['pos'].mean(axis=0) for ls in ['A', 'C', 'D']] for m in mfs]

    return r_c, core_linker_dir

def get_rc_linkersites_neigh(structure):
    """Extracts core positions and linker-sites based on
    which atoms have three neighboring atoms. Only works for
    very specific structures

    Args:
        structure (Atoms): ASE atoms object

    Returns:
        r_c, core_linker_dir
    """
    from ase.neighborlist import (NeighborList, NewPrimitiveNeighborList,
                                  natural_cutoffs)

    nl = NeighborList(natural_cutoffs(structure), self_interaction=False, bothways=True, 
                          primitive=NewPrimitiveNeighborList,
                          )
    nl.update(structure)
    r_c = []
    core_linker_dir = []
    for i in range(len(structure)):
        neigh_ids, offsets = nl.get_neighbors(i)
        if len(neigh_ids)==3:
            r_c.append(structure[i].position)
            core_linker_dir_tmp = []
            for ni in neigh_ids:
                core_linker_dir_tmp.append(structure[ni].position-structure[i].position)
            core_linker_dir.append(core_linker_dir_tmp)
    
    r_c = np.array(r_c)

    return r_c, core_linker_dir

def get_rc_linkersites_beamfit(structure, id_groups, r0_beamfit, linkage_length, width=4.):
    """Extracts positions of cores rc and the respective linkersites from a structure.
       Fits an optimal elastic beam between cores.

    Args:
        structure (Atoms): ASE atoms object
        id_groups (list): list of lists. rc is calculated based on 
                            the center of position of each id_group. 
                            Does not work if id_group is split between periodic images.
        r0_beamfit (float): approx. distance between core sites
        linkage_length (float): distance of linkage site to core-site. Very important for training.
        width (float, optional): Width of corridor between core-sites for the fitting of
                                the elastic beam. Defaults to 4.
    """
    def is_inside(r_c1, v1_ii, pos, atoms, width=4, linkage_length=1.0):
        def area_triangle(p1, p2, p3):
        # Function to calculate the area of a triangle formed by three points
            return abs((p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2.0)

        v1_ii_unit = v1_ii[:2]/np.linalg.norm(v1_ii[:2])
        normal_unit = np.array([-v1_ii_unit[1], v1_ii_unit[0]])

        p = normal_unit*width/2
        q = v1_ii_unit * linkage_length
        A = r_c1[:2] + p + q
        B = r_c1[:2] - p + q
        C = r_c1[:2] + v1_ii[:2] - p - q
        D = r_c1[:2] + v1_ii[:2] + p - q

        M = pos[:2]

        # Calculate the area of rectangle ABCD
        area_ABCD = area_triangle(A, B, C) + area_triangle(C, D, A)

        # Calculate the sum of areas of triangles ABM, BCM, CDM, DAM
        area_AB_M = area_triangle(A, B, M)
        area_BC_M = area_triangle(B, C, M)
        area_CD_M = area_triangle(C, D, M)
        area_DA_M = area_triangle(D, A, M)

        # If the sum of the areas of triangles equals the area of ABCD, M is inside the rectangle
        return np.isclose(area_ABCD,area_AB_M + area_BC_M + area_CD_M + area_DA_M)


    def calc_mindistsq(phis, r_c1, r_c2, atoms, linkage_length):
        # calculates the sum of minimum square distances of atoms to beam

        phi_ii, phi_nii = phis
        vec_rc = r_c2 - r_c1  # vec between two cores
        norm_rc = np.linalg.norm(vec_rc)
        
        # generate positions of the linkage sites based on phi and linkage_length
        linkage_site1 = r_c1 + rot_ar_z(phi_ii) @ vec_rc / norm_rc * linkage_length
        linkage_site2 = r_c2 + rot_ar_z(phi_nii) @ -vec_rc / norm_rc * linkage_length

        # make beam between linkage sites of the two cores
        xs = np.linspace(linkage_site1[0], linkage_site2[0], 100)
        ys = np.linspace(linkage_site1[1], linkage_site2[1], 100)
        normal = np.cross(np.array([0, 0, 1.]), linkage_site2 - linkage_site1)
        norm = np.linalg.norm(linkage_site2 - linkage_site1)
        xbeam = np.zeros(len(xs)); ybeam = np.zeros(len(ys))
        for n, xy in enumerate(zip(xs, ys)):
            x, y = xy
            lens = np.sqrt((xs[0]-x)**2 + (ys[0]-y)**2)
            disp_vec = w(lens/norm, phi_ii, phi_nii) * normal
            xbeam[n] = x+disp_vec[0]
            ybeam[n] = y+disp_vec[1]

        
        # calculate the squared distance of atomic position to beam
        dsq = 0
        for p in atoms.get_positions():
            dx = p[0]-xbeam
            dy = p[1]-ybeam 
            dsq += (dx**2 + dy**2).min()
        return dsq

    # check r0
    if 1.2*r0_beamfit>=structure.cell.cellpar()[2]:
        warnings.warn('r0*1.2 is larger than c-vector. Expanding cell along c...')
        cellnew = structure.cell
        cellnew[2] = 2*r0_beamfit
        structure.set_cell(cellnew)
    
    # extract core positions based center of position of id_groups
    r_cs = []   
    for id_group in id_groups:
        r_c = np.average(structure[id_group].get_positions(), axis=0)
        r_cs.append(r_c)
    # initlialize cg_atoms based on r_cs
    cg_atoms = Atoms(['Y'] * len(r_cs), positions=r_cs, cell=structure.cell, pbc=True)
    cg_atoms = init_cgatoms(cg_atoms=cg_atoms, r0=r0_beamfit, linkage_length=linkage_length)
    neigh_ids = cg_atoms.get_array('neighbor_ids')
    positions_cg = cg_atoms.positions

    # initialize neighborlist. Important for offset
    nl = NeighborList([1.2*r0_beamfit/2] * len(cg_atoms), self_interaction=False, bothways=True, 
                      primitive=NewPrimitiveNeighborList,
                      )
    nl.update(cg_atoms)
    atoms_noH = remove_hatoms(structure.copy())

    core_linker_dir = np.zeros([len(cg_atoms), 3, 3])
    combinations = [] 
    for ii in range(len(cg_atoms)):
        neighbors = neigh_ids[ii]
        neighbors, offsets = nl.get_neighbors(ii)
        
        cells = np.dot(offsets, cg_atoms.cell)
        for jj in range(len(neighbors)):
            if [ii, jj] in combinations:  # no double evaluation
                continue

            v1_ii = positions_cg[neighbors[jj]] + cells[jj] - positions_cg[ii]  # vec from core to neighbor

            # if neighboring core is in neighboring unit-cell
            if not np.array_equal(cells[jj], np.zeros(3)):
                atoms_noH1 = atoms_noH.copy()
                atoms_noH2 = atoms_noH.copy()
                atoms_noH2.translate([cells[jj]])
                atoms_noH = atoms_noH1 + atoms_noH2

            # select atoms in a corridor between the two cores
            atoms_tmp = Atoms()
            for at in atoms_noH:
                r_c1 = cg_atoms[ii].position
                r_c2 = cg_atoms[neighbors[jj]].position + cells[jj]

                if is_inside(r_c1, v1_ii, at.position, atoms_noH, width=width, linkage_length=linkage_length):
                    atoms_tmp.append(at)

            # find optimum angle of linker-site of core and its neighbor
            res = minimize(calc_mindistsq, [0, 0], 
                    args=(r_c1, r_c2, atoms_tmp, linkage_length),
                    options={'gtol': 1e-4, #'disp': True
                             })
            phi_ii, phi_nii = res.x

            # create core_linker_dir for core and neighbor
            core_linker_dir[ii][jj] = rot_ar_z(phi_ii) @ (v1_ii/np.linalg.norm(v1_ii))*linkage_length
            core_linker_dir[neighbors[jj]][jj] = rot_ar_z(phi_nii) @ (-v1_ii/np.linalg.norm(v1_ii))*linkage_length
            
            # keep track of combinations to avoid double evaluation
            combinations.append([ii, jj])
            combinations.append([neighbors[jj], jj])

    cg_atoms.set_array('linker_sites', np.array(core_linker_dir))

    return np.array(r_cs), core_linker_dir

## PART B: Learning

def train_model(core_descriptors, bond_descriptors, energies):
    """Train the model based on the core and bond descriptors and
    the respective energies via RidgeCV

    Args:
        core_descriptors (list): core descriptors
        bond_descriptors (list): bond descriptors
        energies (list): energies

    Returns:
        dict, Ridge: dictionary containing the model parameters 
                    and error scores
    """

    # feature matrix
    X = get_feature_matrix(core_descriptors, bond_descriptors)

    # target values
    y = energies-energies.min()

    reg = RidgeCV(alphas=[1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12], cv=int(len(y)/3)).fit(X, y)
    alpha = reg.alpha_
    cv_score_mean = reg.best_score_

    reg = Ridge(alpha=reg.alpha_).fit(X, y)

    training_model = {
                      'rr_coeff': list(reg.coef_),
                      'rr_incpt': reg.intercept_/len(bond_descriptors[0]),
                      'rr_alpha': alpha,
                      'R2': reg.score(X, y),
                      'MSE': mean_squared_error(reg.predict(X), y),
                      'cross_val_score_mean': cv_score_mean,
                      'n_samples': len(bond_descriptors),
                      }


    return training_model, reg

def get_learning_curve(training_structures, 
                       training_energies,
                       test_structures, 
                       test_energies, 
                       r0,
                       get_rc_linkersites_func, **kwargs):
    """From a training and test set, a learning curve is generated.
    The Mean Square Error is calculated for different number of samples
    from the training structures. 
    Then, we loop from 1 sample to all available training samples and
    train on them respectively. The specific chosen samples are randomized.

    Args:
        training_structures (list): list of ASE atoms to train on
        training_energies (list): list of energies to train on
        test_structures (list): list of ASE atoms to compare to
        test_energies (list): list of energies to compare to
        r0 (float): Approximate distance between nodes
        get_rc_linkersites_func (function): function to determine rc and the linkersites
        **kwargs: arguments for get_rc_linkersites_func

    Returns:
        list, list, list: n_training_structures, MSE_training, MSE_test
    """

    MSE_test = []
    MSE_training = []
    n_training_structures = []
    for n in range(1, len(training_structures)+1, 1):

        ids_train_tmp = [random.randrange(0, len(training_structures)) for _ in range(n)]
        training_structures_tmp = [training_structures[id_training] for id_training in ids_train_tmp]
        training_energies_tmp = [training_energies[id_training] for id_training in ids_train_tmp]
        training_energies_tmp = np.array(training_energies_tmp)


        core_descriptors, bond_descriptors = extract_features(training_structures_tmp, r0, get_rc_linkersites_func, **kwargs)
        training_model, reg = train_model(core_descriptors, bond_descriptors, training_energies_tmp)

        core_descriptors_test, bond_descriptors_test = extract_features(test_structures, r0, get_rc_linkersites_func, **kwargs)
        X_test = get_feature_matrix(core_descriptors_test, bond_descriptors_test)
        y_test = test_energies-training_energies.min()


        print("MSE test", ((reg.predict(X_test)-y_test)**2).mean())
        MSE_test.append(((reg.predict(X_test)-y_test)**2).mean())
        core_descriptors_training, bond_descriptors_training = extract_features(training_structures, r0, get_rc_linkersites_func, **kwargs)
        X_training = get_feature_matrix(core_descriptors_training, bond_descriptors_training)
        y_training = training_energies-training_energies.min()

        print("MSE training", ((reg.predict(X_training)-y_training)**2).mean())
        MSE_training.append(((reg.predict(X_training)-y_training)**2).mean())

        n_training_structures.append(n)

    return n_training_structures, MSE_training, MSE_test

def get_optimal_coeffs(r0, structures, energies, id_groups, width=4):
    linkage_lengths = np.linspace(0.1, r0/2, 30)
    ls = []
    training_models = []
    t_ID = 0
    t_IDs = []
    for l in linkage_lengths:
        print("Linkage length: ", l)
        try:
            core_descriptors, bond_descriptors = extract_features(structures=structures, 
                                                                    r0=r0, 
                                                                    get_rc_linkersites_func=get_rc_linkersites_beamfit, **{'id_groups': id_groups,
                                                                                                                            'r0_beamfit': r0,
                                                                                                                            'linkage_length': l,  
                                                                                                                            'width': width})
            training_model, reg = train_model(core_descriptors, bond_descriptors, energies)
            print('Training successful with cross_val_score_mean:', training_model['cross_val_score_mean'])
            training_model['linkage_length'] = l
            training_model['training_ID'] = t_ID
            training_models.append(training_model)
            ls.append(l)
            t_IDs.append(t_ID)
            t_ID += 1
        except (IndexError, ValueError) as err:  # this can happen for example when it's not possible to find all linker-neighbors due to bad beam-fit
            print('Training failed: ', err)

    scores = [tm['cross_val_score_mean'] for tm in training_models]
    scores, t_IDs = zip(*sorted(zip(scores, t_IDs), reverse=True))

    results = dict()
    results['linkage_lengths'] = ls
    results['training_models'] = training_models
    results['sorted_scores'] = scores
    results['sorted_IDs'] = t_IDs
    results['opt_training_model'] = None
    for t in t_IDs:
        zero_coeffs = [c for c in training_models[t]['rr_coeff'] if np.isclose(c, 0)]
        if len(zero_coeffs)>0:
            continue
        if training_models[t]['rr_coeff'][-1]<0:
            continue
        if training_models[t]['rr_coeff'][2]<0:  # might cause trouble for non-linear linkages?
            continue
        results['opt_training_model'] = training_models[t]
        return results
    return results  # in case no opt_training_model was found