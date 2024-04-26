import random
import numpy as np
from ase import Atoms
from cgf.cycles import cycle_graph, find_cycles
from cgf.motifs import find_unique_motifs
from cgf.surrogate import get_feature_matrix, _get_core_descriptors, _get_bond_descriptors, _get_bonds
from cgf.cgatoms import find_topology, find_linker_neighbors

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error

## PART A: Feature Extraction
# 1. Define Motifs
# 2. Extract Energies and atoms
# 3. find_cycles and create cycle_graph
# 4. find unique motifs
# 5. collect_descriptors

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

    print('number of samples: %d' % (len(bond_descriptors)))
    print('number of linkers: %d' % (len(bond_descriptors[0])))
    print('number of descriptors per linker: %d' % (len(bond_descriptors[0][0])))
    print('number of cores: %d' % (len(core_descriptors[0])))
    print('number of descriptors per core: %d' % (len(core_descriptors[0][0])))
        
    return core_descriptors, bond_descriptors

def get_rc_linkersites_ids(structure, id_groups):
    """A method to obtain rc and the linkersites based on manually selected ids.
    Each group contains the ids of atoms representing the linkersites.
    The core-position is evaluated based on the center of the linkersites.

    Does not work if linkersites are split-between periodic images.

    Args:
        structure (atoms): ASE atoms object
        id_groups (list): List of Lists of atomic ids

    Returns:
        np.array, list: core positions and linker-site vectors for each node
    """
    
    r_cs = []   
    core_linker_dir = []

    for id_group in id_groups:
        r_c = np.average(structure[id_group].get_positions(), axis=0)
        r_cs.append(r_c)
        cld_tmp = []
        for idg in id_group:
            cld_tmp.append(structure.get_positions()[idg] - r_c)
        core_linker_dir.append(cld_tmp)

    return np.array(r_cs), core_linker_dir

## PART B: Learning

def train_model(core_descriptors, bond_descriptors, energies):
    """Train the model based on the core and bond descriptors and
    the respective energies

    Args:
        core_descriptors (list): core descriptors
        bond_descriptors (list): bond descriptors
        energies (list): energies

    Returns:
        dict, Ridge: dictionary containing the model parameters 
                    and the trained Ridge object
    """

    # feature matrix
    X = get_feature_matrix(core_descriptors, bond_descriptors)

    # target values
    y = energies-energies.min()

    # Ridge Regression with Cross Validation
    reg = RidgeCV(alphas=[1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12]).fit(X, y)
    reg.score(X, y)
    alpha = reg.alpha_
    print('reg alpha: ', reg.alpha_)
    print('reg score: ', reg.score(X, y))

    # choose best alpha and then do regular ridge regression
    reg = Ridge(alpha=reg.alpha_).fit(X, y)

    print('reg score with alpha: ', reg.score(X, y))

    print('mean squared error: ', mean_squared_error(reg.predict(X), y))
    print('reg coreff: ', reg.coef_)
    print('unnormalized reg intercept: ', reg.intercept_)

    training_model = {'rr_coeff': list(reg.coef_),
                      'rr_incpt': reg.intercept_/len(bond_descriptors[0]),
                      'rr_alpha': alpha,
                      'rr_score': reg.score(X, y),
                      'n_samples': len(bond_descriptors),
                      'MSE': mean_squared_error(reg.predict(X), y)}


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