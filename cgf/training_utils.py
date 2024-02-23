import random

import numpy as np
from cgf.cycles import cycle_graph, find_cycles
from cgf.motifs import find_unique_motifs
from cgf.surrogate import collect_descriptors, get_feature_matrix
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error

## PART A: Feature Extraction
# 1. Define Motifs
# 2. Extract Energies and atoms
# 3. find_cycles and create cycle_graph
# 4. find unique motifs
# 5. collect_descriptors

def extract_features(motif, r0, atoms_list):
    """Extracts the core and linker descriptors from a list of
    ASE atoms objects

    Args:
        motif (nx.Graph): For example:
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
        r0 (float): Approximate distance between nodes
        atoms_list (list): list of ASE atoms objects

    Returns:
        list, list: core_descriptors, bond_descriptors
    """
    # find the cycles in the first structure
    # it is assumed that the topology does not change and we can reuse this information

    cy = find_cycles(atoms_list[0])

    G_cy = cycle_graph(cy, atoms_list[0].positions)

    # annotate cycles with cycle length
    for n in G_cy.nodes:
        G_cy.nodes[n]['cl'] = len(G_cy.nodes[n]['cycle'])


    mfs = find_unique_motifs(motif, G_cy)

    core_descriptors, bond_descriptors = collect_descriptors(atoms_list, cy, mfs, r0)
    print('number of samples: %d' % (len(bond_descriptors)))
    print('number of linkers: %d' % (len(bond_descriptors[0])))
    print('number of descriptors per linker: %d' % (len(bond_descriptors[0][0])))
    print('number of cores: %d' % (len(core_descriptors[0])))
    print('number of descriptors per core: %d' % (len(core_descriptors[0][0])))

    return core_descriptors, bond_descriptors


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
                       motif, 
                       r0):
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
        motif (nx.Graph): motif for extract_features
        r0 (float): Approximate distance between nodes

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


        core_descriptors, bond_descriptors = extract_features(motif=motif, atoms_list=training_structures_tmp, r0=r0)
        training_model, reg = train_model(core_descriptors, bond_descriptors, training_energies_tmp)

        core_descriptors_test, bond_descriptors_test = extract_features(motif=motif, atoms_list=test_structures, r0=r0)
        X_test = get_feature_matrix(core_descriptors_test, bond_descriptors_test)
        y_test = test_energies-training_energies.min()


        print("MSE test", ((reg.predict(X_test)-y_test)**2).mean())
        MSE_test.append(((reg.predict(X_test)-y_test)**2).mean())
        core_descriptors_training, bond_descriptors_training = extract_features(motif=motif, atoms_list=training_structures, r0=r0)
        X_training = get_feature_matrix(core_descriptors_training, bond_descriptors_training)
        y_training = training_energies-training_energies.min()

        print("MSE training", ((reg.predict(X_training)-y_training)**2).mean())
        MSE_training.append(((reg.predict(X_training)-y_training)**2).mean())

        n_training_structures.append(n)

    return n_training_structures, MSE_training, MSE_test