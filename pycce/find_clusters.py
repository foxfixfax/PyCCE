import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import time
from itertools import combinations
from scipy.sparse import csr_matrix


def make_graph(atoms, r_dipole, r_inner=0, max_size=5000):
    """
    Make a connectivity matrix for bath spins
    @param atoms: ndarray
    ndarray of bath spins (should contain 'xyz' in dtype)
    @param r_dipole: float
        maximum connectivity distance
    @param r_inner: float
        minimum connectivity distance
    @return: csr_matrix
        connectivity matrix
    """
    if atoms.size < max_size:
        dist_matrix = np.linalg.norm(atoms['xyz'][:, np.newaxis, :] - atoms['xyz'][np.newaxis, :, :], axis=-1)
        atoms_within = np.logical_and(dist_matrix < r_dipole, dist_matrix > r_inner)
    else:
        atoms_within = np.zeros((atoms.size, atoms.size), dtype=bool)
        for i, a in enumerate(atoms):
            dist = np.linalg.norm(atoms['xyz'][i:] - a['xyz'], axis=-1)
            atoms_within[i, i:] = (dist < r_dipole) & (dist > r_inner)
    counter = np.count_nonzero(atoms_within)
    try:
        print('Average number of neighbours is {:.1f}'.format(counter / atoms.shape[0]))
    except ZeroDivisionError:
        print('No spins, no neighbours.')

    # Generate sparse matrix contain connectivity
    graph = csr_matrix(atoms_within, dtype=np.bool)

    return graph


# Import from connected components from scipy
def connected_components(csgraph, directed=False, connection='weak', return_labels=True):
    return scipy.sparse.csgraph.connected_components(csgraph, directed=directed, connection=connection,
                                                     return_labels=return_labels)


def find_subclusters(maximum_order, graph, labels, n_components, strong=False):
    """
    Find subclusters from connectivity matrix
    @param maximum_order: int
        Maximum size of the clusters to find
    @param graph: csr_matrix
        connectivity matrix
    @param labels: ndarray
        The length-N array of labels of the connected components.
    @param n_components: int
        The number of connected components
    @param strong: bool
        Whether to find only completely interconnected clusters (default False)
    @return: dict
        dict with keys corresponding to size of the cluster, and value corresponds to ndarray of shape (M, N),
        M is the number of clusters of given size, N is the size of the cluster.
        Each row contains indexes of the bath spins included in the given cluster
    """
    # bool 1D array which is true when given element of graph corresponds to
    # cluster component
    clusters = {}
    for k in range(1, maximum_order + 1):
        clusters[k] = []
    print('Number of disjointed clusters is {}'.format(n_components))
    for component in range(n_components):
        vert_pos = (labels == component)
        vertices = np.nonzero(vert_pos)[0]

        # print('{} cluster contains {} components'.format(component, ncomp))

        # if ncomp <= CCE_order:
        #
        #     clusters[ncomp].append(vertices[np.newaxis, :])
        #
        # else:

        subclusters = {1: vertices[:, np.newaxis]}
        clusters[1].append(vertices[:, np.newaxis])
        if vertices.size >= 2:
            for order in range(2, maximum_order + 1):

                if order == 2:
                    # Retrieve upper right triangle (remove i,j pairs with i>j),
                    # choose only rows corresponding to vertices in the subcluster
                    csrmat = scipy.sparse.triu(graph, k=0, format='csr')[vertices]
                    # Change to coordinate format of matrix
                    coomat = csrmat.tocoo()
                    # rows, col give row and colum indexes, which correspond to
                    # edges of the graph. as we already slised out the rows,
                    # to obtain correct row indexes we need to use vertices array
                    row_ind, col_ind = vertices[coomat.row], coomat.col

                    bonds = np.column_stack([row_ind, col_ind])
                    subclusters[order] = bonds
                    # Check if [1,2] row in a matrix(Nx2):  any(np.equal(a, [1, 2]).all(1))

                else:

                    # General way to compute clusters for any order >= 3
                    # but for simplicity consider CCE4

                    # List of cluster of size 4
                    ltriplets = []

                    # For ith triplet check i+1:N pairs, if one of them contains
                    # one and only one element of jth pair, they form a cluster of 4
                    # There is no need to check the last one, as it would be included
                    # into quartet already if it were to be a part of one
                    for i in range(subclusters[order - 1].shape[0] - 1):

                        # The triplet under study
                        test = subclusters[order - 1][i]

                        # For cluster i,j,k (i>j>k, as all indexes are stored in increasing order)
                        # consider only bonds l, n with l >= i, n >= j without loss of generality
                        testbonds = bonds[np.all(bonds >= test[:2], axis=1)]

                        # cond is an bool 2D array of shape (testbonds.shape[0], test.size)
                        # i.e. number of rows corresponds to number of testbonds,
                        # lenght of the row is equal to the length of the test cluster (3 in case CCE4)
                        # cond[i,j] is True if bond[i] contains element of test[j], otherwise False

                        # To construct this array the following procedure is applied:
                        # Reshape testbonds from (n, 2) to (n, 2, 1)
                        # when asked to do logical operation == testbonds is broadcasted to shape (n, 2, order - 1)
                        # In the case of CCE4 (n, 2, 3). Resulting 3D bool array has True entry i,j,k
                        # If j element of testbonds[i] is equal to k element of test
                        # Applying logical operation any along 2nd axis (axis=1, any element of the bond i)
                        # we obtain resulting array cond

                        cond = np.any(testbonds.reshape(testbonds.shape + (1,)) == test, axis=1)
                        # Check which of testbonds form a cluster with the triplet i,j,k
                        # rows is 1D bool array, rows[i] is True if bond[i] contains exactly 1 element of
                        # test triplet
                        rows = np.equal(np.count_nonzero(cond, axis=1), 1)
                        # Prepare 2D array with nrows = number of rows with nonzero entry,
                        # ncols = length of test cluster (for CCE4 is 3)
                        tiled_test = np.tile(test, (np.count_nonzero(rows), 1))

                        if tiled_test.shape[-1] > 2:
                            # From test indexes for each row[i] of nonzero rows choose those indexes, which are not
                            # present in the bond[i],given by reverse cond array
                            flatten = tiled_test[~cond[rows]]
                            # Obtaining correct indexes from tiled test gives flattened array
                            # which should be reshaped nack into (nrows, order - bond).
                            # For CCE4 we need to add 2 indexes
                            # to bond to create a quartet, therefore appendix should have shape (nrows, 2)
                            appendix = flatten.reshape(flatten.size // (order - 2), order - 2)
                        else:
                            # For CCE3 it's easier to do in this way
                            # (probably, idk, I really just don't want to break it)
                            appendix = tiled_test[~cond[rows]][:, np.newaxis]

                        triplets = np.concatenate((testbonds[rows], appendix), axis=1)

                        # If strong keyword was used, the program will find only the completely interconnected clusters
                        # For CCE4 this means that from the given triplet i,j,k to form an interconnected array
                        # i,j,k,l, vertex l should have edges il, jl, kl. Therefore the quartet will appear 3 times
                        # in the array triplets. we choose unique quartets, and from them choose only quartets that
                        # appeared 3 times.
                        if strong and triplets.any():
                            unique, counts = np.unique(np.sort(triplets, axis=1), axis=0, return_counts=True)
                            triplets = unique[counts == order - 1]

                            if triplets.any():
                                ltriplets.append(triplets)
                                # print(triplets)

                        else:
                            ltriplets.append(triplets)

                    # Transform list of numpy arrays into numpy array
                    try:
                        ltriplets = np.concatenate(ltriplets, axis=0)
                        ltriplets = np.unique(np.sort(ltriplets, axis=1), axis=0)
                    except ValueError:
                        break

                    subclusters[order] = ltriplets

                clusters[order].append(subclusters[order])

    for o in range(1, maximum_order + 1):
        if clusters[o]:
            # print(clusters[o])
            clusters[o] = np.concatenate(clusters[o], axis=0)
        else:
            print('Set of clusters of order {} is empty!'.format(o))
            clusters.pop(o)

    return clusters


def expand_clusters(sc):
    """
    Expand dict with subclusters so each cluster include all possible additions of one more bath spin
    @param sc: dict
    initial clusters dictionary
    @return: dict
    dict of expanded clusters
    """
    indexes = np.arange(sc[1].size, dtype=np.int32)
    comb = np.array([*combinations(indexes, 2)], dtype=np.int32)

    newsc = {}
    newsc[1] = indexes[:, np.newaxis]
    newsc[2] = comb

    for o in sorted(sc)[1:]:
        lexpanded = []

        for test in sc[o]:
            cond = np.any(comb.reshape(comb.shape + (1,)) == test, axis=1)
            rows = np.equal(np.count_nonzero(cond, axis=1), 1)

            tiled_test = np.tile(test, (np.count_nonzero(rows), 1))

            flatten = tiled_test[~cond[rows]]
            appendix = flatten.reshape(-1, o - 1)

            triplets = np.concatenate((comb[rows], appendix), axis=1)
            lexpanded.append(triplets)

        lexpanded = np.concatenate(lexpanded, axis=0)
        lexpanded = np.unique(np.sort(lexpanded, axis=1), axis=0)
        newsc[o + 1] = lexpanded

    return newsc
