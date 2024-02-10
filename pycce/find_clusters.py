from collections.abc import MutableMapping
from itertools import combinations

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from scipy.sparse import csr_matrix


class _Clusters(MutableMapping):
    """
    NOT IMPLEMENTED YET. Specific Class for storing the clusters objects
    """

    def __init__(self, ):
        self._data = {}

    def __getitem__(self, key):
        return self._data[key]

    def __delitem__(self, key):
        del self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"{type(self).__name__}(" + " ".join(str(x) for x in self.keys()) + ")"

    def keys(self):
        return self._data.keys()


def generate_clusters(bath, r_dipole, order, r_inner=0, ignore=None, strong=False, nclusters=None):
    """
    Generate clusters for the bath spins.

    Args:

        bath (BathArray): Array of bath spins.
        r_dipole (float): Maximum connectivity distance.
        order (int): Maximum size of the clusters to find.

        r_inner (float): Minimum connectivity distance.
        ignore (list or str, optional):
            If not None, includes the names of bath spins which are ignored in the cluster generation.

        strong (bool): Whether to find only completely interconnected clusters (default False).

        nclusters (dict): Dictionary which contain maximum number of clusters of the given size.
            Has the form ``n_clusters = {order: number}``, where ``order`` is the size of the cluster,
            ``number`` is the maximum number of clusters with this size.

            If provided, sorts the clusters by the strength of cluster interaction,
            equal to the lowest pairwise interaction in the cluster. Then the strongest ``number`` of clusters is
            taken.

    Returns:

        dict:
            Dictionary with keys corresponding to size of the cluster,
            and value corresponds to ndarray of shape (M, N).
            Here M is the number of clusters of given size, N is the size of the cluster.
            Each row contains indexes of the bath spins included in the given cluster.
    """
    graph = make_graph(bath, r_dipole, r_inner=r_inner, ignore=ignore, max_size=5000)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    if nclusters is None:
        clusters = find_subclusters(order, graph, labels, n_components, strong=strong)
    else:
        clusters = find_valid_subclusters(graph, order, strong=strong, nclusters=nclusters, bath=bath)

    if ignore is not None and order > 0:
        if isinstance(ignore, (str, np.str)):
            clusters[1] = clusters[1][bath[clusters[1]]['N'] != ignore].reshape(-1, 1)
        else:
            for n in ignore:
                clusters[1] = clusters[1][bath[clusters[1]]['N'] != n].reshape(-1, 1)

    return clusters


def make_graph(bath, r_dipole, r_inner=0, ignore=None, max_size=5000):
    """
    Make a connectivity matrix for bath spins.

    Args:
        bath (BathArray): Array of bath spins.
        r_dipole (float): Maximum connectivity distance.
        r_inner (float): Minimum connectivity distance.
        ignore (list or str, optional):
            If not None, includes the names of bath spins which are ignored in the cluster generation.
        max_size (int): Maximum size of the bath before less optimal (but less memory intensive) approach is used.

    Returns:
        crs_matrix: Connectivity matrix.
    """

    if bath.size < max_size:
        dist_matrix = np.linalg.norm(bath['xyz'][:, np.newaxis, :] - bath['xyz'][np.newaxis, :, :], axis=-1)
        atoms_within = np.logical_and(dist_matrix < r_dipole, dist_matrix > r_inner)

    else:
        atoms_within = np.zeros((bath.size, bath.size), dtype=bool)
        for i, a in enumerate(bath):
            dist = np.linalg.norm(bath['xyz'][i:] - a['xyz'], axis=-1)
            atoms_within[i, i:] = (dist < r_dipole) & (dist > r_inner)
    if ignore is not None:
        if isinstance(ignore, (str, np.str)):
            atoms_within = atoms_within & (bath['N'] != ignore)[np.newaxis, :]
        else:
            for n in ignore:
                atoms_within = atoms_within & (bath['N'] != n)[np.newaxis, :]

    if bath.shape[0] == 0:
        print('No spins, no neighbours.')

    # Generate sparse connectivity matrix
    graph = csr_matrix(atoms_within, dtype=bool)

    return graph


# Import from connected components scipy
def connected_components(csgraph, directed=False, connection='weak', return_labels=True):
    """
    Find connected components using ``scipy.sparse.csgraph``.
    See documentation of ``scipy.sparse.csgraph.connected_components``
    """
    return scipy.sparse.csgraph.connected_components(csgraph, directed=directed, connection=connection,
                                                     return_labels=return_labels)


def find_subclusters(maximum_order, graph, labels, n_components, strong=False):
    """
    Find subclusters from connectivity matrix.

    Args:
        maximum_order (int):
            Maximum size of the clusters to find.
        graph (csr_matrix): Connectivity matrix.
        labels (ndarray with shape (n,)): Array of labels of the connected components.
        n_components (int): The number of connected components n.
        strong (bool): Whether to find only completely interconnected clusters (default False).

    Returns:
        dict:
            Dictionary with keys corresponding to size of the cluster,
            and value corresponds to ndarray of shape (M, N).
            Here M is the number of clusters of given size, N is the size of the cluster.
            Each row contains indexes of the bath spins included in the given cluster.
    """

    clusters = {}
    for k in range(1, maximum_order + 1):
        clusters[k] = []
    # print('Number of disjointed clusters is {}'.format(n_components))
    for component in range(n_components):
        vert_pos = (labels == component)
        vertices = np.nonzero(vert_pos)[0]

        # print('{} cluster contains {} components'.format(component, ncomp))

        subclusters = {1: vertices[:, np.newaxis]}

        clusters[1].append(vertices[:, np.newaxis])

        if vertices.size >= 2 and maximum_order > 1:
            # Retrieve upper right triangle (remove i,j pairs with i>j),
            # choose only rows corresponding to vertices in the subcluster
            csrmat = scipy.sparse.triu(graph, k=0, format='csr')[vertices]
            # Change to coordinate format of matrix
            coomat = csrmat.tocoo()
            # rows, col give row and colum indexes, which correspond to
            # edges of the graph. as we already chose the rows,
            # to obtain initial correct row indexes we need to use vertices array
            row_ind, col_ind = vertices[coomat.row], coomat.col

            bonds = np.column_stack([row_ind, col_ind])
            subclusters[2] = bonds
            clusters[2].append(bonds)

            # Check if [1,2] row in a matrix(Nx2):  any(np.equal(a, [1, 2]).all(1))

            for order in range(3, maximum_order + 1):

                # General way to compute clusters for any order >= 3
                # but for simplicity consider CCE4

                # List of cluster of size 4
                ltriplets = []

                # For ith triplet look at all bonds, if ith triplet contains
                # one and only one element of jth bond, ith triplet and jth bond form cluster of 4
                # There is no need to include last triplet, as it would be included
                # into quartet already if it were to be a part of one
                for i in range(subclusters[order - 1].shape[0] - 1):

                    # The triplet under study
                    test = subclusters[order - 1][i]

                    # For cluster i,j,k (i>j>k, as all indexes are stored in increasing order)
                    # consider only bonds l, n with l >= i, n >= j without loss of generality
                    testbonds = bonds[np.all(bonds >= test[:2], axis=1)]

                    # cond is a bool 2D array of shape (testbonds.shape[0], test.size)
                    # i.e. number of rows corresponds to number of testbonds,
                    # length of the row is equal to the length of the test cluster (3 in case CCE4)
                    # cond[i,j] is True if bond[i] contains element of test[j], otherwise False

                    # To construct this array the following procedure is applied:
                    # Reshape testbonds from (n, 2) to (n, 2, 1)
                    # when asked to do logical operation "==" testbonds is broadcasted to shape (n, 2, order - 1)
                    # In the case of CCE4 (n, 2, 3). Resulting 3D bool array has True entry i,j,k
                    # If j element of testbonds[i] is equal to k element of test
                    # Applying logical operation "any" along 2nd axis (axis=1, any element of the bond i)
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
                        # For CCE3 it's faster to do in this way
                        # (but I really just don't want to break it)
                        appendix = tiled_test[~cond[rows]][:, np.newaxis]

                    triplets = np.concatenate((testbonds[rows], appendix), axis=1)

                    # If strong keyword was used, the program will find only the completely interconnected clusters
                    # For CCE4 this means that from the given triplet i,j,k to form an interconnected array
                    # i,j,k,l, vertex l should have edges il, jl, kl. Therefore, the quartet will appear 3 times
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
            clusters[o] = np.concatenate(clusters[o], axis=0)
        else:
            print('Set of clusters of order {} is empty!'.format(o))
            clusters.pop(o)

    return clusters


def combine_clusters(cs1, cs2):
    """
    Combine two dictionaries with clusters.

    Args:
        cs1 (dict): First cluster dictionary with keys corresponding to size of the cluster,
            and value corresponds to ndarray of shape (M, N).

        cs2 (dict): Second cluster dictionary with the same structure.

    Returns:
        dict: Combined dictionary with unique clusters from both dictionaries.
    """
    keys_1 = list(cs1.keys())
    keys_2 = list(cs2.keys())
    keys = {*keys_1, *keys_2}
    cs_combined = {}
    for k in keys:
        if k in keys_1 and k in keys_2:
            indexes = np.concatenate((cs1[k], cs2[k]))
            cs_combined[k] = np.unique(np.sort(indexes, axis=1), axis=0)
        elif k in keys_1:
            cs_combined[k] = cs1[k]
        elif k in keys_2:
            cs_combined[k] = cs2[k]
    return cs_combined


def expand_clusters(sc):
    """
    Expand dict so each new cluster will include all possible additions of one more bath spin. This increases
    maximum size of the cluster by one.

    Args:
        sc (dict): Initial clusters dictionary.

    Returns:
        dict: Dictionary with expanded clusters.
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


def _default_strength(bath, bonds):
    # function to be used to normalize how to compute strength in the future
    row_ind, col_ind = bonds.T

    # cos_theta = (bath[col_ind].z - bath[row_ind].z) / r

    r = bath[col_ind].dist(bath[row_ind])

    gyros_1 = bath[col_ind].gyro
    gyros_2 = bath[row_ind].gyro
    if len(gyros_1.shape) > 1:
        gyros_1 = np.abs(gyros_1.reshape(gyros_1.shape[0], -1)).max(axis=1)
    if len(gyros_2.shape) > 1:
        gyros_2 = np.abs(gyros_2.reshape(gyros_2.shape[0], -1)).max(axis=1)

    return np.abs(gyros_1 * gyros_2 / r ** 3)  # * (1 - 2 * cos_theta ** 2))


def find_valid_subclusters(graph, maximum_order, nclusters=None, bath=None, strong=False, compute_strength=None):
    """
    Find subclusters from connectivity matrix.

    Args:
        maximum_order (int):
            Maximum size of the clusters to find.
        graph (csr_matrix): Connectivity matrix.
        nclusters (dict): Dictionary which contain maximum number of clusters of the given size.
        bath (BathArray): Array of bath spins.
        strong (bool): Whether to find only completely interconnected clusters (default False).

    Returns:
        dict:
            Dictionary with keys corresponding to size of the cluster,
            and value corresponds to ndarray of shape (M, N).
            Here M is the number of clusters of given size, N is the size of the cluster.
            Each row contains indexes of the bath spins included in the given cluster.
    """
    # This function is called when nclusters is provided in the Simulator class

    clusters = {1: np.arange(graph.shape[0])[:, np.newaxis]}

    if nclusters is not None and isinstance(nclusters, int):
        nclusters = {k: nclusters for k in range(1, maximum_order + 1)}

    if maximum_order > 1:
        strength = {}
        # Retrieve upper right triangle (remove i,j pairs with i>j),
        csrmat = scipy.sparse.triu(graph, k=0, format='csr')
        # Change to coordinate format of matrix
        coomat = csrmat.tocoo()
        row_ind, col_ind = coomat.row, coomat.col

        bonds = np.column_stack([row_ind, col_ind])

        if nclusters is not None:
            strength[2] = 1 / _default_strength(bath, bonds)
            ordered = strength[2].argsort()  # smallest strength - largest coupling # [::-1] if strength = np.abs
            bonds = bonds[ordered]
            strength[2] = strength[2][ordered]

            if 2 in nclusters:
                bonds = bonds[:nclusters[2]]
                strength[2] = strength[2][:nclusters[2]]

        clusters[2] = bonds

        del coomat, row_ind, col_ind

        for order in range(3, maximum_order + 1):

            ltriplets = []
            # list of triplet strength (used only when nclusters is not None)
            ltstr = []

            for i in range(clusters[order - 1].shape[0] - 1):

                # The triplet under study
                test = clusters[order - 1][i]
                tripletstrength = None

                # For cluster i,j,k (i>j>k, as all indexes are stored in increasing order)
                # consider only bonds l, n with l >= i, n >= j without loss of generality
                choosebonds = np.all(bonds >= test[:2], axis=1)
                testbonds = bonds[choosebonds]

                cond = np.any(testbonds.reshape(testbonds.shape + (1,)) == test, axis=1)
                # Check which of testbonds form a cluster with the triplet i,j,k
                # rows is 1D bool array, rows[i] is True if bond[i] contains exactly 1 element of
                # test triplet
                rows = np.equal(np.count_nonzero(cond, axis=1), 1)
                # Prepare 2D array with nrows = number of rows with nonzero entry,
                # ncols = length of test cluster (for CCE4 is 3)
                tiled_test = np.tile(test, (np.count_nonzero(rows), 1))

                if tiled_test.shape[-1] > 2:
                    flatten = tiled_test[~cond[rows]]
                    appendix = flatten.reshape(flatten.size // (order - 2), order - 2)
                else:
                    appendix = tiled_test[~cond[rows]][:, np.newaxis]

                triplets = np.concatenate((testbonds[rows], appendix), axis=1)

                if nclusters is not None:
                    teststrength = strength[order - 1][i]
                    tripletstrength = strength[2][choosebonds][rows]
                    # tripletstrength[tripletstrength > teststrength] = teststrength to choose the smallest coupling
                    # as a strength of the triplet
                    tripletstrength += teststrength

                if strong and triplets.any():
                    unique, index, counts = np.unique(np.sort(triplets, axis=1), axis=0, return_index=True,
                                                      return_counts=True)

                    triplets = unique[counts == order - 1]

                    if triplets.any():
                        ltriplets.append(triplets)

                        if nclusters is not None:
                            tripletstrength = tripletstrength[index[counts == order - 1]]
                            ltstr.append(tripletstrength)

                else:
                    ltriplets.append(triplets)
                    if nclusters is not None:
                        ltstr.append(tripletstrength)

            # Transform list of numpy arrays into numpy array

            try:

                ltriplets = np.concatenate(ltriplets, axis=0)

                # First order by lowest strength, so from two identical triplets
                # one with lower strength will be first in np.unique call

                if nclusters is not None:
                    ltstr = np.concatenate(ltstr)
                    ordered_by_lowest_strength = ltstr.argsort()  # [::-1]  # reverse indexes b/c strength is 1/strength
                    ltriplets = ltriplets[ordered_by_lowest_strength]
                    ltstr = ltstr[ordered_by_lowest_strength]

                ltriplets, indexes = np.unique(np.sort(ltriplets, axis=1), axis=0, return_index=True)

                if nclusters is not None:
                    ltstr = ltstr[indexes]
                    ordered_by_strength = ltstr.argsort()  # 0 strength - all bonds are strongly coupled

                    ltriplets = ltriplets[ordered_by_strength]
                    ltstr = ltstr[ordered_by_strength]

                    if order in nclusters:
                        ltstr = ltstr[:nclusters[order]]
                        ltriplets = ltriplets[:nclusters[order]]

            except ValueError:
                print('Set of clusters of order {} is empty!'.format(order))
                break

            clusters[order] = ltriplets
            if nclusters is not None:
                strength[order] = ltstr

    return clusters
