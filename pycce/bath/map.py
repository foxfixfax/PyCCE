import copy

import numpy as np
from collections import defaultdict
from collections.abc import MutableMapping
from scipy.sparse.sputils import isintlike

TwoLayerDict = lambda: defaultdict(dict)


class InteractionMap(MutableMapping):
    """
    Dict-like object containing information about tensor interactions between two spins.

    Each key is a tuple of two bath spin indexes.

    Args:
        rows (array-like with shape (n,)):
            Indexes of the bath spins, appearing on the left in the pairwise interaction.
        columns (array-like with shape (n,)):
            Indexes of the bath spins, appearing on the right in the pairwise interaction.
        tensors (array-like with shape (n, 3, 3)):
            Tensors of pairwise interactions between two spins with the indexes in ``rows`` and ``columns``.
    Attributes:
        mapping (dict): Actual dictionary storing the data.
    """

    def __init__(self, rows=None, columns=None, tensors=None):
        self.mapping = dict()
        self._indexes = None

        if (rows is not None) & (columns is not None) & (tensors is not None):
            self[rows, columns] = tensors
            self.__gen_indexes()

    @property
    def indexes(self):
        """
        ndarray with shape (n, 2): Array with the indexes of pairs of bath spins, for which the tensors are stored.
        """
        if self._indexes is None:
            self.__gen_indexes()
        return self._indexes

    @indexes.setter
    def indexes(self, newindexes):
        self._indexes = newindexes

    def __getitem__(self, key):
        a, b = _index(key)
        try:
            if a < b:
                return self.mapping[a, b]
            else:
                return self.mapping[b, a].T
        except (TypeError, ValueError):
            try:
                vs = []
                for j, k in zip(a, b):
                    v = self[j, k]
                    vs.append(v)
                return np.asarray(vs)

            except TypeError as e:
                raise TypeError('invalid index format') from e

    def __delitem__(self, key):
        a, b = _index(key)

        if isintlike(a) and isintlike(b):
            if a < b:
                del self.mapping[a, b]
            else:
                del self.mapping[b, a]
        else:
            for j, k in zip(a, b):
                if j < k:
                    del self.mapping[j, k]
                else:
                    del self.mapping[k, j]

        self.indexes = None

    def __setitem__(self, key, value):
        value = np.asarray(value, dtype=np.float64)

        if value.size == 9:
            value = value.reshape(3, 3)
        else:
            value = value.reshape(-1, 3, 3)
        a, b = _index(key)

        if isintlike(a) and isintlike(b):
            assert value.size == 9, 'Tensor should have shape of (3, 3)'
            if a < b:
                self.mapping[a, b] = value
            else:
                self.mapping[b, a] = value.T

        else:
            try:
                if value.size == 9:
                    for j, k in zip(a, b):
                        if j < k:
                            self.mapping[j, k] = value
                        else:
                            self.mapping[k, j] = value.T
                else:
                    for j, k, v in zip(a, b, value):
                        if j < k:
                            self.mapping[j, k] = v
                        else:
                            self.mapping[k, j] = v.T

            except TypeError as e:
                raise TypeError('invalid index format') from e

        self.indexes = None

    def shift(self, start, inplace=True):
        """
        Add an offset ``start`` to the indexes. If ``inplace`` is False, returns the copy of InteractionMap.

        Args:
            start (int): Offset in indexes.
            inplace (bool): If True, makes changes inplace. Otherwise returns copy of the map.

        Returns:
            InteractionMap: Map with shifted indexes.
        """

        if inplace:
            imap = self
        else:
            imap = copy.deepcopy(self)

        for i, j in imap.indexes:
            imap.mapping[i + start, j + start] = imap.mapping.pop((i, j))

        imap.indexes = None

        return imap

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.indexes)

    def __repr__(self):
        return f"{type(self).__name__}({self.mapping})"

    # def update(self, **kwargs):
    #     # Prevent direct usage of update
    #     raise NotImplementedError("Direct modification to InteractionMap element "
    #                               "is not allowed.")

    def keys(self):
        return self.mapping.keys()

    def items(self):
        return self.mapping.items()

    def __gen_indexes(self):
        self.indexes = np.fromiter((ind for pair in self.mapping.keys() for ind in pair),
                                   dtype=np.int32).reshape(-1, 2)

    def subspace(self, array):
        r"""
        Get new InteractionMap with indexes readressed from array. Within the subspace indexes are renumbered.

        Examples:

            The subspace of [3,4,7] indexes will contain InteractionMap only within [3,4,7] elements
            with new indexes [0, 1, 2].

                >>> import numpy as np
                >>> im = InteractionMap()
                >>> im[0, 3] = np.eye(3)
                >>> im[3, 7] = np.ones(3)
                >>> for k in im: print(k, '\n', im[k],)
                (0, 3)
                [[1. 0. 0.]
                 [0. 1. 0.]
                 [0. 0. 1.]]
                (3, 7)
                 [[1. 1. 1.]
                  [1. 1. 1.]
                  [1. 1. 1.]]
                >>> array = [3, 4, 7]
                >>> sim = im.subspace(array)
                >>> for k in sim: print(k, '\n', sim[k])
                (0, 2)
                [[1. 1. 1.]
                 [1. 1. 1.]
                 [1. 1. 1.]]


        Args:
            array (ndarray): Either bool array containing True for elements within the subspace
                or array of indexes presented in the subspace.

        Returns:
            InteractionMap: The map for the subspace.
        """

        array = np.asarray(array)

        if array.dtype == bool:
            ind = np.arange(array.size, dtype=np.int32)
            indexes = ind[array]
        else:
            indexes = array

        newindexes = np.arange(indexes.size, dtype=np.int32)

        where = np.isin(self.indexes[:, 0], indexes) & np.isin(self.indexes[:, 1], indexes)
        pairs = self.indexes[where]

        xind = indexes.argsort()
        ypos = np.searchsorted(indexes[xind], pairs.flatten())

        indices = xind[ypos]
        newpairs = newindexes[indices].reshape(-1, 2)

        newdict = {}

        for (oi, oj), (i, j) in zip(pairs, newpairs):
            if i < j:
                newdict[i, j] = self[oi, oj]
            else:
                newdict[j, i] = self[oj, oi]

        return InteractionMap.from_dict(newdict, presorted=True)

    @classmethod
    def from_dict(cls, dictionary, presorted=False):
        """
        Generate InteractionMap from the dictionary.
        Args:
            dictionary (dict): Dictionary with tensors.
            presorted (bool): If true, assumes that the keys in the dictionary were already presorted.

        Returns:
            InteractionMap: New instance generated from the dictionary.
        """
        obj = cls()
        if presorted:
            obj.mapping = dictionary
        else:
            for k in dictionary:
                obj[k] = dictionary[k]
        return obj

    def __add__(self, other):
        new_obj = InteractionMap()
        keys_1 = list(self.keys())
        keys_2 = list(other.keys())

        for k in {*keys_1, *keys_2}:

            if (k in keys_1) and (k in keys_2):
                assert (other[k] == self[k]).all(), f'Error, tensor {k} has different properties in provided mappings'
                new_obj[k] = self[k]

            elif k in keys_1:
                new_obj[k] = self[k]
            else:
                new_obj[k] = other[k]
        return new_obj

    # TODO implement compressed tensors
    def _compress(self):
        self.tensors = np.empty((self.indexes.shape[0], 3, 3), dtype=np.float64)

        for j, ind in enumerate(self.indexes):
            self.tensors[j] = self[ind]


# This implementation is slower than the one above :(
class _CompressedIMap:
    def __init__(self, indexes=None, tensors=None, imap=None):
        self.tensors = tensors
        self.indexes = indexes
        if self.indexes is not None:
            unsorted = self.indexes[:, 1] < self.indexes[:, 0]
            if unsorted.any():
                self.tensors[unsorted] = np.transpose(self.tensors[unsorted], axes=(0, 2, 1))
                self.indexes[unsorted] = self.indexes[unsorted][..., ::-1]
        if imap is not None:
            self.tensors = np.empty((imap.indexes.shape[0], 3, 3), dtype=np.float64)
            self.indexes = imap.indexes.copy()

            for j, ind in enumerate(imap.indexes):
                self.tensors[j] = imap[ind]

    def subspace(self, array):

        array = np.asarray(array)

        if array.dtype == bool:
            ind = np.arange(array.size, dtype=np.int32)
            indexes = ind[array]
        else:
            # Indexes of the subspace. E.g. 1, 3, 5, 7
            indexes = array

        # Indexes in the subspace 0, 1, 2, 3
        newindexes = np.arange(indexes.size, dtype=np.int32)
        # Which indexes are present
        full = (self.indexes[:, :, np.newaxis] == indexes[np.newaxis, np.newaxis, :])
        where = full.any(-1).all(-1)
        where_exactly = np.nonzero(full[where])
        # lastpairs[where_exactly[:-1]] = newindexes[where_exactly[-1]]
        newpairs = newindexes[where_exactly[-1]].reshape(-1, 2)
        newtensors = self.tensors[where]
        return _CompressedIMap(newpairs, newtensors)

    def __getitem__(self, key):
        a, b = _index(key)

        which = (self.indexes[0] == a) & (self.indexes[b] == a)

        if not key.shape[-1] == 2:
            raise KeyError

        unsorted = (key[..., :-1] > key[..., 1:])
        any_unsorted = unsorted.any()

        if any_unsorted:
            key[unsorted] = key[unsorted][..., ::-1]
        which = (self.indexes[0] == key).all(1)

        if not np.count_nonzero(which) == key.size // 2:
            raise KeyError

        tensors = self.tensors[which]

        if any_unsorted:
            tensors[unsorted] = np.transpose(tensors[unsorted], axes=(0, 2, 1))
        if key.size == 2:
            return tensors[0]

        return tensors

    def __setitem__(self, key, value):
        value = np.asarray(value)
        which = (self.indexes == key).all(1)
        if not which.any():
            raise KeyError
        key.argsort(1)
        value.sort()


def _index(key):
    try:
        a, b = key
    except TypeError as e:
        raise TypeError('invalid index format') from e
    return a, b
