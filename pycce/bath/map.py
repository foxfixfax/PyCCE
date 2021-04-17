from collections import defaultdict
from collections.abc import MutableMapping

import numpy as np
from scipy.sparse.sputils import isintlike
from numba import jit

TwoLayerDict = lambda: defaultdict(dict)


class InteractionMap(MutableMapping):
    def __init__(self, rows=None, columns=None, tensors=None):
        self.mapping = dict()
        self._indexes = None

        if (rows is not None) & (columns is not None) & (tensors is not None):
            assert len(columns) == len(tensors) and len(rows) == len(columns), "Data dimensions do not match"
            self[rows, columns] = tensors

    @property
    def indexes(self):
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

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.indexes)

    def __repr__(self):
        return f"{type(self).__name__} with {len(self.indexes)} pairwise tensors"

    def update(self, **kwargs):
        # Prevent direct usage of update
        raise NotImplementedError("Direct modification to InteractionMap element "
                                  "is not allowed.")

    def keys(self):
        return self.mapping.keys()

    def items(self):
        return self.mapping.items()

    def __gen_indexes(self):
        self.indexes = np.fromiter((ind for pair in self.mapping.keys() for ind in pair),
                                   dtype=np.int32).reshape(-1, 2)

    def subspace(self, array):
        """
        Get new InteractionMap with indexes readressed from array. Within the subspace indexes are renumbered
        E.g. array = [3,4,7]. Subspace will contain InteractionMap only within [3,4,7] elements
        with new indexes [0, 1, 2]
        :param array: ndarray
            either bool array containing True for elements within the subspace
            or array of indexes presented in the subspace
        :return: InteractionMap
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

        xind = np.argsort(indexes)
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
        obj = cls()
        if presorted:
            obj.mapping = dictionary
        else:
            for k in dictionary:
                obj[k] = dictionary[k]
        return obj

    # TODO implement compressed tensors
    def compress(self):
        self.tensors = np.empty((self.indexes.shape[0] * 2, 3, 3))
        self.tensors = None
        pass

@jit(nopython=True)
def find_first(item, vec):
    """
    return the index of the first occurence of item in vec. Otherwise returns None
    :param item: obj
        object to find in the vec
    :param vec: arraylike
        an array of objects
    :return: int or None
        index of item if it is in vec, otherwise None
    """
    """
    """
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return None


def imap_vectorized_item(list_of_keys, list_of_values, get):
    keys = np.asarray(list_of_keys)
    values = np.asarray(list_of_values)

    getreshuffle = get[:, 0] > get[:, 1]
    get.sort(axis=1)
    na = np.newaxis
    where = ((keys[:, na, 0] == get[na, :, 0]) & (keys[:, na, 1] == get[na, :, 1]))
    indkeys, indget = np.nonzero(where)
    vs = values[indkeys]
    vs[getreshuffle] = vs[getreshuffle].transpose(0, 2, 1)
    return vs


def _index(key):
    try:
        a, b = key
    except TypeError as e:
        raise TypeError('invalid index format') from e
    return a, b
