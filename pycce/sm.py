from collections.abc import MutableMapping

import numpy as np


# HBAR = 1.05457172  # When everything else in rad, kHz, ms, G, A


class SpinMatrix:
    """
    Class containing the spin matrices in Sz basis.

    Args:
        s (float): Total spin.
    """

    def __init__(self, s):
        dim = int(2 * s + 1 + 1e-8)

        projections = np.linspace(-s, s, dim, dtype=np.complex128)

        self.plus = np.zeros((dim, dim), dtype=np.complex128)

        for i in range(dim - 1):
            self.plus[i, i + 1] += np.sqrt(s * (s + 1) -
                                           projections[i] * projections[i + 1])

        self.minus = self.plus.conj().T

        self.s = s
        self.dim = dim

        self.x = 1 / 2. * (self.plus + self.minus)
        self.y = 1 / 2j * (self.plus - self.minus)
        self.z = np.diag(projections[::-1])

        self.eye = np.eye(dim, dtype=np.complex128)

    def __repr__(self):
        return "Spin-{:.1f} matrices x, y, z".format(self.s)


class MatrixDict(MutableMapping):
    """
    Class for storing the SpinMatrtix objects.

    """

    def __init__(self, *spins):
        self._data = {}
        if spins:
            for s in spins:
                self[s] = SpinMatrix[s]

    def __getitem__(self, key):
        if key not in self._data.keys():
            self._data[key] = SpinMatrix(key)
        return self._data[key]

    def __delitem__(self, key):
        del self._data[key]

    def __setitem__(self, key, value):
        if not isinstance(value, SpinMatrix):
            raise ValueError('Illegal definition of MatrixDict element. Only accepts SpinMatrix objects')
        self._data[key] = value

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"{type(self).__name__}(" + " ".join(str(x) for x in self.keys()) + ")"

    def keys(self):
        return self._data.keys()


_smc = MatrixDict()
