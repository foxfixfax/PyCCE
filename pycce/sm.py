from collections.abc import MutableMapping

import numpy as np
from numba import jit
from pycce.utilities import expand


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
        self.p = self.plus

        for i in range(dim - 1):
            self.plus[i, i + 1] += np.sqrt(s * (s + 1) -
                                           projections[i] * projections[i + 1])

        self.minus = self.plus.conj().T
        self.m = self.minus

        self.s = s
        self.dim = dim

        self.x = 1 / 2. * (self.plus + self.minus)
        self.y = 1 / 2j * (self.plus - self.minus)
        self.z = np.diag(projections[::-1])

        self.eye = np.eye(dim, dtype=np.complex128)

        self._stevens = {}

    def stev(self, k, q):
        try:
            return self._stevens[(k, q)]
        except KeyError:
            self._stevens[(k, q)] = stevo(self, k, q)

        return self._stevens[(k, q)]

    def __repr__(self):
        return "Spin-{:.1f} matrices x, y, z".format(self.s)


class MatrixDict(MutableMapping):
    """
    Class for storing the SpinMatrix objects.

    """

    def __init__(self, *spins):
        self._data = {}
        if spins:
            for s in spins:
                self[s] = SpinMatrix[s]

    def __getitem__(self, key):
        try:
            cond = key in self._data.keys()
        except TypeError:
            key = key[()]
            cond = key in self._data.keys()
        if not cond:
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

# Stevens operators
from scipy.special import comb


def _a(k, q, m, j=1., i=1):
    if k == q and m == 0:
        return 1
    elif q + m > k or m < 0:
        return 0

    q = q + 1

    first = (2 * q + m - 1) * _a(k, q, m - 1, j=j, i=i) + (q * (q - 1) - m * (m + 1) / 2) * _a(k, q, m, j=j, i=i)
    second = 0

    for n in range(1, k - q - m + 1):
        stuff = comb(m + n, m) * (j * (j + 1)) ** i - comb(m + n, m - 1) - comb(m + n, m - 2)
        second += (-1) ** n * stuff * _a(k, q, m + n, j=j, i=i)

    return first + second


def _f(k, q):
    a_s = []
    for m in range(0, k - q + 1):
        for i in range(0, int((k - q - m) / 2 + 1)):
            a_s.append(_a(k, q, m, j=1, i=i))
    a_s = np.abs(np.array(a_s, dtype=int))
    a_s = a_s[a_s > 0]

    return np.gcd.reduce(a_s)


def stevo(sm, k, q):
    """
    Stevens operators (from I.D. Ryabov, Journal of Magnetic Resonance 140, 141–145 (1999)).

    Args:
        sm (SpinMatrix): Spin matrices of the given spin.
        k (int): :math:`k` index of the Stevens operator.
        q (int): :math:`q` index of the Stevens operator.

    Returns:
        ndarray with shape (n, n): Stevens operator representation in :math:`S_z` basis.
    """
    spin = sm.s
    mp = np.linalg.matrix_power
    if q == 0:
        alpha = 2
    elif (k % 2 == 1) or (q % 2 == 0):
        alpha = 1
    else:
        alpha = 1 / 2

    sign = np.sign(q)
    if q < 0:
        q = -q
        pref = alpha / (2j * _f(k, q))

    else:
        pref = alpha / (2 * _f(k, q))

    full = 0
    for m in range(0, k - q + 1):
        full += _a(k, q, m, j=spin) * ((mp(sm.plus, q) + sign * (-1) ** (k - q - m) * mp(sm.minus, q)) @ mp(sm.z, m))

    return pref * full


def dimensions_spinvectors(bath=None, central_spin=None):
    """
    Generate two arrays, containing dimensions of the spins in the cluster and the vectors with spin matrices.

    Args:
        bath (BathArray with shape (n,)): Array of the n spins within cluster.
        central_spin (CenterArray, optional): If provided, include dimensions of the central spins.

    Returns:
        tuple: *tuple* containing:

            * **ndarray with shape (n,)**: Array with dimensions for each spin.

            * **list**: List with vectors of spin matrices for each spin in the cluster
              (Including central spin if ``central_spin`` is not None). Each with  shape (3, N, N) where
              ``N = prod(dimensions)``.
    """
    dimensions = []

    if bath is not None:
        dimensions += [n.dim for n in bath]

    if central_spin is not None:
        try:
            for c in central_spin:
                dimensions += [c.dim]

        except TypeError:
            dimensions += [central_spin.dim]

    dimensions = np.array(dimensions, dtype=np.int32)
    vectors = vecs_from_dims(dimensions)

    return dimensions, vectors


@jit(cache=True, nopython=True)
def vecs_from_dims(dimensions):
    """
    Generate ndarray of spin vectors, given the array of spin dimensions.

    Args:
        dimensions (ndarray with shape (n,)): Dimensions of spins.

    Returns:
        ndarray with shape (n, 3, X, X): Array of spin vectors in full Hilbert space.
    """
    td = dimensions.prod()
    vectors = np.zeros((len(dimensions), 3, td, td), dtype=np.complex128)
    for j, d in enumerate(dimensions):
        vectors[j] = spinvec(j, dimensions)
    return vectors


@jit(cache=True, nopython=True)
def spinvec(j, dimensions):
    """
    Generate single spin vector, given the index and dimensions of all spins in the cluster.

    Args:
        j (int): Index of the spin.
        dimensions (ndarray with shape (n,)): Dimensions of spins.

    Returns:
        ndarray with shape (3, X, X): Spin vector of :math:`j`-sth spin in full Hilbert space.

    """
    x, y, z = numba_gen_sm(dimensions[j])
    vec = np.stack((expand(x, j, dimensions),
                    expand(y, j, dimensions),
                    expand(z, j, dimensions))
                   )
    return vec


@jit(cache=True, nopython=True)
def numba_gen_sm(dim):
    """
    Numba-friendly spin matrix.
    Args:
        dim (int): dimensions of the spin marix.

    Returns:
        ndarray:
    """
    s = (dim - 1) / 2
    projections = np.linspace(-s, s, dim).astype(np.complex128)
    plus = np.zeros((dim, dim), dtype=np.complex128)

    for i in range(dim - 1):
        plus[i, i + 1] += np.sqrt(s * (s + 1) -
                                  projections[i] * projections[i + 1])

    minus = plus.conj().T
    x = 1 / 2. * (plus + minus)
    y = 1 / 2j * (plus - minus)
    z = np.diag(projections[::-1])
    return x, y, z
