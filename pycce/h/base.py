import numpy as np
from pycce.sm import dimensions_spinvectors, vecs_from_dims


class Hamiltonian:
    r"""
    Class containing properties of the Hamiltonian.

    Essentially wrapper for ndarray with additional attributes of ``dimensions`` and ``spins``.

    Usual methods (e.g. ``__setitem__`` or ``__getitem__``) access the ``data`` attribute.

    .. note::

        Algebraic operations with Hamiltonian will return ndarray instance.

    Args:
        dimensions (array-like): array of the dimensions for each spin in the Hilbert space of the Hamiltonian.

    Attributes:
        dimensions (ndarray): array of the dimensions for each spin in the Hilbert space of the Hamiltonian.
        spins (ndarray): array of the spins, spanning the Hilbert space of the Hamiltonian.
        vectors (list): list with spin vectors of form ``[[Ix, Iy, Iz], [Ix, Iy, Iz], ...]``.
        data (ndarray): matrix representation of the Hamiltonian.

    """

    def __init__(self, dimensions, vectors=None, data=None):
        self.dimensions = np.asarray(dimensions)
        self.spins = (dimensions - 1) / 2
        if vectors is None:
            vectors = vecs_from_dims(dimensions)
        self.vectors = vectors

        tdim = self.dimensions.prod()
        if data is None:
            self.data = np.zeros((tdim, tdim), dtype=np.complex128)
        else:
            assert data.shape == (tdim, tdim), "Wrong data shape"
            self.data = data.astype(np.complex128)

    def __getitem__(self, item):
        return self.data.__getitem__(item)

    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)

    def __delitem__(self, key):
        self.data.__delitem__(key)

    #
    # def __getattr__(self, item):
    #     if item in dir(self):
    #         return getattr(self, item)
    #     else:
    #         return getattr(self.data, item)

    @classmethod
    def from_bath(cls, bath, center=None):
        dim, vectors = dimensions_spinvectors(bath, central_spin=center)
        return cls(dim, vectors=vectors)

    def __add__(self, other):
        res = self.data.__add__(other)
        return Hamiltonian(self.dimensions, self.vectors, res)

    def __iadd__(self, other):
        self.data.__iadd__(other)
        return self

    def __sub__(self, other):
        res = self.data.__sub__(other)
        return Hamiltonian(self.dimensions, self.vectors, res)

    def __isub__(self, other):
        self.data.__isub__(other)
        return self

    def __mul__(self, other):
        res = self.data.__mul__(other)
        return Hamiltonian(self.dimensions, self.vectors, res)

    def __imul__(self, other):
        self.data.__imul__(other)
        return self

    def __matmul__(self, other):
        res = self.data.__matmul__(other)
        return Hamiltonian(self.dimensions, self.vectors, res)

    def __imatmul__(self, other):
        self.data.__imatmul__(other)
        return self

    def __truediv__(self, other):
        res = self.data.__truediv__(other)
        return Hamiltonian(self.dimensions, self.vectors, res)

    def __itruediv__(self, other):
        self.data.__itruediv__(other)
        return self

    def __floordiv__(self, other):
        res = self.data.__floordiv__(other)
        return Hamiltonian(self.dimensions, self.vectors, res)

    def __ifloordiv__(self, other):
        self.data.__ifloordiv__(other)
        return self

    def __mod__(self, other):
        res = self.data.__mod__(other)
        return Hamiltonian(self.dimensions, self.vectors, res)

    def __imod__(self, other):
        self.data.__imod__(other)
        return self

    def __pow__(self, other):
        res = self.data.__pow__(other)
        return Hamiltonian(self.dimensions, self.vectors, res)

    def __ipow__(self, other):
        self.data.__ipow__(other)
        return self
