from pycce.utilities import *


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

    def __init__(self, dimensions, vectors=None):
        self.dimensions = np.asarray(dimensions)
        self.spins = (dimensions - 1) / 2
        if vectors is None:
            vectors = []
            for j, s in enumerate(self.spins):
                vectors.append(spinvec(s, j, dimensions))
            vectors = np.asarray(vectors)
        self.vectors = vectors

        tdim = np.prod(dimensions)
        self.data = np.zeros((tdim, tdim), dtype=np.complex128)

    def __getitem__(self, item):
        return self.data.__getitem__(item)

    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)

    def __delitem__(self, key):
        self.data.__delitem__(key)

    def __getattr__(self, item):
        if item in dir(self):
            return getattr(self, item)
        else:
            return getattr(self.data, item)

    # def __add__(self, other):
    #     self.data.__add__(other)
    #
    # def __sub__(self, other):
    #     self.data.__sub__(other)
    #
    # def __mul__(self, other):
    #     self.data.__mul__(other)
    #
    # def __matmul__(self, other):
    #     self.data.__matmul__(other)
    #
    # def __truediv__(self, other):
    #     self.data.__truediv__(other)
    #
    # def __floordiv__(self, other):
    #     self.data.__floordiv__(other)
    #
    # def __mod__(self, other):
    #     self.data.__mod__(other)
    #
    # def __pow__(self, other):
    #     self.data.__pow__(other)

    # def __ilshift__(self, other):
    #     self.data.__ilshift__(self, other)
    #
    # def __irshift__(self, other):
    #     self.data.__irshift__(self, other)
    #
    # def __iand__(self, other):
    #     self.data.__iand__(self, other)
    #
    # def __ixor__(self, other):
    #     self.data.__ixor__(self, other)
    #
    # def __ior__(self, other):
    #     self.data.__ior__(self, other)
