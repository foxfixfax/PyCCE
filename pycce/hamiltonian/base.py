import numpy as np
from pycce.utilities import *


class Hamiltonian:
    r"""
    Class containing properties of the Hamiltonian.

    Essentially wrapper for ndarray with additional attributes of ``dimensions`` and ``spins``.

    Usual methods (e.g. ``__setitem__`` or ``__getitem__``) access the ``data`` attribute.

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
            self.vectors = []
            for j, s in enumerate(self.spins):
                self.vectors.append(spinvec(s, j, dimensions))
        else:
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

    # def __iadd__(self, other):
    #     self.data.__iadd__(self, other)
    #
    # def __isub__(self, other):
    #     self.data.__isub__(self, other)
    #
    # def __imul__(self, other):
    #     self.data.__imul__(self, other)
    #
    # def __imatmul__(self, other):
    #     self.data.__imatmul__(self, other)
    #
    # def __itruediv__(self, other):
    #     self.data.__itruediv__(self, other)
    #
    # def __ifloordiv__(self, other):
    #     self.data.__ifloordiv__(self, other)
    #
    # def __imod__(self, other):
    #     self.data.__imod__(self, other)
    #
    # def __ipow__(self, other):
    #     self.data.__ipow__(self, other)
    #
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
