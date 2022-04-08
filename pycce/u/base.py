from collections import defaultdict

import numpy as np
import scipy.linalg
from pycce.utilities import expand, shorten_dimensions


class Propagator:
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

    def __init__(self, base_hamiltonian):
        self.base_hamiltonian = base_hamiltonian

    def generate_base_hamiltonian(self):
        return 1

    def generate_rotations(self, pulses, dimensions=None,
                           bath=None, vectors=None, central_spin=None):
        """
        Generate list of matrix representations of the rotations, induced by the sequence of the pulses.

        The rotations are stored in the ``.rotation`` attribute of the each ``Pulse`` object
        and in ``Sequence.rotations``.

        Args:
            dimensions (ndarray with shape (N,)): Array of spin dimensions in the system.
            bath (BathArray with shape (n,)): Array of bath spins in the system.
            vectors (ndarray  with shape (N, 3, prod(dimensions), prod(dimensions))):
                 Array with vectors of spin matrices for each spin in the system.

            central_spin (CenterArray): Optional. Array of central spins.

        Returns:
            tuple: *tuple* containing:

            * **list** or **None**: List with delays before each pulse or None if equispaced.

            * **list**: List with matrix representations of the rotation from each pulse.
        """
        cs = central_spin
        ndims = len(dimensions)
        if cs is not None:
            nc = len(cs)  # number of central spins
            shortdims = shorten_dimensions(dimensions, nc)

        self.delays = None
        self.rotations = None
        equispaced = not any(p._has_delay for p in pulses.data)

        if equispaced:
            delays = None

        else:
            delays = [p.delay if p.delay is not None else 0 for p in pulses.data]

        rots = []
        # Sigma as in central spin array is total spin
        total_sigma = {}
        separate_sigma = defaultdict(dict)
        for p in pulses.data:

            initial_rotation = rotation = 1

            if p.angle and cs is not None:
                if p.which is None:
                    if p.axis not in total_sigma:
                        total_sigma[p.axis] = expand(cs.sigma[p.axis], ndims - nc, shortdims)
                    rotation = center_rotation(total_sigma, p.axis, p.angle)
                else:
                    for i in p.which:
                        c = cs[i]
                        if p.axis not in separate_sigma[i]:
                            separate_sigma[i][p.axis] = expand(c.sigma[p.axis], ndims - nc, shortdims)
                        rotation = np.dot(center_rotation(separate_sigma[i], p.axis, p.angle), rotation)

            if p.bath_names is not None:
                if vectors.shape != bath.shape:
                    vectors = vectors[:bath.shape[0]]
                    # print(vectors.shape)
                properties = np.broadcast(p.bath_names, p.bath_axes, p.bath_angles)

                for name, axis, angle in properties:
                    if angle:
                        which = (bath.N == name)

                        if any(which):
                            vecs = vectors[which]
                            rotation = np.dot(bath_rotation(vecs, axis, angle), rotation)

            if initial_rotation is rotation:
                rotation = None

            p.rotation = rotation

            rots.append(rotation)

        self.delays = delays
        self.rotations = rots
        return delays, rots


_rot = {'x': 0, 'y': 1, 'z': 2}


def bath_rotation(vectors, axis, angle):
    """
    Generate rotation of the bath spins with given spin vectors.

    Args:
        vectors (ndarray with shape (n, 3, x, x)): Array of *n* bath spin vectors.
        axis (str): Axis of rotation.
        angle (float): Angle of rotation.

    Returns:
        ndarray with shape (x, x): Matrix representation of the spin rotation.

    """
    ax = _rot[axis]  # name -> index
    if (angle == np.pi) and (vectors[0][0, 0, 0] < 1):  # only works for spin-1/2
        rotation = -1j * 2 * vectors[0][ax]  # 2 here is to transform into pauli matrices
        for v in vectors[1:]:
            np.matmul(rotation, -1j * 2 * v[ax], out=rotation)
    else:
        rotation = scipy.linalg.expm(-1j * vectors[0][ax] * angle)
        for v in vectors[1:]:
            add = scipy.linalg.expm(-1j * v[ax] * angle)
            np.matmul(rotation, add, out=rotation)

    return rotation


def center_rotation(sigma, axis, angle):
    """
    Generate rotation of the central spins with given spin vectors.

    Args:
        sigma (ndarray with shape (n, 3, x, x)): Pauli matrices.
        axis (str): Axis of rotation.
        angle (float): Angle of rotation.

    Returns:
        ndarray with shape (x, x): Matrix representation of the spin rotation.

    """
    if angle == np.pi:
        rotation = -1j * sigma[axis]
    else:
        rotation = scipy.linalg.expm(-1j * sigma[axis] * angle / 2)
    return rotation
