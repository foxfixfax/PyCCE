from pycce.sm import _smc
from pycce.sm import dimensions_spinvectors
from pycce.utilities import expand
from pycce.bath.array import _process_key_operator
from .base import Hamiltonian
from .functions import *


def bath_hamiltonian(bath, mfield):
    r"""
    Compute hamiltonian containing only the bath spins.

    Args:
        bath (BathArray):
            array of all bath spins in the cluster.

        mfield (ndarray with shape (3,) or func):
            Magnetic field of type ``mfield = np.array([Bx, By, Bz])``
            or callable with signature ``mfield(pos)``, where ``pos`` is ndarray with shape (3,) with the
            position of the spin.

    Returns:
        Hamiltonian: Hamiltonian of the given cluster without qubit.
    """
    dims, vectors = dimensions_spinvectors(bath, central_spin=None)
    clusterint = bath_interactions(bath, vectors) + custom_hamiltonian(bath, dims)

    for ivec, n in zip(vectors, bath):
        if callable(mfield):
            clusterint += expanded_single(ivec, n.gyro, mfield(n.xyz), n.Q, n.detuning)
        else:
            clusterint += expanded_single(ivec, n.gyro, mfield, n.Q, n.detuning)

    return Hamiltonian(dims, vectors, data=clusterint)


def total_hamiltonian(bath, center, mfield):
    """
    Compute total Hamiltonian of the given cluster.

    Args:
        bath (BathArray): Array of bath spins.

        center(CenterArray): Array of central spins.

        mfield (ndarray with shape (3,) or func):
            Magnetic field of type ``mfield = np.array([Bx, By, Bz])``
            or callable with signature ``mfield(pos)``, where ``pos`` is ndarray with shape (3, ) with the
            position of the spin.

    Returns:
        Hamiltonian: hamiltonian of the given cluster, including central spin.

    """

    dims, vectors = dimensions_spinvectors(bath, central_spin=center)

    totalh = bath_interactions(bath, vectors)

    ncenters = len(center)

    for i, c in enumerate(center):
        if callable(mfield):
            totalh += self_central(vectors[bath.size + i], mfield(c.xyz), c.zfs, c.gyro, c.detuning)
        else:
            # print('svec', vectors[bath.size + i].flags['C_CONTIGUOUS'])
            # print('tensor', c.zfs.flags['C_CONTIGUOUS'])
            totalh += self_central(vectors[bath.size + i], mfield, c.zfs, c.gyro, c.detuning)

    totalh += center_interactions(center, vectors[bath.size:])
    totalh += custom_hamiltonian(center, dims, offset=bath.size)
    totalh += custom_hamiltonian(bath, dims)

    for ivec, n in zip(vectors, bath):
        if callable(mfield):
            hsingle = expanded_single(ivec, n.gyro, mfield(n.xyz), n.Q, n.detuning)
        else:
            hsingle = expanded_single(ivec, n.gyro, mfield, n.Q, n.detuning)

        hhyperfine = 0

        for i in range(ncenters):
            if ncenters == 1:
                hf = n.A
            else:
                hf = n.A[i]

            hhyperfine += hyperfine(hf, vectors[bath.size + i], ivec)

        totalh += hsingle + hhyperfine

    return Hamiltonian(dims, vectors, data=totalh)


def central_hamiltonian(center, magnetic_field, hyperfine=None, bath_state=None):
    r"""
    Compute Hamiltonian, containing only central spin.

    Args:
        center (CenterArray or Center): Center spin.

        magnetic_field (ndarray with shape (3,) or func):
            Magnetic field of type ``magnetic_field = np.array([Bx, By, Bz])``
            or callable with signature ``magnetic_field(pos)``, where ``pos`` is ndarray with shape (3,) with the
            position of the spin.

        hyperfine (ndarray with shape (..., n, 3, 3)): Array of hyperfine tensors of bath spins.

        bath_state (ndarray with shape (n, )): Array of :math:`S_z` projections of bath spins.

    Returns:
        Hamiltonian: Central spin Hamiltonian.

    """
    dims, vectors = dimensions_spinvectors(central_spin=center)
    try:
        ncenters = len(center)
        single_center = False
    except TypeError:
        single_center = True
        ncenters = None

    if single_center:
        if callable(magnetic_field):
            totalh = self_central(vectors[0], magnetic_field(center.xyz),
                                  center.zfs, center.gyro, center.detuning)
        else:
            # print('svec', vectors[0].flags['C_CONTIGUOUS'])
            # print('tensor', center.zfs.flags['C_CONTIGUOUS'])
            totalh = self_central(vectors[0], magnetic_field,
                                  center.zfs, center.gyro, center.detuning)
        if hyperfine is not None and bath_state is not None:
            totalh += overhauser_central(vectors[0], hyperfine, bath_state)
        return totalh

    totalh = 0

    for i, c in enumerate(center):
        if callable(magnetic_field):
            totalh += self_central(vectors[i], magnetic_field(c.xyz), c.zfs, c.gyro)
        else:
            # print('svec', vectors[i].flags['C_CONTIGUOUS'])
            # print('tensor', c.zfs.flags['C_CONTIGUOUS'])
            totalh += self_central(vectors[i], magnetic_field, c.zfs, c.gyro)

        if hyperfine is not None and bath_state is not None:
            if ncenters == 1:
                hf = hyperfine
            else:
                hf = hyperfine[..., i, :, :]

            totalh += overhauser_central(vectors[i], hf, bath_state)

    totalh += center_interactions(center, vectors)
    totalh += custom_hamiltonian(center, dims)

    return Hamiltonian(dims, vectors, data=totalh)


def custom_hamiltonian(spins, dims=None, offset=0):
    """
    Custom addition to the Hamiltonian from the spins in the given array.

    Args:
        spins (BathArray or CenterArray): Array of the spins.
        dims (ndarray with shape (n, )): Dimensions of all spins in the cluster.
        offset (int): Index of the dimensions of the first spin from array in ``dims``. Default 0.

    Returns:
        ndarray with shape (N, N): Addition to the Hamiltonian.
    """
    if dims is None:
        dims = spins.dim
    add = 0
    for index, b in enumerate(spins):
        if b.h:
            add += custom_single(b.h, index + offset, dims)

    return add


def custom_single(h, index, dims):
    """
    Custom addition to the Hamiltonian from the dictionary with the parameters.

    Args:
        h (dict): Dictionary with coupling parameters.
        index (int): Index of the spin in ``dims``.
        dims (ndarray with shape (n, )): Dimensions of all spins in the cluster.

    Returns:
        ndarray with shape (N, N): Addition to the Hamiltonian.

    """
    sm = _smc[(dims[index] - 1) / 2]
    ham = 0
    for key in h:
        add = _process_key_operator(key, h[key], sm)
        ham += add
    return expand(ham, index, dims)


