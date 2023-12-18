import numpy as np
from numba import jit, types #  generated_jit,

from pycce.constants import HBAR_MU0_O4PI, PI2, ELECTRON_GYRO
from pycce.utilities import tensor_vdot, vec_tensor_vec
from numba.extending import overload

@jit(cache=True, nopython=True)
def expanded_single(ivec, gyro, mfield, self_tensor, detuning=.0):
    """
    Function to compute the single bath spin term.

    Args:
        ivec (ndarray with shape (3, n, n)): Spin vector of the bath spin in the full Hilbert space of the cluster.
        gyro (float or ndarray with shape (3, 3)):
        mfield (ndarray wtih shape (3, ): Magnetic field of type ``mfield = np.array([Bx, By, Bz])``.
        self_tensor (ndarray with shape (3, 3)): tensor of self-interaction of type IPI where I is bath spin.
        detuning (float): Additional term of d*Iz allowing to simulate different energy splittings of bath spins.

    Returns:
        ndarray with shape (n, n): Single bath spin term.

    """
    hzeeman = zeeman(ivec, gyro, mfield)
    hself = np.zeros(hzeeman.shape, dtype=hzeeman.dtype)

    if ivec[2, 0, 0].real > 0.5:
        hself = vec_tensor_vec(ivec, self_tensor, ivec)

        # v_ivec = np.einsum('ij,jkl->ikl', self_tensor, ivec, dtype=np.complex128)
        # hself = np.einsum('lij,ljk->ik', ivec, v_ivec, dtype=np.complex128)

    if detuning:
        hself += detuning * ivec[2]

    return hself + hzeeman


def _innerzeeman(gyro):

    if isinstance(gyro, float):
        def z(ivec, gyro, mfield):
            return -gyro / PI2 * (mfield[0] * ivec[0] + mfield[1] * ivec[1] + mfield[2] * ivec[2])

    # else assume tensor
    else:
        def z(ivec, gyro, mfield):
            gsvec = tensor_vdot(gyro / PI2, ivec)
            return mfield[0] * gsvec[0] + mfield[1] * gsvec[1] + mfield[2] * gsvec[2]

    return z

overload(_innerzeeman)(lambda gyro: _innerzeeman)

@jit(cache=True, nopython=True)
def zeeman(ivec, gyro, mfield):
    """
    Function
    Args:
        ivec (ndarray with shape (3, n, n)): Spin vector of the  spin in the full Hilbert space of the cluster.
        gyro (float or ndarray with shape (3, 3)): Gyromagnetic ratio of the spin.
        mfield (ndarray with shape (3, )): Magnetic field at the position of the spin.

    Returns:
        ndarray with shape (n, n): Zeeman interactions.

    """
    return _innerzeeman(gyro)(ivec, gyro, mfield)

def _inner_dd_tensor(g1, g2):
    """
    Generate dipole-dipole interaction tensor.

    Args:
        coord_1 (ndarray with shape (3,)): Coordinates of the first spin.
        coord_2 (ndarray with shape (3,)): Coordinates of the second spin.
        g1 (float or ndarray with shape (3, 3)): Gyromagnetic ratio of the first spin.
        g2 (float or ndarray with shape (3, 3)): Gyromagnetic ratio of the second spin.

    Returns:
        ndarray with shape (3, 3): Interaction tensor.

    """
    if isinstance(g2, float) and isinstance(g1, float):
        def func(coord_1, coord_2, g1, g2):
            p_tensor = gen_pos_tensor(coord_1, coord_2) * HBAR_MU0_O4PI / PI2
            p_tensor *= g2 * g1
            return p_tensor

    elif isinstance(g1, float):
        def func(coord_1, coord_2, g1, g2):
            p_tensor = gen_pos_tensor(coord_1, coord_2) * HBAR_MU0_O4PI / PI2
            p_tensor = g1 * p_tensor @ g2
            return p_tensor
    elif isinstance(g2, float):
        def func(coord_1, coord_2, g1, g2):
            p_tensor = gen_pos_tensor(coord_1, coord_2) * HBAR_MU0_O4PI / PI2
            p_tensor = g1 @ p_tensor * g2
            return p_tensor
    else:
        def func(coord_1, coord_2, g1, g2):
            p_tensor = gen_pos_tensor(coord_1, coord_2) * HBAR_MU0_O4PI / PI2
            p_tensor = g1 @ p_tensor @ g2
            return p_tensor
    return func

overload(_inner_dd_tensor)(lambda g1, g2: _inner_dd_tensor)

@jit(cache=True, nopython=True)
def dd_tensor(coord_1, coord_2, g1, g2):
    """
    Generate dipole-dipole interaction tensor.

    Args:
        coord_1 (ndarray with shape (3,)): Coordinates of the first spin.
        coord_2 (ndarray with shape (3,)): Coordinates of the second spin.
        g1 (float or ndarray with shape (3, 3)): Gyromagnetic ratio of the first spin.
        g2 (float or ndarray with shape (3, 3)): Gyromagnetic ratio of the second spin.

    Returns:
        ndarray with shape (3, 3): Interaction tensor.

    """
    return _inner_dd_tensor(g1,g2)(coord_1, coord_2, g1, g2)

@jit(cache=True, nopython=True)
def dipole_dipole(coord_1, coord_2, g1, g2, ivec_1, ivec_2):
    """
    Compute dipole_dipole interactions between two bath spins.

    Args:
        coord_1 (ndarray with shape (3,)): Coordinates of the first spin.
        coord_2 (ndarray with shape (3,)): Coordinates of the second spin.
        g1 (float): Gyromagnetic ratio of the first spin.
        g2 (float): Gyromagnetic ratio of the second spin.
        ivec_1 (ndarray with shape (3, n, n)): Spin vector of the first spin in the full Hilbert space of the cluster.
        ivec_2 (ndarray with shape (3, n, n)): Spin vector of the second spin in the full Hilbert space of the cluster.

    Returns:
        ndarray with shape (n, n): Dipole-dipole interactions.

    """
    p_tensor = dd_tensor(coord_1, coord_2, g1, g2)
    return vec_tensor_vec(ivec_1, p_tensor, ivec_2)


@jit(cache=True, nopython=True)
def gen_pos_tensor(coord_1, coord_2):
    """
    Generate positional tensor -(3*r @ r.T - r*r), used for hyperfine tensor (without gyro factor).

    Args:
        coord_1 (ndarray with shape (3,)): Coordinates of the first spin.
        coord_2 (ndarray with shape (3,)): Coordinates of the second spin.

    Returns:
        ndarray with shape (3, 3): Positional tensor.

    """
    pos = coord_1 - coord_2
    r = np.linalg.norm(pos)

    return -(3 * np.outer(pos, pos) - np.eye(3, dtype=np.complex128) * r ** 2) / (r ** 5)


def bath_interactions(nspin, ivectors):
    """
    Compute interactions between bath spins.

    Args:
        nspin (BathArray): Array of the bath spins in the given cluster.
        ivectors (array-like): array of expanded spin vectors, each with shape (3, n, n).

    Returns:
        ndarray with shape (n, n): All intrabath interactions of bath spins in the cluster.

    """

    nnuclei = len(nspin)
    imap = nspin.imap
    dd = 0
    if imap is None:
        for i in range(nnuclei):
            for j in range(i + 1, nnuclei):
                n1 = nspin[i]
                n2 = nspin[j]

                ivec_1 = ivectors[i]
                ivec_2 = ivectors[j]
                dd += dipole_dipole(n1['xyz'], n2['xyz'], n1.gyro, n2.gyro, ivec_1, ivec_2)
    else:
        for i in range(nnuclei):
            for j in range(i + 1, nnuclei):
                n1 = nspin[i]
                n2 = nspin[j]

                ivec_1 = ivectors[i]
                ivec_2 = ivectors[j]

                try:
                    tensor = imap[i, j]
                    dd += vec_tensor_vec(ivec_1, tensor, ivec_2)
                except KeyError:
                    dd += dipole_dipole(n1['xyz'], n2['xyz'], n1.gyro, n2.gyro, ivec_1, ivec_2)
    return dd


@jit(cache=True, nopython=True)
def bath_mediated(hyperfines, ivectors, energy_state, energies, projections):
    r"""
    Compute all hyperfine-mediated interactions between bath spins.

    Args:
        hyperfines (ndarray with shape (n, 3, 3)): Array of hyperfine tensors of the bath spins in the given cluster.
        ivectors (array-like): array of expanded spin vectors, each with shape (3, n, n).
        energy_state (float): Energy of the qubit state on which the interaction is conditioned.
        energies (ndarray with shape (2s - 1,)): Array of energies of all states of the central spin.
        projections (ndarray with shape (2s - 1, 3)):
            Array of vectors of the central spin matrix elements of form:

            .. math::

                [\bra{i}\hat{S}_x\ket{j}, \bra{i}\hat{S}_y\ket{j}, \bra{i}\hat{S}_z\ket{j}],

            where  :math:`\ket{i}` are different states of the central spin.

    Returns:
        ndarray with shape (n, n): Hyperfine-mediated interactions.

    """
    mediated = np.zeros(ivectors[0, 0].shape, dtype=np.complex128)

    others_mask = energies != energy_state
    energies = energies[others_mask]
    projections = projections[others_mask]

    for energy_j, s_ij in zip(energies, projections):
        element_ij = np.zeros(ivectors[0, 0].shape, dtype=np.complex128)
        element_ji = np.zeros(ivectors[0, 0].shape, dtype=np.complex128)

        for hf, ivec in zip(hyperfines, ivectors):
            element_ij += conditional_hyperfine(hf, ivec, s_ij)
            element_ji += conditional_hyperfine(hf, ivec, s_ij.conj())
        mediated += element_ij @ element_ji / (energy_state - energy_j)

    return mediated


@jit(cache=True, nopython=True)
def conditional_hyperfine(hyperfine_tensor, ivec, projections):
    r"""
    Compute conditional hyperfine Hamiltonian.

    Args:
        hyperfine_tensor (ndarray with shape (3, 3)): Tensor of hyperfine interactions of the bath spin.
        ivec (ndarray with shape (3, n, n)): Spin vector of the bath spin in the full Hilbert space of the cluster.
        projections (ndarray with shape (3,)):
            Array of vectors of the central spin matrix elements of form:

            .. math::

                [\bra{i}\hat{S}_x\ket{j}, \bra{i}\hat{S}_y\ket{j}, \bra{i}\hat{S}_z\ket{j}],

            where :math:`\ket{j}` are different states of the central spin.
            If :math:`\ket{i} = \ket{j}`, produces the usual conditioned hyperfine interactions
            and just equal to projections of :math:`\hat{S}_z` of the central spin state
            :math:`[\braket{\hat{S}_x}, \braket{\hat{S}_y}, \braket{\hat{S}_z}]`.

            If :math:`\ket{i} \neq \ket{j}`, gives second order perturbation.

    Returns:
        ndarray with shape (n, n): Conditional hyperfine interaction.

    """

    a_ivec = tensor_vdot(hyperfine_tensor, ivec)
    return projections[0] * a_ivec[0] + projections[1] * a_ivec[1] + projections[2] * a_ivec[2]


@jit(cache=True, nopython=True)
def hyperfine(hyperfine_tensor, svec, ivec):
    """
    Compute hyperfine interactions between central spin and bath spin.

    Args:
        hyperfine_tensor (ndarray with shape (3, 3)): Tensor of hyperfine interactions of the bath spin.
        svec (ndarray with shape (3, n, n)): Spin vector of the central spin in the full Hilbert space of the cluster.
        ivec (ndarray with shape (3, n, n)): Spin vector of the bath spin in the full Hilbert space of the cluster.

    Returns:
        ndarray with shape (n, n): Hyperfine interaction.

    """
    return vec_tensor_vec(svec, hyperfine_tensor, ivec)


@jit(cache=True, nopython=True)
def self_central(svec, mfield, tensor, gyro=ELECTRON_GYRO, detuning=0):
    r"""
    Function to compute the central spin term in the Hamiltonian.

    Args:
        svec (ndarray with shape (3, n, n)): Spin vector of the central spin in the full Hilbert space of the cluster.
        mfield (ndarray wtih shape (3,): Magnetic field of type ``mfield = np.array([Bx, By, Bz])``.
        tensor (ndarray with shape (3, 3)):
            Zero Field Splitting tensor of the central spin.
        gyro (float or ndarray with shape (3,3)):
            gyromagnetic ratio of the central spin OR tensor corresponding to interaction between magnetic field and
            central spin.
        detuning (float): Energy detuning from the Zeeman splitting in kHz.

    Returns:
        ndarray with shape (n, n): Central spin term.

    """
    hzeeman = zeeman(svec, gyro, mfield)
    hself = np.zeros(hzeeman.shape, dtype=hzeeman.dtype)

    if svec[2, 0, 0].real > 0.5:
        hself = vec_tensor_vec(svec, tensor, svec)

    if detuning:
        hself += detuning * svec[2]

    return hself + hzeeman


def center_interactions(center, vectors):
    r"""
    Compute interactions between central spins.

    Args:
        center (CenterArray): Array of central spins
        vectors (ndarray with shape (x, 3, n, n)): Array of spin vectors of central spins.

    Returns:
        ndarray with shape (n, n): Central spin Overhauser term.

    """
    ncenters = len(center)
    if (ncenters == 1) or (not center.imap):
        return 0

    ham = 0

    for i in range(ncenters):
        for j in range(i + 1, ncenters):
            try:
                vec_1 = vectors[i]
                vec_2 = vectors[j]
                tensor = center.imap[i, j]
                ham += vec_tensor_vec(vec_1, tensor, vec_2)
            except KeyError:
                pass
    return ham


@jit(cache=True, nopython=True)
def overhauser_central(svec, others_hyperfines, others_state):
    """
    Compute Overhauser field term on the central spin from all other spins, not included in the cluster.

    Args:
        svec (ndarray with shape (3, n, n)): Spin vector of the central spin in the full Hilbert space of the cluster.
        others_hyperfines (ndarray with shape (m, 3, 3)):
            Array of hyperfine tensors for all bath spins not included in the cluster.
        others_state (ndarray with shape (m,) or (m, 3)):
            Array of :math:`I_z` projections for each bath spin outside of the given cluster.

    Returns:
        ndarray with shape (n, n): Central spin Overhauser term.

    """

    zfield = (others_hyperfines[..., 2, 2] * others_state).sum()

    return zfield * svec[2]


@jit(cache=True, nopython=True)
def overhauser_bath(ivec, position, gyro,
                    other_gyros, others_position, others_state):
    """
    Compute Overhauser field term on the bath spin in the cluster from all other spins, not included in the cluster.

    Args:
        ivec (ndarray with shape (3, n, n)): Spin vector of the bath spin in the full Hilbert space of the cluster.
        position (ndarray with shape (3,)): Position of the bath spin.
        gyro (float): Gyromagnetic ratio of the bath spin.
        other_gyros (ndarray with shape (m,)):
            Array of the gyromagnetic ratios of the bath spins, not included in the cluster.
        others_position (ndarray with shape (m, 3)):
            Array of the positions of the bath spins, not included in the cluster.
        others_state (ndarray with shape (m,) or (m, 3)):
            Array of :math:`I_z` projections for each bath spin outside of the given cluster.

    Returns:
        ndarray with shape (n, n): Bath spin Overhauser term.

    """

    pre = np.asarray(gyro * other_gyros * HBAR_MU0_O4PI / PI2)

    pos = position - others_position
    r = np.sqrt((pos ** 2).sum(axis=-1))
    zfield = (pre / r ** 5 * (r ** 2 - 3 * pos[..., 2] ** 2) * others_state).sum()

    return zfield * ivec[2]  # + xfield * ivec[0] + yfield * ivec[1]


@jit(cache=True, nopython=True)
def overhauser_from_tensors(vec, tensors, projected_state):
    """
    Compute Overhauser field from array of tensors.

    Args:
        vec (ndarray with shape (3, n, n)): Spin vector of the bath spin in the full Hilbert space of the cluster.

        tensors (ndarray with shape (N, 3, 3)): Array of interaction tensors.

        projected_state (ndarray with shape (N, )):
            Array of :math:`I_z` projections of the spins outside of the given cluster..

    Returns:
        ndarray with shape (n, n): Bath spin Overhauser term.

    """
    return (tensors[:, 2, 2] * projected_state).sum() * vec[2]


def projected_addition(vectors, bath, center, state):
    r"""
    Compute the first order addition of the interactions with the cental spin to the cluster Hamiltonian.

    Args:
        vectors (array-like): Array of expanded spin vectors, each with shape (3, n, n).
        bath (BathArray): Array of bath spins.
        center (CenterArray): Array of central spins.
        state (str, bool, or array-like): Identificator of the qubit spin. ``'alpha'`` or ``True``
            for :math:`\ket{0}` state, ``'beta'`` of ``False`` for :math:`\ket{1}` state.

    Returns:
        ndarray with shape (n, n): Addition to the Hamiltonian.
    """
    ncenters = len(center)
    addition = 0

    for ivec, n in zip(vectors, bath):
        try:

            iterator = iter(state)  # iterable

        except TypeError:
            for i, c in enumerate(center):
                if ncenters > 1:
                    hf = n.A[i]
                else:
                    hf = n.A
                projections = c.get_projections(state)
                addition += conditional_hyperfine(hf, ivec, projections)
        else:

            for i, (c, s) in enumerate(zip(center, iterator)):
                if ncenters > 1:
                    hf = n.A[i]
                else:
                    hf = n.A

                projections = c.get_projections(s)
                addition += conditional_hyperfine(hf, ivec, projections)

    energy = center.get_energy(state)

    if energy is not None:
        for i, c in enumerate(center):
            if ncenters > 1:
                hf = bath.A[:, i]
            else:
                hf = bath.A

            projections_state_all = c.get_projections_all(state)

            addition += bath_mediated(hf, vectors, energy,
                                      center.energies, projections_state_all)

    return addition


def center_external_addition(vectors, cluster, outer_spin, outer_state):
    """
    Compute the first order addition of the interactions between central spin and external bath spins
    to the cluster Hamiltonian.

    Args:
        vectors (array-like): Array of expanded spin vectors, each with shape (3, n, n).
        cluster (BathArray): Array of cluster spins.
        outer_spin (BathArray with shape (o, )): Array of the spins outside the cluster.
        outer_state (ndarray with shape (o, )): Array of the :math:`S_z` projections of the external bath spins.

    Returns:
        ndarray with shape (n, n): Addition to the Hamiltonian.
    """
    addition = 0
    ncenters = vectors.shape[0] - cluster.size

    for i, v in enumerate(vectors[cluster.size:]):
        if ncenters == 1:
            hf = outer_spin.A
        else:
            hf = outer_spin.A[:, i]

        addition += overhauser_central(v, hf, outer_state)

    return addition


def bath_external_point_dipole(vectors, cluster, outer_spin, outer_state):
    """
    Compute the first order addition of the point-dipole interactions between cluster spins and external bath spins
    to the cluster Hamiltonian.

    Args:
        vectors (array-like): Array of expanded spin vectors, each with shape (3, n, n).
        cluster (BathArray): Array of cluster spins.
        outer_spin (BathArray with shape (o, )): Array of the spins outside the cluster.
        outer_state (ndarray with shape (o, )): Array of the :math:`S_z` projections of the external bath spins.

    Returns:
        ndarray with shape (n, n): Addition to the Hamiltonian.
    """
    addition = 0
    for ivec, n in zip(vectors, cluster):
        addition += overhauser_bath(ivec, n.xyz, n.gyro, outer_spin.gyro,
                                    outer_spin.xyz, outer_state)
    return addition


def external_spins_field(vectors, indexes, bath, projected_state):
    """
    Compute the first order addition of the point-dipole interactions between cluster spins and external bath spins
    to the cluster Hamiltonian.

    Args:
        vectors (array-like): Array of expanded spin vectors, each with shape (3, n, n).
        indexes (ndarray with shape (n,)): Array of indexes of bath spins inside the given cluster.
        bath (BathArray with shape (N,)): Array of all bath spins.
        projected_state (ndarray with shape (N, ):  Array of the :math:`S_z` projections of all bath spins.

    Returns:
        ndarray with shape (n, n): Addition to the Hamiltonian.
    """
    outer_mask = np.ones(bath.size, dtype=bool)
    outer_mask[indexes] = False

    outer_spin = bath[outer_mask]
    outer_state = projected_state[outer_mask]
    cluster = bath[indexes]

    addition = center_external_addition(vectors, cluster, outer_spin, outer_state)

    if bath.imap is None:
        addition += bath_external_point_dipole(vectors, cluster, outer_spin, outer_state)

        return addition

    imap_indexes = bath.imap.indexes
    remove_j = np.ones(indexes.shape, dtype=bool)

    for j, (ind, ivec) in enumerate(zip(indexes, vectors)):

        outer_mask[:] = True
        outer_mask[indexes] = False

        where_index = (imap_indexes == ind)

        if where_index.any():
            remove_j[j] = False
            which_pairs = where_index.any(axis=1) & (~np.isin(imap_indexes, indexes[remove_j]).any(axis=1))
            remove_j[j] = True

            other_indexes = imap_indexes[which_pairs][~(where_index[which_pairs])]

            addition += overhauser_from_tensors(ivec, bath.imap.data[which_pairs], projected_state[other_indexes])

            outer_mask[other_indexes] = False

        if outer_mask.any():
            outer_spin = bath[outer_mask]
            outer_state = projected_state[outer_mask]
            n = bath[ind]
            addition += overhauser_bath(ivec, n.xyz, n.gyro, outer_spin.gyro,
                                        outer_spin.xyz, outer_state)

    return addition
