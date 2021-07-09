import numpy as np
from pycce.constants import HBAR, PI2, ELECTRON_GYRO
from pycce.utilities import dimensions_spinvectors, expand


def expanded_single(ivec, gyro, mfield, self_tensor, detuning=.0):
    """
    Function to compute the single bath spin term.

    Args:
        ivec (ndarray with shape (3, n, n)): Spin vector of the bath spin in the full Hilbert space of the cluster.
        gyro (float or ndarray with shape (3, 3)):
        mfield (ndarray wtih shape (3,): Magnetic field of type ``mfield = np.array([Bx, By, Bz])``.
        self_tensor (ndarray with shape (3, 3)): tensor of self-interaction of type IPI where I is bath spin.
        detuning (float): Additional term of d*Iz allowing to simulate different energy splittings of bath spins.

    Returns:
        ndarray with shape (n, n): Single bath spin term.

    """
    if isinstance(gyro, (float, int)):
        hzeeman = -gyro / PI2 * (mfield[0] * ivec[0] + mfield[1] * ivec[1] + mfield[2] * ivec[2])
    # else assume tensor
    else:
        gsvec = np.einsum('ij,jkl->ikl', gyro / PI2, ivec, dtype=np.complex128)
        hzeeman = np.einsum('lij,ljk->ik', mfield, gsvec, dtype=np.complex128)

    hself = 0

    if ivec[2, 0, 0] > 0.5:
        v_ivec = np.einsum('ij,jkl->ikl', self_tensor, ivec, dtype=np.complex128)
        hself = np.einsum('lij,ljk->ik', ivec, v_ivec, dtype=np.complex128)

    if detuning:
        hself += detuning * ivec[2]

    return hself + hzeeman


# def bath_single(bath, vectors, mfield):
#     """
#     Compute isolated bath spin terms for all spins in the bath
#
#     Args:
#         bath (BathArray): Array of the bath spins in the given cluster.
#         vectors (array-like): array of expanded spin vectors, each with shape (3, n, n).
#         mfield (ndarray wtih shape (3,): Magnetic field of type ``mfield = np.array([Bx, By, Bz])``.
#
#     Returns:
#         ndarray with shape (n, n): All single bath spin terms.
#
#     """
#     hsingle = 0
#
#     for j, n in enumerate(bath):
#         ivec = vectors[j]
#         hsingle += expanded_single(ivec, n.gyro, mfield, n['Q'], n.detuning)
#
#     return hsingle


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

    pre = g1 * g2 * HBAR / PI2

    pos = coord_1 - coord_2
    r = np.linalg.norm(pos)

    p_tensor = -pre * (3 * np.outer(pos, pos) -
                       np.eye(3, dtype=np.complex128) * r ** 2) / (r ** 5)

    p_ivec = np.einsum('ij,jkl->ikl', p_tensor, ivec_2,
                       dtype=np.complex128)  # p_ivec = Ptensor @ Ivector
    # DD = IPI = IxPxxIx + IxPxyIy + ..
    hdd = np.einsum('lij,ljk->ik', ivec_1, p_ivec, dtype=np.complex128)

    return hdd


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
                    tensor_ivec = np.einsum('ij,jkl->ikl', tensor, ivec_2,
                                            dtype=np.complex128)  # p_ivec = Ptensor @ Ivector
                    dd += np.einsum('lij,ljk->ik', ivec_1, tensor_ivec, dtype=np.complex128)
                except KeyError:
                    dd += dipole_dipole(n1['xyz'], n2['xyz'], n1.gyro, n2.gyro, ivec_1, ivec_2)
    return dd


def bath_mediated(nspin, ivectors, energy_state, energies, projections):
    r"""
    Compute all hyperfine-mediated interactions between bath spins.

    Args:
        nspin (BathArray): Array of the bath spins in the given cluster.
        ivectors (array-like): array of expanded spin vectors, each with shape (3,n,n).
        energy_state (float): Energy of the qubit state on which the interaction is conditioned.
        energies (ndarray with shape (2s-1,)): Array of energies of all states of the central spin.
        projections (ndarray with shape (2s-1, 3)):
            Array of vectors of the central spin matrix elements of form:

            .. math::

                [\bra{i}\hat{S}_x\ket{j}, \bra{i}\hat{S}_y\ket{j}, \bra{i}\hat{S}_z\ket{j}],

            where  :math:`\ket{i}` are different states of the central spin.

    Returns:
        ndarray with shape (n, n): Hyperfine-mediated interactions.

    """
    mediated = 0

    others_mask = energies != energy_state
    energies = energies[others_mask]
    projections = projections[others_mask]

    for energy_j, s_ij in zip(energies, projections):
        element_ij = 0
        element_ji = 0

        for n, ivec in zip(nspin, ivectors):
            element_ij += conditional_hyperfine(n['A'], ivec, s_ij)
            element_ji += conditional_hyperfine(n['A'], ivec, s_ij.conj())

        mediated += element_ij @ element_ji / (energy_state - energy_j)

    return mediated


# def bath_mediated_old(nspin, ivectors, energy_state,
#                   energies, projections):
#     """
#     Compute all hyperfine mediated interactions between bath spins.
#
#     Args:
#         nspin (BathArray): Array of the bath spins in the given cluster.
#         ivectors (array-like): array of expanded spin vectors, each with shape (3,n,n).
#         energy_state (float): Energy of the qubit state on which the interaction is conditioned.
#         energies (ndarray with shape (2s-1,)): Array of energies of all states of the central spin.
#         projections (ndarray with shape (2s-1, 3)):
#             Array of vectors of the central spin matrix elements of form:
#             [<state|Sx|other>, <state|Sy|other>, <state|Sz|other>],
#             where |state> is the qubit state on which the interaction is conditioned, and |other> are all states.
#
#     Returns:
#         hamiltonian (ndarray with shape (n, n)): hyperfine-mediated interactions.
#
#     """
#     nnuclei = nspin.shape[0]
#     mediated = 0
#
#     others_mask = energies != energy_state
#     energies = energies[others_mask]
#     projections = projections[others_mask]
#
#     for i in range(nnuclei):
#         for j in range(i + 1, nnuclei):
#             n1 = nspin[i]
#             n2 = nspin[j]
#
#             ivec_1 = ivectors[i]
#             ivec_2 = ivectors[j]
#
#             mediated += hyperfine_mediated_old(n1['A'], n2['A'], ivec_1, ivec_2, energy_state, projections, energies)
#     return mediated


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

    return np.einsum('i,ijk->jk', projections @ hyperfine_tensor, ivec)


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
    aivec = np.einsum('ij,jkl->ikl', hyperfine_tensor, ivec)  # AIvec = Atensor @ Ivector
    # HF = SPI = SxPxxIx + SxPxyIy + ..
    h_hf = np.einsum('lij,ljk->ik', svec, aivec)
    return h_hf


def self_central(svec, mfield, zfs=None, gyro=ELECTRON_GYRO):
    """
    Function to compute the central spin term in the Hamiltonian.
 
    Args:
        svec (ndarray with shape (3, n, n)): Spin vector of the central spin in the full Hilbert space of the cluster.
        mfield (ndarray wtih shape (3,): Magnetic field of type ``mfield = np.array([Bx, By, Bz])``.
        zfs (ndarray with shape (3, 3)):
            Zero Field Splitting tensor of the central spin.
        gyro (float or ndarray with shape (3,3)):
            gyromagnetic ratio of the central spin OR tensor corresponding to interaction between magnetic field and
            central spin.

    Returns:
        ndarray with shape (n, n): Central spin term.

    """
    H0 = 0
    if svec[2, 0, 0] > 1 / 2:
        dsvec = np.einsum('ij,jkl->ikl', zfs, svec,
                          dtype=np.complex128)  # AIvec = Atensor @ Ivector
        # H0 = SDS = SxDxxSx + SxDxySy + ..
        H0 = np.einsum('lij,ljk->ik', svec, dsvec, dtype=np.complex128)

    # if gyro is number
    if isinstance(gyro, (np.floating, float, int)):
        # print(svec, mfield)
        H1 = -gyro / PI2 * (mfield[0] * svec[0] + mfield[1] * svec[1] + mfield[2] * svec[2])
    # else assume tensor
    else:
        gsvec = np.einsum('ij,jkl->ikl', gyro / PI2, svec,
                          dtype=np.complex128)  # AIvec = Atensor @ Ivector
        # H0 = SDS = SxDxxSx + SxDxySy + ..
        H1 = np.einsum('lij,ljk->ik', mfield, gsvec, dtype=np.complex128)

    return H1 + H0


def overhauser_central(svec, others_hyperfines, others_state):
    """
    Compute Overhauser field term on the central spin from all other spins, not included in the cluster.

    Args:
        svec (ndarray with shape (3, n, n)): Spin vector of the central spin in the full Hilbert space of the cluster.
        others_hyperfines (ndarray with shape (m, 3, 3)):
            Array of hyperfine tensors for all bath spins not included in the cluster.
        others_state (ndarray with shape (m,) or (m, 3)):
            Array of Iz projections for each bath spin outside of the given cluster.

    Returns:
        ndarray with shape (n, n): Central spin Overhauser term.

    """

    if len(others_state.shape) > 1:
        zfield = np.sum(others_hyperfines[..., 2, 2] * others_state[..., 2])

    else:
        zfield = np.sum(others_hyperfines[..., 2, 2] * others_state)

    return zfield * svec[2]


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
            Array of Iz projections for each bath spin outside of the given cluster.

    Returns:
        ndarray with shape (n, n): Bath spin Overhauser term.

    """
    pre = np.asarray(gyro * other_gyros * HBAR / PI2)

    pos = position - others_position
    r = np.linalg.norm(pos, axis=-1)
    if len(others_state.shape) == 1:
        # xfield = np.sum(pre / r ** 5 * (- 3 * pos[:, 2] * pos[:, 0]) * others_state)
        # yfield = np.sum(pre / r ** 5 * (- 3 * pos[:, 2] * pos[:, 1]) * others_state)
        zfield = np.sum(pre / r ** 5 * (r ** 2 - 3 * pos[:, 2] ** 2) * others_state)
        # Not sure which is more physical yet..
        return zfield * ivec[2] # + xfield * ivec[0] + yfield * ivec[1]

    else:
        posxpos = np.einsum('ki,kj->kij', pos, pos)

        r = r[..., np.newaxis, np.newaxis]
        pre = pre[..., np.newaxis, np.newaxis]
        identity = np.eye(3, dtype=np.float64)
        dd = -(3 * posxpos - identity[np.newaxis, ...] * r ** 2) / (r ** 5) * pre

        field = np.einsum('ij,ijk->k', others_state, dd)

        return np.einsum('k,klm->lm', field, ivec)


def eta_hamiltonian(nspin, central_spin, alpha, beta, eta):
    """
    EXPERIMENTAL. Compute hamiltonian with eta-term - gradually turn off or turn on the secular interactions for
    alpha and beta qubit states.

    Args:
        nspin (BathArray): Array of the bath spins in the given cluster.
        central_spin (float): central spin.
        alpha (ndarray with shape (2s+1,)):
            Vector representation of the alpha qubit state in Sz basis.
        beta (ndarray with shape (2s+1,)):
            Vector representation of the beta qubit state in Sz basis.
        eta (float): Value of dimensionless parameter eta (from 0 to 1).

    Returns:
        ndarray with shape (n, n): Eta term.

    """

    dimensions, vectors = dimensions_spinvectors(nspin, central_spin=central_spin)
    AIzi = 0

    for j, ivec in enumerate(vectors[:-1]):
        AIzi += np.einsum('j,jkl->kl', nspin[j]['A'][2, :], ivec, dtype=np.complex128)

    up_down = (1 - eta) / 2 * (np.tensordot(alpha, alpha, axes=0) + np.tensordot(beta, beta, axes=0))
    H_eta = expand(up_down, nspin.shape[0], dimensions) @ AIzi
    return H_eta
