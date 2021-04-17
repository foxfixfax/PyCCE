from pycce.sm import _smc
from pycce.units import HBAR, ELECTRON_GYRO
from pycce.utilities import *


# HBAR = 1.05457172  # When everything else in rad, kHz, ms, G, A


def expanded_single(ivec, gyro, mfield, self_tensor):
    if isinstance(gyro, (float, int)):
        hzeeman = -gyro * (mfield[0] * ivec[0] + mfield[1] * ivec[1] + mfield[2] * ivec[2])
    # else assume tensor
    else:
        gsvec = np.einsum('ij,jkl->ikl', gyro, ivec, dtype=np.complex128)
        hzeeman = np.einsum('lij,ljk->ik', mfield, gsvec, dtype=np.complex128)
    hself = 0
    if ivec[2, 0, 0] > 0.5:
        v_ivec = np.einsum('ij,jkl->ikl', self_tensor, ivec, dtype=np.complex128)
        hself = np.einsum('lij,ljk->ik', ivec, v_ivec, dtype=np.complex128)

    return hself + hzeeman


def dipole_dipole(coord_1, coord_2, g1, g2, ivec_1, ivec_2):
    """
    Compute dipole_dipole interactions between two bath spins
    :param coord_1: ndarray with shape (3,)
        coordinates of the first spin
    :param coord_2: ndarray with shape (3,)
        coordinates of the second spin
    :param g1: float
        gyromagnetic ratio of the first spin
    :param g2: float
        gyromagnetic ratio of the second spin
    :param ivec_1: ndarray with shape (3, d, d)
        where d is total size of the Hilbert space. Vector of [Ix, Iy, Iz] for the first bath spin
    :param ivec_2: ndarray with shape (3, d, d)
        where d is total size of the Hilbert space. Vector of [Ix, Iy, Iz] for the second bath spin
    :return: ndarray of shape (d, d)
    dipole dipole interactions of two bath spins
    """
    pre = g1 * g2 * HBAR

    pos = coord_1 - coord_2
    r = np.linalg.norm(pos)

    p_tensor = -pre * (3 * np.outer(pos, pos) -
                       np.eye(3, dtype=np.complex128) * r ** 2) / (r ** 5)

    p_ivec = np.einsum('ij,jkl->ikl', p_tensor, ivec_2,
                       dtype=np.complex128)  # p_ivec = Ptensor @ Ivector
    # DD = IPI = IxPxxIx + IxPxyIy + ..
    hdd = np.einsum('lij,ljk->ik', ivec_1, p_ivec, dtype=np.complex128)

    return hdd


def bath_interactions(nspin, ivectors, imap=None, raise_error=False):
    """
    Compute interactions between bath spins
    :param nspin: BathArray
        array of bath spins
    :param ivectors: array-like
        array of expanded spin vectors
    :param imap: InteractionMap
        optional. dictionary-like object containing tensors for all bath spin pairs
    :param raise_error: bool
        optional. If true and imap is not None, raises error when cannot find pair of nuclear spins in imap. Default
        False
    :return: ndarray of shape (d, d)
    bath interactions of bath spins in the cluster
    """
    nnuclei = nspin.shape[0]
    ntype = nspin.types

    dd = 0
    if imap is None:
        for i in range(nnuclei):
            for j in range(i + 1, nnuclei):
                n1 = nspin[i]
                n2 = nspin[j]

                ivec_1 = ivectors[i]
                ivec_2 = ivectors[j]

                dd += dipole_dipole(n1['xyz'], n2['xyz'], ntype[n1].gyro, ntype[n2].gyro, ivec_1, ivec_2)
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
                    if raise_error:
                        raise KeyError("InteractionMap doesn't contain all spin pairs."
                                       " You might try setting raise_error=False instead")
                    else:
                        dd += dipole_dipole(n1['xyz'], n2['xyz'], ntype[n1].gyro, ntype[n2].gyro, ivec_1, ivec_2)
    return dd


def hyperfine_mediated(atensor_1, atensor_2, ivec_1, ivec_2, energy, projections_state_others, others_energies):
    hamil = 0

    for energy_j, s_ij in zip(others_energies, projections_state_others):
        first = np.einsum('i,ijk->jk', s_ij @ atensor_1, ivec_1)
        second = np.einsum('i,ijk->jk', s_ij.conj() @ atensor_2, ivec_2)
        hamil += first @ second / (energy - energy_j)

    return hamil


def bath_mediated(nspin, ivectors, energy_state,
                  energies, projections):
    nnuclei = nspin.shape[0]
    mediated = 0

    others_mask = energies != energy_state
    energies = energies[others_mask]
    projections = projections[others_mask]

    for i in range(nnuclei):
        for j in range(i + 1, nnuclei):
            n1 = nspin[i]
            n2 = nspin[j]

            ivec_1 = ivectors[i]
            ivec_2 = ivectors[j]

            mediated += hyperfine_mediated(n1['A'], n2['A'], ivec_1, ivec_2, energy_state, projections, energies)
    return mediated


def conditional_hyperfine(hyperfine_tensor, ivec, projections):
    """
    Compute projected hyperfine Hamiltonian for one state of the central spin
    :param hyperfine_tensor: np.array of shape (3,3)
        hyperfine interactions of n spin
    :param ivec: ndarray with shape (3, d, d)
        d is the total size of the Hilbert space. [Ix, Iy, Iz] array of the bath spin
    :param projections: np.ndarray of shape (3,)
        projections of the central spin qubit levels [<Sx>, <Sy>, <Sz>]
    :return: ndarray of shape (d, d)
    """
    aprojected = projections @ hyperfine_tensor
    hyperfine = (aprojected[0] * ivec[0] +
                 aprojected[1] * ivec[1] +
                 aprojected[2] * ivec[2])

    return hyperfine


def hyperfine(hyperfine_tensor, svec, ivec):
    """
    Compute hyperfine interactions between central spin spin_matrix and bath spin I
    Compute projected hyperfine Hamiltonian for one state of the central spin
    :param hyperfine_tensor: np.array of shape (3,3)
        hyperfine interactions of n spin
    :param svec: ndarray with shape (3, d, d)
        d is the total size of the Hilbert space. [Sx, Sy, Sz] array of the central spin
    :param ivec: ndarray with shape (3, d, d)
        d is the total size of the Hilbert space. [Ix, Iy, Iz] array of the bath spin
    :return: ndarray
    """
    aivec = np.einsum('ij,jkl->ikl', hyperfine_tensor, ivec)  # AIvec = Atensor @ Ivector
    # HF = SPI = SxPxxIx + SxPxyIy + ..
    H_HF = np.einsum('lij,ljk->ik', svec, aivec)
    return H_HF


def self_central(svec, mfield, D=None, gyro=ELECTRON_GYRO):
    """
    central spin Hamiltonian
    :param mfield: ndarray with shape (3,)
        magnetic field of format (Bx, By, Bz)
    :param s: float
        Total spin of the central spin
    :param gyro: float
        gyromagnetic ratio (in rad/(ms*Gauss)) of the central spin
    :param D: float or ndarray with shape (3,3)
        D parameter in central spin ZFS OR total ZFS tensor
    :return: ndarray
    """
    H0 = 0
    if svec[2, 0, 0] > 1 / 2:
        dsvec = np.einsum('ij,jkl->ikl', D, svec,
                          dtype=np.complex128)  # AIvec = Atensor @ Ivector
        # H0 = SDS = SxDxxSx + SxDxySy + ..
        H0 = np.einsum('lij,ljk->ik', svec, dsvec, dtype=np.complex128)

    # if gyro is number
    if isinstance(gyro, (np.floating, float, int)):
        H1 = -gyro * (mfield[0] * svec[0] + mfield[1] * svec[1] + mfield[2] * svec[2])
    # else assume tensor
    else:
        gsvec = np.einsum('ij,jkl->ikl', gyro, svec,
                          dtype=np.complex128)  # AIvec = Atensor @ Ivector
        # H0 = SDS = SxDxxSx + SxDxySy + ..
        H1 = np.einsum('lij,ljk->ik', mfield, gsvec, dtype=np.complex128)

    return H1 + H0


def overhauser_central(svec, others_hyperfines, others_state):
    if len(others_state.shape) > 1:
        zfield = np.sum(others_hyperfines[:, 2, 2] * others_state[:, 2])
    else:
        zfield = np.sum(others_hyperfines[:, 2, 2] * others_state)
    return zfield * svec[2]


def overhauser_bath(ivec, position, gyro,
                    other_gyros, others_position, others_state):
    pre = gyro * other_gyros * HBAR

    pos = position - others_position
    r = np.linalg.norm(pos, axis=1)
    if len(others_state.shape) == 1:
        # if not check:
        #     cos_theta = pos[:, 2] / r
        #     zfield = np.sum(pre / r ** 3 * (1 - 3 * cos_theta ** 2) * others_state)
        #     return zfield * ivec[2]
        # else:
        xfield = np.sum(pre / r ** 5 * (- 3 * pos[:, 2] * pos[:, 0]) * others_state)
        yfield = np.sum(pre / r ** 5 * (- 3 * pos[:, 2] * pos[:, 1]) * others_state)
        zfield = np.sum(pre / r ** 3 * (1 - 3 * pos[:, 2] ** 2 / r ** 2) * others_state)

        return xfield * ivec[0] + yfield * ivec[1] + zfield * ivec[2]


    else:
        posxpos = np.einsum('ki,kj->kij', pos, pos)

        r = r[:, np.newaxis, np.newaxis]
        pre = pre[:, np.newaxis, np.newaxis]
        identity = np.eye(3, dtype=np.float64)
        dd = -(3 * posxpos - identity[np.newaxis, :, :] * r ** 2) / (r ** 5) * pre
        # print(dd.shape)
        field = np.einsum('ij,ijk->k', others_state, dd)

        return np.einsum('k,klm->lm', field, ivec)


def eta_hamiltonian(nspin, central_spin, alpha, beta, eta):
    """
    EXPERIMENTAL. Compute hamiltonian with eta-term - gradually turn off or turn on the secular interactions for
    state and beta qubit states
    :param nspin: ndarray with shape (n,)
        ndarray of bath spins in the given cluster with size n
    :param central_spin: float
        total spin of the central spin
    :param alpha: np.ndarray with shape (2s+1,)
        state state of the qubit
    :param beta: np.ndarray with shape (2s+1,)
        beta state of the qubit
    :param eta: value of dimensionless parameter eta (from 0 to 1)
    :return:
    """

    dimensions, vectors = dimensions_spinvectors(nspin, central_spin=central_spin)
    AIzi = 0

    for j, ivec in enumerate(vectors[:-1]):
        AIzi += np.einsum('j,jkl->kl', nspin[j]['A'][2, :], ivec, dtype=np.complex128)

    up_down = (1 - eta) / 2 * (np.tensordot(alpha, alpha, axes=0) + np.tensordot(beta, beta, axes=0))
    H_eta = expand(up_down, nspin.shape[0], dimensions) @ AIzi
    return H_eta
