import operator

import numpy as np
from pycce.cluster_expansion import cluster_expansion_decorator
from pycce.hamiltonian import expand, zeeman, projected_hyperfine, mf_hamiltonian
from pycce.hamiltonian import total_hamiltonian, dipole_dipole, quadrupole, generate_dimensions
from pycce.sm import _smc

from .density_matrix import propagator_dm
from .mean_field_dm import generate_dm0


def correlation_it_j0(operator_i, operator_j, dm0_expanded, U):
    """
    compute correlation function of the operator i at time t and operator j at time 0
    @param operator_i: ndarray
        matrix representation of operator i
    @param operator_j: ndarray
        matrix representation of operator j
    @param dm0_expanded: ndarray
        initial density matrix of the cluster
    @param U: ndarray
        propagator
    @return: corr
        1D-ndarray of autocorrelation
    """
    operator_i_t = np.matmul(np.transpose(U.conj(), axes=(0, 2, 1)), np.matmul(operator_i, U))
    it_j0 = np.matmul(operator_i_t, operator_j)
    matmul = np.matmul(dm0_expanded, it_j0)
    corr = matmul.trace(axis1=1, axis2=2, dtype=np.complex128)

    return corr


@cluster_expansion_decorator(result_operator=operator.iadd,
                             contribution_operator=operator.imul,
                             removal_operator=operator.isub,
                             addition_operator=np.sum)
def decorated_noise_correlation(cluster, allspin, dm0, B, D, E,
                                timespace,
                                gyro_e=-17608.597050):
    """
    EXPERIMENTAL Decorated function to compute noise correlation with gCCE (without mean field)
    @param subclusters: dict
        dict of subclusters included in different CCE order
        of structure {int order: np.array([[i,j],[i,j]])}
    @param allnspin: ndarray
        array of all bath
    @param ntype: dict
        dict with NSpinType objects inside, each key - name of the isotope
    @param dm0: ndarray
        initial density matrix of the central spin
    @param I: dict
        dict with SpinMatrix objects inside, each key - spin
    @param S: QSpinMatrix
        QSpinMatrix of the central spin
    @param B: ndarray
        Magnetic field of B = np.array([Bx, By, Bz])
    @param D: float
        D parameter in central spin ZFS
    @param E: float
        E parameter in central spin ZFS
    @param timespace: ndarray
        Time points at which to compute autocorrelation
    @param gyro_e: float
        gyromagnetic ratio (in rad/(msec*Gauss)) of the central spin
    @return: ndarray
        autocorrelation function
    """
    nspin = allspin[cluster]
    central_spin = (dm0.shape[0] - 1) / 2

    H, dimensions = total_hamiltonian(nspin, central_spin, B, D, E=E, central_gyro=gyro_e)
    U = propagator_dm(timespace, H, 0, None, None, dimensions)
    dm0_expanded = expand(dm0, len(dimensions) - 1, dimensions) / np.prod(dimensions[:-1])

    AIs = 0
    ntype = nspin.types
    for j, n in enumerate(nspin):
        s = ntype[n].s

        ivec = np.array([expand(_smc[s].x, j, dimensions),
                         expand(_smc[s].y, j, dimensions),
                         expand(_smc[s].z, j, dimensions)],
                        dtype=np.complex128)

        atensor = n['A']

        a_ivec = np.array([atensor[0, 0] * ivec[0], atensor[1, 1] * ivec[1], atensor[2, 2] * ivec[2]])
        AIs += a_ivec

    AI_x = correlation_it_j0(AIs[0], AIs[0], dm0_expanded, U)
    AI_y = correlation_it_j0(AIs[1], AIs[1], dm0_expanded, U)
    AI_z = correlation_it_j0(AIs[2], AIs[2], dm0_expanded, U)

    return np.array([AI_x, AI_y, AI_z])


@cluster_expansion_decorator(result_operator=operator.iadd,
                             contribution_operator=operator.imul,
                             removal_operator=operator.isub,
                             addition_operator=np.sum)
def mean_field_noise_correlation(cluster, allspin, dm0, B, D, E,
                                           timespace, bath_state, gyro_e=-17608.597050):
    """
    Decorated function to compute noise autocorrelation function
    with gCCE and MC sampling of the bath states
    @param subclusters: dict
        dict of subclusters included in different CCE order
        of structure {int order: np.array([[i,j],[i,j]])}
    @param allnspin: ndarray
        array of all bath
    @param ntype: dict
        dict with NSpinType objects inside, each key - name of the isotope
    @param dm0: ndarray
        initial density matrix of the central spin
    @param B: ndarray
        Magnetic field of B = np.array([Bx, By, Bz])
    @param D: float
        D parameter in central spin ZFS
    @param E: float
        E parameter in central spin ZFS
    @param timespace: ndarray
        Time points at which to compute
    @param bath_state: list
        List of nuclear spin states. if len(shape) == 1, contains Sz projections of nuclear spins.
        Otherwise, contains array of initial dms of nuclear spins
    @param gyro_e: float
        gyromagnetic ratio (in rad/(msec*Gauss)) of the central spin
    @return: ndarray
        autocorrelation function
    """
    central_spin = (dm0.shape[0] - 1) / 2
    nspin = allspin[cluster]
    others_mask = np.ones(allspin.shape, dtype=bool)
    others_mask[cluster] = False
    others = allspin[others_mask]
    others_state = bath_state[others_mask]

    states = bath_state[others_mask]
    H, dimensions = mf_hamiltonian(nspin, central_spin, B, others, others_state, D, E, gyro_e)
    U = propagator_dm(timespace, H, 0, None, None, dimensions)
    dmtotal0 = generate_dm0(dm0, dimensions, states)

    AIs = 0
    ntype = nspin.types

    for j, n in enumerate(nspin):
        s = ntype[n].s

        ivec = np.array([expand(_smc[s].x, j, dimensions),
                         expand(_smc[s].y, j, dimensions),
                         expand(_smc[s].z, j, dimensions)],
                        dtype=np.complex128)

        atensor = n['A']

        a_ivec = np.array([atensor[0, 0] * ivec[0], atensor[1, 1] * ivec[1], atensor[2, 2] * ivec[2]])
        AIs += a_ivec

    AI_x = correlation_it_j0(AIs[0], AIs[0], dmtotal0, U)
    AI_y = correlation_it_j0(AIs[1], AIs[1], dmtotal0, U)
    AI_z = correlation_it_j0(AIs[2], AIs[2], dmtotal0, U)

    return np.array([AI_x, AI_y, AI_z])


@cluster_expansion_decorator(result_operator=operator.iadd, contribution_operator=operator.imul)
def projected_noise_correlation(cluster, allspin, projections_state, B, timespace):
    """
    Decorated function to compute autocorrelation function with conventional CCE
    @param subclusters: dict
        dict of subclusters included in different CCE order
        of structure {int order: np.array([[i,j],[i,j]])}
    @param allnspin: ndarray
        array of all bath
    @param ntype: dict
        dict with NSpinType objects inside, each key - name of the isotope
    @param I: dict
        dict with SpinMatrix objects inside, each key - spin
    @param S: QSpinMatrix
        QSpinMatrix of the central spin
    @param B: ndarray
        Magnetic field of B = np.array([Bx, By, Bz])
    @param timespace: ndarray
        Time points at which to compute autocorrelation function
    @return: ndarray
        autocorrelation function
    """
    nspin = allspin[cluster]
    ntype = nspin.types

    dimensions = generate_dimensions(nspin)
    nnuclei = nspin.shape[0]

    tdim = np.prod(dimensions, dtype=np.int32)

    H = np.zeros((tdim, tdim), dtype=np.complex128)
    AIs = 0
    ivectors = []

    for j, n in enumerate(nspin):
        s = ntype[n].s

        if s > 1 / 2:
            H_quad = quadrupole(n['Q'], s)
            H_single = zeeman(ntype[n].gyro, s, B) + H_quad
        else:
            H_single = zeeman(ntype[n].gyro, s, B)

        H_HF = projected_hyperfine(n['A'], s, projections_state)

        H_j = H_single + H_HF
        H += expand(H_j, j, dimensions)

        ivec = np.array([expand(_smc[s].x, j, dimensions),
                         expand(_smc[s].y, j, dimensions),
                         expand(_smc[s].z, j, dimensions)],
                        dtype=np.complex128)

        ATensor = n['A']

        # AIvec = np.einsum('ij,jkl->ikl', ATensor, ivec,
        #                   dtype=np.complex128)  # AIvec = Atensor @ Ivector
        # # AIs.append(AIvec)
        AIvec = np.array([ATensor[0, 0] * ivec[0], ATensor[1, 1] * ivec[1], ATensor[2, 2] * ivec[2]])
        AIs += AIvec

        ivectors.append(ivec)

    for i in range(nnuclei):
        for j in range(i + 1, nnuclei):
            n1 = nspin[i]
            n2 = nspin[j]

            ivec_1 = ivectors[i]
            ivec_2 = ivectors[j]

            H_DD = dipole_dipole(n1['xyz'], n2['xyz'], ntype[n1].gyro, ntype[n2].gyro, ivec_1, ivec_2)

            H += H_DD

    eval0, evec0 = np.linalg.eigh(H)

    eigen_exp0 = np.exp(-1j * np.tensordot(timespace,
                                           eval0, axes=0), dtype=np.complex128)

    U = np.matmul(np.einsum('ij,kj->kij', evec0, eigen_exp0,
                            dtype=np.complex128),
                  evec0.conj().T, dtype=np.complex128)
    dm0_expanded = np.eye(tdim) / tdim

    AI_x = correlation_it_j0(AIs[0], AIs[0], dm0_expanded, U)
    AI_y = correlation_it_j0(AIs[1], AIs[1], dm0_expanded, U)
    AI_z = correlation_it_j0(AIs[2], AIs[2], dm0_expanded, U)

    return np.array([AI_x, AI_y, AI_z])
