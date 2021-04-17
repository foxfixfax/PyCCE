import numpy as np
import operator

from pycce.cluster_expansion import cluster_expansion_decorator
from pycce.hamiltonian import total_hamiltonian, mean_field_hamiltonian, bath_interactions, expanded_single, \
    conditional_hyperfine, \
    dimensions_spinvectors
from pycce.units import ELECTRON_GYRO
from .density_matrix import propagator, generate_dm0, gen_density_matrix
from .mean_field_dm import generate_bath_state
from ..sm import _smc


def correlation_it_j0(operator_i, operator_j, dm0_expanded, U):
    """
    compute correlation function of the operator i at time t and operator j at time 0
    :param operator_i: ndarray
        matrix representation of operator i
    :param operator_j: ndarray
        matrix representation of operator j
    :param dm0_expanded: ndarray
        initial density matrix of the cluster
    :param U: ndarray
        propagator
    :return: corr
        1D-ndarray of autocorrelation
    """
    operator_i_t = np.matmul(np.transpose(U.conj(), axes=(0, 2, 1)), np.matmul(operator_i, U))
    # operator_j_t = np.matmul(np.transpose(U.conj(), axes=(0, 2, 1)), np.matmul(operator_j, U))
    it_j0 = np.matmul(operator_i_t, operator_j)  # + np.matmul(operator_j, operator_i_t)) / 2
    matmul = np.matmul(dm0_expanded, it_j0)
    corr = matmul.trace(axis1=1, axis2=2, dtype=np.complex128)

    return corr


def compute_correlations(nspin, dm0_expanded, U, central_spin=None):
    a_is = np.zeros((3, *U.shape[1:]), dtype=np.complex128)

    dimensions, vectors = dimensions_spinvectors(nspin, central_spin=central_spin)
    for j, n in enumerate(nspin):
        ivec = vectors[j]
        hyperfine_tensor = n['A']
        aivec = np.array([hyperfine_tensor[0, 0] * ivec[0],
                          hyperfine_tensor[1, 1] * ivec[1],
                          hyperfine_tensor[2, 2] * ivec[2]])
        # aivec = np.einsum('ij,jkl->ikl', hyperfine_tensor, ivec)
        a_is += aivec

    # AI_x = correlation_it_j0(AIs[0], AIs[0], dm0_expanded, U)
    # AI_y = correlation_it_j0(AIs[1], AIs[1], dm0_expanded, U)
    AI_z = correlation_it_j0(a_is[2], a_is[2], dm0_expanded, U)

    return AI_z  # np.array([AI_x, AI_y, AI_z])


@cluster_expansion_decorator(result_operator=operator.iadd, contribution_operator=operator.imul)
def projected_noise_correlation(cluster, allspin, projections_state, mfield, timespace, states=None,
                                imap=None, map_error=False):
    """
    Decorated function to compute autocorrelation function with conventional CCE
    """
    bath = allspin[cluster]
    if states is not None:
        states = states[cluster]

    ntype = bath.types

    dimensions, ivectors = dimensions_spinvectors(bath, central_spin=None)

    totalh = 0

    for ivec, n in zip(ivectors, bath):
        hsingle = expanded_single(ivec, ntype[n].gyro, mfield, n['Q'])

        hf_state = conditional_hyperfine(n['A'], ivec, projections_state)

        totalh += hsingle + hf_state

    totalh += bath_interactions(bath, ivectors, imap=imap, raise_error=map_error)
    time_propagator = propagator(timespace, totalh)

    dm0_expanded = gen_density_matrix(states, dimensions=dimensions)

    return compute_correlations(bath, dm0_expanded, time_propagator, central_spin=None)


@cluster_expansion_decorator(result_operator=operator.iadd,
                             contribution_operator=operator.imul,
                             removal_operator=operator.isub,
                             addition_operator=np.sum)
def decorated_noise_correlation(cluster, allspin, dm0, B, D,
                                timespace,
                                gyro_e=ELECTRON_GYRO, states=None):
    """
    EXPERIMENTAL Decorated function to compute noise correlation with gCCE (without mean field)
    """
    nspin = allspin[cluster]
    if states is not None:
        states = states[cluster]

    central_spin = (dm0.shape[0] - 1) / 2

    totalh = total_hamiltonian(nspin, B, D, central_gyro=gyro_e, central_spin=central_spin)
    time_propagator = propagator(timespace, totalh)
    dmtotal0 = generate_dm0(dm0, totalh.dimensions, states=states)

    return compute_correlations(nspin, dmtotal0, time_propagator, central_spin=central_spin)


@cluster_expansion_decorator(result_operator=operator.iadd,
                             contribution_operator=operator.imul,
                             removal_operator=operator.isub,
                             addition_operator=np.sum)
def mean_field_noise_correlation(cluster, allspin, dm0, magnetic_field, D, timespace, bath_state,
                                 gyro_e=ELECTRON_GYRO, imap=None, map_error=None):
    """
    Decorated function to compute noise autocorrelation function
    with gCCE and MC sampling of the bath states
    :param subclusters: dict
        dict of subclusters included in different CCE order
        of structure {int order: np.array([[i,j],[i,j]])}
    :param allnspin: ndarray
        array of all bath
    :param ntype: dict
        dict with NSpinType objects inside, each key - name of the isotope
    :param dm0: ndarray
        initial density matrix of the central spin
    :param magnetic_field: ndarray
        Magnetic field of mfield = np.array([Bx, By, Bz])
    :param D: float
        D parameter in central spin ZFS
    :param timespace: ndarray
        Time points at which to compute
    :param bath_state: list
        List of nuclear spin states. if len(shape) == 1, contains Sz projections of nuclear spins.
        Otherwise, contains array of initial dms of nuclear spins
    :param gyro_e: float
        gyromagnetic ratio (in rad/(msec*Gauss)) of the central spin
    :return: ndarray
        autocorrelation function
    """
    nspin = allspin[cluster]
    central_spin = (dm0.shape[0] - 1) / 2

    if imap is not None:
        imap = imap.subspace(cluster)

    others_mask = np.ones(allspin.shape, dtype=bool)
    others_mask[cluster] = False
    others = allspin[others_mask]
    others_state = bath_state[others_mask]

    states = bath_state[~others_mask]

    totalh = mean_field_hamiltonian(nspin, magnetic_field, others, others_state, D,
                                    central_gyro=gyro_e,
                                    central_spin=central_spin,
                                    imap=imap,
                                    map_error=map_error)
    time_propagator = propagator(timespace, totalh)

    dmtotal0 = generate_dm0(dm0, totalh.dimensions, states)

    return compute_correlations(nspin, dmtotal0, time_propagator, central_spin=central_spin)


def noise_sampling(clusters, bath, dm0, timespace, magnetic_field, zfs,
                   gyro_e=ELECTRON_GYRO, imap=None, map_error=None,
                   nbstates=100, seed=None, parallel_states=False,
                   direct=False, parallel=False):
    """
    EXPERIMENTAL Compute noise auto correlation function
    using generalized CCE with Monte-Carlo bath state sampling
    :param timespace: 1D-ndarray
        time points at which compute density matrix
    :param magnetic_field: ndarray
        magnetic field as (Bx, By, Bz)
    :param zfs: ndarray with shape (3,3)
        ZFS tensor of central spin in rad * kHz
    :param nbstates: int
        Number of random bath states to sample
    :param seed: int
        Seed for the RNG
    :param parallel: bool
        whether to use MPI to parallelize the calculations of density matrix
        for each random bath state
    :return: ndarray
        Autocorrelation function of the noise, in (kHz*rad)^2 of shape (N, 3)
        where N is the number of time points and at each point (Ax, Ay, Az) are noise autocorrelation functions
    """

    if parallel_states:
        try:
            from mpi4py import MPI
        except ImportError:
            print('Parallel states failed: mpi4py is not found. Running serial')
            parallel_states = False

    root_divider = nbstates

    if parallel_states:
        comm = MPI.COMM_WORLD

        size = comm.Get_size()
        rank = comm.Get_rank()

        remainder = nbstates % size
        add = int(rank < remainder)
        nbstates = nbstates // size + add

        if seed:
            seed = seed + rank
    else:
        rank = 0

    averaged_corr = 0

    for bath_state in generate_bath_state(bath, nbstates, seed=seed):

        corr = mean_field_noise_correlation(clusters, bath, dm0, magnetic_field, zfs,
                                            timespace, bath_state,
                                            gyro_e=gyro_e, direct=direct, parallel=parallel,
                                            imap=imap, map_error=map_error)

        averaged_corr += corr

    if parallel_states:
        root_corr = np.array(np.zeros(averaged_corr.shape), dtype=np.complex128)
        comm.Reduce(averaged_corr, root_corr, MPI.SUM, root=0)

    else:
        root_corr = averaged_corr

    if rank == 0:
        root_corr /= root_divider
        _smc.clear()
        return root_corr
    else:
        return
