import numpy as np
import numpy.ma as ma
from pycce.cluster_expansion import cluster_expansion_decorator
from pycce.hamiltonian import projected_hamiltonian
from pycce.constants import PI2
from .density_matrix import gen_density_matrix, generate_bath_state, _check_projected_states


def propagators(timespace, H0, H1, N, as_delay=False):
    """
    Function to compute propagators U0 and U1 in conventional CCE.

    Args:
        timespace (ndarray with shape (t, )): Time delay values at which to compute propagators.

        H0 (ndarray with shape (2s+1, 2s+1)): Hamiltonian projected on alpha qubit state.

        H1 (ndarray with shape (2s+1, 2s+1)): Hamiltonian projected on beta qubit state.

        N (int): number of pulses in CPMG.

        as_delay (bool):
            True if time points are delay between pulses.
            False if time points are total time.

    Returns:
        tuple: *tuple* containing:

            * **ndarray with shape (t, 2s+1, 2s+1)**:
              Matrix representation of the propagator conditioned on the alpha qubit state for each time point.
            * **ndarray with shape (t, 2s+1, 2s+1)**:
              Matrix representation of the propagator conditioned on the beta qubit state for each time point.

    """
    if not as_delay and N:
        timespace = timespace / (2 * N)

    eval0, evec0 = np.linalg.eigh(H0 * PI2)
    eval1, evec1 = np.linalg.eigh(H1 * PI2)

    eigen_exp0 = np.exp(-1j * np.tensordot(timespace,
                                           eval0, axes=0), dtype=np.complex128)
    eigen_exp1 = np.exp(-1j * np.tensordot(timespace,
                                           eval1, axes=0), dtype=np.complex128)

    v0 = np.matmul(np.einsum('ij,kj->kij', evec0, eigen_exp0,
                             dtype=np.complex128),
                   evec0.conj().T, dtype=np.complex128)

    v1 = np.matmul(np.einsum('ij,kj->kij', evec1, eigen_exp1,
                             dtype=np.complex128),
                   evec1.conj().T, dtype=np.complex128)

    if not N:
        return v0, v1

    V0_HE = np.matmul(v0, v1, dtype=np.complex128)
    V1_HE = np.matmul(v1, v0, dtype=np.complex128)

    if N == 1:
        return V0_HE, V1_HE

    V0 = np.matmul(V0_HE, V1_HE, dtype=np.complex128)  # v0 @ v1 @ v1 @ v0
    V1 = np.matmul(V1_HE, V0_HE, dtype=np.complex128)  # v1 @ v0 @ v0 @ v1

    U0 = np.linalg.matrix_power(V0, N // 2)
    U1 = np.linalg.matrix_power(V1, N // 2)

    if N % 2 == 1:
        U0 = np.linalg.matmul(U0, V0_HE)
        U1 = np.linalg.matmul(U1, V1_HE)

    return U0, U1


def compute_coherence(H0, H1, timespace, N, as_delay=False, states=None):
    """
    Function to compute cluster coherence function in conventional CCE.

    Args:
        H0 (ndarray): Hamiltonian projected on alpha qubit state.
        H1 (ndarray): Hamiltonian projected on beta qubit state.
        timespace (ndarray): Time points at which to compute coherence function.
        N (int): Number of pulses in CPMG.
        as_delay (bool):
            True if time points are delay between pulses,
            False if time points are total time.
        states (ndarray): ndarray of bath states in any accepted format.

    Returns:
        coherence_function (ndarray):
            Coherence function of the central spin.

    """
    # if timespace was given not as delay between pulses,
    # divide to obtain the delay
    U0, U1 = propagators(timespace, H0.data, H1.data, N, as_delay=as_delay)

    # coherence_function = np.trace(np.matmul(U0, np.transpose(
    #     U1.conj(), axes=(0, 2, 1))), axis1=1, axis2=2) / U0.shape[1]
    # coherence_function is computed as Tr[rho U0 U1dagger]; rho = Identity / dim
    if states is None:
        coherence_function = np.einsum('zij,zij->z', U0, U1.conj()) / U0.shape[1]

    else:
        dm = gen_density_matrix(states, dimensions=H0.dimensions)
        # tripple einsum is slow
        # coherence_function = np.einsum('zli,ij,zlj->z', U0, dm, U1.conj())
        dmUdagger = np.matmul(dm, np.transpose(U1.conj(), axes=(0, 2, 1)))
        coherence_function = np.trace(np.matmul(U0, dmUdagger), axis1=1, axis2=2)
    return coherence_function


@cluster_expansion_decorator
def decorated_coherence_function(cluster, allspin, projections_alpha, projections_beta, magnetic_field, timespace, N,
                                 as_delay=False, states=None, projected_states=None, **kwargs):
    """
    Overarching decorated function to compute coherence function in conventional CCE.

    Args:
        cluster (dict):
            clusters included in different CCE orders of structure ``{int order: ndarray([[i,j],[i,j]])}``.
        allspin (BathArray):
            array of all bath spins
        projections_alpha (ndarray):
            ndarray containing projections of alpha state
            :math:`[\braket{\hat{S}_x}, \braket{\hat{S}_y}, \braket{\hat{S}_z}]`.
        projections_beta (ndarray):
            ndarray containing projections of beta state
            :math:`[\braket{\hat{S}_x}, \braket{\hat{S}_y}, \braket{\hat{S}_z}]`.
        magnetic_field (ndarray):
            Magnetic field of type ``mfield = np.array([Bx, By, Bz])``.
        timespace (ndarray):
            Time points at which to compute coherence.
        N (int):
            number of pulses in CPMG sequence.
        as_delay (bool):
            True if time points are delay between pulses, False if time points are total time.
        states (list):
            list of bath states in any accepted format.
        projected_states (ndarray): ndarray of ``shape = len(allspin)``
            containing z-projections of the bath spins states.

        **kwargs (any): Additional arguments for projected_hamiltonian.

    Returns:
        coherence_function (ndarray):
            Coherence function of the central spin.
    """
    nspin = allspin[cluster]

    states, others, other_states = _check_projected_states(cluster, allspin, states, projected_states)
    # if imap is not None:
    #     imap = imap.subspace(cluster)

    H0, H1 = projected_hamiltonian(nspin, projections_alpha, projections_beta, magnetic_field,
                                   others=others,
                                   other_states=other_states, **kwargs)

    coherence = compute_coherence(H0, H1, timespace, N, as_delay=as_delay, states=states)
    return coherence


def monte_carlo_coherence(cluster, allspin, projections_alpha, projections_beta, magnetic_field, timespace, N,
                          as_delay=False,
                          nbstates=100, seed=None, masked=True,
                          parallel_states=False,
                          fixstates=None, direct=False, parallel=False,
                          **kwargs):
    r"""
    Compute coherence of the central spin using conventional CCE with Monte-Carlo bath state sampling.

    Args:
        cluster (dict):
            clusters included in different CCE orders of structure ``{int order: ndarray([[i,j],[i,j]])}``.
        allspin (BathArray):
            array of all bath spins.
        projections_alpha (ndarray):
            ndarray containing projections of alpha state
            :math:`[\braket{\hat{S}_x}, \braket{\hat{S}_y}, \braket{\hat{S}_z}]`.
        projections_beta (ndarray):
            ndarray containing projections of beta state
            :math:`[\braket{\hat{S}_x}, \braket{\hat{S}_y}, \braket{\hat{S}_z}]`.
        magnetic_field (ndarray):
            Magnetic field of type ``mfield = np.array([Bx, By, Bz])``.
        timespace (ndarray):
            Time points at which to compute coherence.
        N (int):
            number of pulses in CPMG sequence.
        as_delay (bool):
            True if time points are delay between pulses, False if time points are total time.
        nbstates (int):
            Number of random bath states to sample.
        seed (int):
            Seed for the RNG.
        masked (bool):
            True if mask numerically unstable points (with coherence > 1) in the averaging over bath states
            False if not. Default True.
        parallel_states (bool):
            True if use MPI to parallelize the calculations of density matrix
            for each random bath state.
        fixstates (dict):
            dict of which bath states to fix. Each key is the index of bath spin,
            value - fixed :math:`\hat{I}_z` projection of the mixed state of bath spin.
        direct (bool):
            True if use the direct approach in cluster expansion
        parallel (bool):
            True if use MPI for parallel computing of the cluster contributions.
        **kwargs (any):
            Additional keyword arguments for the projected_hamiltonian.

    Returns:
        coherence_function (ndarray):
            coherence function of the central spin
    """
    if parallel_states:
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD

            size = comm.Get_size()
            rank = comm.Get_rank()

            remainder = nbstates % size
            add = int(rank < remainder)
            nbstates = nbstates // size + add

            if seed:
                seed = seed + rank

        except ImportError:
            print('Parallel failed: mpi4py is not found. Running serial')
            parallel_states = False
            rank = 0

    else:
        rank = 0

    if masked:
        divider = np.zeros(timespace.shape, dtype=np.int32)
    else:
        divider = nbstates

    average = np.zeros(timespace.size, dtype=np.complex128)

    for bath_state in generate_bath_state(allspin, nbstates, seed=seed, fixstates=fixstates, parallel=parallel):
        coherence = decorated_coherence_function(cluster, allspin, projections_alpha, projections_beta,
                                                 magnetic_field, timespace, N, as_delay=as_delay, states=bath_state,
                                                 projected_states=bath_state,
                                                 parallel=parallel, direct=direct, **kwargs)
        if masked:
            proper = np.abs(coherence) <= np.abs(coherence[0])
            divider += proper.astype(np.int32)
            coherence[~proper] = 0.

        average += coherence

    if parallel_states:
        root_result = np.array(np.zeros(average.shape), dtype=np.complex128)
        comm.Reduce(average, root_result, MPI.SUM, root=0)
        if masked:
            root_divider = np.zeros(divider.shape, dtype=np.int32)
            comm.Reduce(divider, root_divider, MPI.SUM, root=0)
        else:
            root_result = divider
    else:
        root_result = average
        root_divider = divider

    if rank == 0:
        root_result = ma.array(root_result, fill_value=0j, dtype=np.complex128)

        if masked:
            root_result[root_divider == 0] = ma.masked
        root_result /= root_divider

        return root_result
    else:
        return
