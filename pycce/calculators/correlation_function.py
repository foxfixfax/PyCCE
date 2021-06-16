import numpy as np
import operator

from pycce.cluster_expansion import cluster_expansion_decorator
from pycce.hamiltonian import total_hamiltonian, bath_interactions, expanded_single, conditional_hyperfine, \
    dimensions_spinvectors
from pycce.constants import ELECTRON_GYRO
from .density_matrix import propagator, generate_dm0, gen_density_matrix, _check_projected_states

from .monte_carlo import monte_carlo_decorator


def correlation_it_j0(operator_i, operator_j, dm0_expanded, U):
    """
    Function to compute correlation function of the operator i at time t and operator j at time 0

    Args:
        operator_i (ndarray with shape (n, n)):
            Matrix representation of operator i.
        operator_j (ndarray with shape (n, n)):
            Matrix representation of operator j.
        dm0_expanded (ndarray with shape (n, n)):
            Initial density matrix of the cluster.
        U (ndarray with shape (t, n, n)):
            Time evolution propagator, evaluated over t time points.

    Returns:
        ndarray with shape (t,): Autocorrelation of the z-noise at each time point.

    """

    operator_i_t = np.matmul(np.transpose(U.conj(), axes=(0, 2, 1)), np.matmul(operator_i, U))
    # operator_j_t = np.matmul(np.transpose(U.conj(), axes=(0, 2, 1)), np.matmul(operator_j, U))
    it_j0 = np.matmul(operator_i_t, operator_j)  # + np.matmul(operator_j, operator_i_t)) / 2
    matmul = np.matmul(dm0_expanded, it_j0)
    corr = matmul.trace(axis1=1, axis2=2, dtype=np.complex128)

    return corr


def compute_correlations(nspin, dm0_expanded, U, central_spin=None):
    """
    Function to compute correlations for the given cluster, given time propagator U.

    Args:
        nspin (BathArray):
            BathArray of the given cluster of bath spins.
        dm0_expanded (ndarray with shape (n, n)):
            Initial density matrix of the cluster.
        U (ndarray with shape (t, n, n)):
            Time evolution propagator, evaluated over t time points.
        central_spin (float):
            Value of the central spin.

    Returns:
        ndarray with shape (t,):
            correlation of the Overhauser field, induced by the given cluster at each time point.

    """
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
def projected_noise_correlation(allspin, cluster, projections_state, magnetic_field, timespace, states=None,
                                ):
    """
    Decorated function to compute autocorrelation function with conventional CCE.

    Args:
        cluster (dict):
            Clusters included in different CCE orders of structure {int order: ndarray([[i,j],[i,j]])}.
        allspin (BathArray):
            Array of all bath spins.
        projections_state (ndarray with shape (3,)):
            Array containing projections of state state [<Sx>, <Sy>, <Sz>].
        magnetic_field (ndarray with shape (3,)):
            Magnetic field of type mfield = np.array([Bx, By, Bz]).
        timespace (ndarray with shape (t,)):
            Time points at which to compute autocorrelation.
        states (ndarray):
            Array of bath states in any accepted format.

    Returns:
        ndarray with shape (t,): Autocorrelation of the bath spin noise along z-axis.

    """
    bath = allspin[cluster]
    if states is not None:
        states = states[cluster]

    ntype = bath.types

    dimensions, ivectors = dimensions_spinvectors(bath, central_spin=None)

    totalh = 0

    for ivec, n in zip(ivectors, bath):
        hsingle = expanded_single(ivec, ntype[n].gyro, magnetic_field, n['Q'])

        hf_state = conditional_hyperfine(n['A'], ivec, projections_state)

        totalh += hsingle + hf_state

    totalh += bath_interactions(bath, ivectors)
    time_propagator = propagator(timespace, totalh)

    dm0_expanded = gen_density_matrix(states, dimensions=dimensions)

    return compute_correlations(bath, dm0_expanded, time_propagator)


@cluster_expansion_decorator(result_operator=operator.iadd,
                             contribution_operator=operator.imul,
                             removal_operator=operator.isub,
                             addition_operator=np.sum)
def decorated_noise_correlation(allspin, cluster, dm0, magnetic_field, zfs, timespace, bath_state=None,
                                projected_bath_state=None,
                                gyro_e=ELECTRON_GYRO):
    """
    Decorated function to compute noise autocorrelation function with gCCE.

    Args:
        cluster (dict):
            Clusters included in different CCE orders of structure ``{int order: ndarray([[i,j],[i,j]])}``.
        allspin (BathArray):
            Array of all bath spins.
        dm0 (ndarray with shape (2s+1, 2s+1)):
            Density matrix of the initial state of the central spin.
        magnetic_field (ndarray with shape (3,)):
            Magnetic field of type ``mfield = np.array([Bx, By, Bz])``.
        zfs (ndarray with shape (3,3)):
            Zero Field Splitting tensor of the central spin.
        timespace (ndarray with shape (t,)):
            Time points at which to compute autocorrelation.
        bath_state (ndarray):
            Array of bath states in any accepted format.
        projected_bath_state (ndarray):
            Array of bath states in z-projections format.
        gyro_e (float or ndarray with shape (3,3)):
            Gyromagnetic ratio of the central spin.

            **OR**

            Tensor corresponding to interaction between magnetic field and central spin.

    Returns:
        ndarray with shape (t,): Autocorrelation of the bath spin noise along z-axis.

    """
    nspin = allspin[cluster]
    central_spin = (dm0.shape[0] - 1) / 2

    states, others, other_states = _check_projected_states(cluster, allspin, bath_state, projected_bath_state)

    totalh = total_hamiltonian(nspin, magnetic_field, zfs, others=others, other_states=other_states,
                               central_gyro=gyro_e, central_spin=central_spin)

    time_propagator = propagator(timespace, totalh.data)

    dmtotal0 = generate_dm0(dm0, totalh.dimensions, states)

    return compute_correlations(nspin, dmtotal0, time_propagator, central_spin=central_spin)


@monte_carlo_decorator
def monte_carlo_noise(bath, clusters, dm0, timespace, magnetic_field, zfs,
                      gyro_e=ELECTRON_GYRO, bath_state=None, parallel=False, direct=False):
    """
    Compute noise auto correlation function using generalized CCE with Monte-Carlo bath state sampling.
    Note that because the function is decorated, the actual call differs from the one above by virtue of adding
    several additional keywords (see ``monte_carlo_decorator`` for details).

    Args:
        bath (BathArray):
            Array of all bath spins.
        clusters (dict):
            Clusters included in different CCE orders of structure {int order: ndarray([[i,j],[i,j]])}.
        dm0 (ndarray with shape (2s+1, 2s+1)):
            Density matrix of the initial state of the central spin.
        timespace (ndarray with shape (t,)):
            Time points at which to compute autocorrelation.
        magnetic_field (ndarray with shape (3, )):
            Magnetic field of type mfield = np.array([Bx, By, Bz]).
        zfs (ndarray with shape (3, 3)):
            Zero Field Splitting tensor of the central spin.
        gyro_e (float or ndarray with shape (3, 3)):
            gyromagnetic ratio of the central spin OR
            tensor corresponding to interaction between magnetic field and central spin.
        nbstates (int):
            Number of random bath states to sample.
        seed (int):
            Seed for the RNG.
        parallel_states (bool):
            True if use MPI to parallelize the calculations of density matrix
            for each random bath state.
        direct (bool):
            True if use the direct approach in cluster expansion.
        parallel (bool):
            True if use MPI for parallel computing of the cluster contributions.

    Returns:
        ndarray with shape (t,): Autocorrelation of the bath spin noise along z-axis.

    """
    corr = decorated_noise_correlation(bath, clusters, dm0, magnetic_field, zfs,
                                       timespace, bath_state, projected_bath_state=bath_state,
                                       gyro_e=gyro_e, direct=direct, parallel=parallel)
    return corr
