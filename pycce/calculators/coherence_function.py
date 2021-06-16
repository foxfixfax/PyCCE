import numpy as np
from pycce.bath.array import BathArray
from pycce.cluster_expansion import cluster_expansion_decorator
from pycce.constants import PI2, ELECTRON_GYRO
from pycce.hamiltonian import projected_hamiltonian, total_hamiltonian
from pycce.utilities import generate_projections

from .density_matrix import gen_density_matrix, _check_projected_states
from .monte_carlo import monte_carlo_decorator


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
        ndarray: Coherence function of the central spin.

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
def decorated_coherence_function(allspin, cluster, projections_alpha, projections_beta, magnetic_field, timespace, N,
                                 as_delay=False, states=None, projected_states=None, **kwargs):
    """
    Inner decorated function to compute coherence function in conventional CCE.

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
        ndarray: Coherence function of the central spin.
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


def compute_cce_coherence(bath, clusters, timespace, alpha, beta, magnetic_field, pulses,
                          central_spin=None, as_delay=False,
                          bath_state=None, projected_bath_state=None,
                          zfs=None, gyro_e=ELECTRON_GYRO,
                          direct=False, parallel=False, second_order=False, level_confidence=0.95,
                          **kwargs):
    """
        Function to compute coherence of the central spin using CCE.

    Args:
        bath (BathArray): Array of all bath spins.

        clusters (dict): Clusters included in different CCE orders of structure ``{int order: ndarray([[i,j],[i,j]])}``.

        timespace (ndarray): Time points at which to compute coherence function.

        alpha (ndarray with shape (2s+1,) or int):
            Vector representation of the alpha qubit state in :math:`\hat{S}_z` basis or
            index of the energy eigenstate to be considered as one.


        beta (ndarray with shape (2s+1,) or int):
            Vector representation of the beta qubit state in :math:`\hat{S}_z` basis or
            index of the energy eigenstate to be considered as one.

        magnetic_field (ndarray): Magnetic field of type ``mfield = np.array([Bx, By, Bz])``.
        pulses (int): Number of pulses in CPMG sequence.

        central_spin (float): Spin of the qubit.

        as_delay (bool):
            True if time points are delay between pulses, False if time points are total time.

        bath_state (list):
            List of bath states in any accepted format.

        projected_bath_state (ndarray): ndarray of ``shape = len(allspin)``
            containing z-projections of the bath spins states.

        zfs (ndarray with shape (3,3)): Zero Field Splitting tensor of the central spin.
        gyro_e (float or ndarray with shape (3, 3)):
            Gyromagnetic ratio of the central spin

            **OR**

            tensor corresponding to interaction between magnetic field and
            central spin.

        direct (bool):
            True if use direct approach (requires way more memory but might be more numerically stable).
            False if use memory efficient approach. Default False.

        parallel (bool):
            True if parallelize calculation of cluster contributions over different mpi processes.
            Default False.

        second_order (bool):
            True if add second order perturbation theory correction to the cluster Hamiltonian.
            If set to True sets the qubit states as eigenstates of central spin Hamiltonian from the following
            procedure. If qubit states are provided as vectors in :math:`S_z` basis,
            for each qubit state compute the fidelity of the qubit state and
            all eigenstates of the central spin and chose the one with fidelity higher than ``level_confidence``.
            If such state is not found, raises an error.

        level_confidence (float): Maximum fidelity of the qubit state to be considered eigenstate of the
            central spin Hamiltonian. Default 0.95.

        **kwargs: Additional keywords for ``projected_hamiltonian``.

    Returns:
        ndarray: Coherence function of the central spin.
    """

    alpha = np.asarray(alpha)
    beta = np.asarray(beta)

    if second_order:
        if central_spin is None:
            central_spin = (alpha.size - 1) / 2

        assert central_spin > 0, f"Incorrect spin: {central_spin}"

        hamilton = total_hamiltonian(BathArray((0,)), magnetic_field, zfs,
                                     others=bath, other_states=projected_bath_state,
                                     central_spin=central_spin,
                                     central_gyro=gyro_e)

        en, eiv = np.linalg.eigh(hamilton)

        if alpha.shape and beta.shape:
            ai = _close_state_index(alpha, eiv, level_confidence=level_confidence)
            bi = _close_state_index(beta, eiv, level_confidence=level_confidence)
        else:
            ai = alpha
            bi = beta

        alpha = eiv[:, ai]
        beta = eiv[:, bi]

        energy_alpha = en[ai]
        energy_beta = en[bi]

        energies = en

        projections_alpha_all = np.array([generate_projections(alpha, s) for s in eiv.T])
        projections_beta_all = np.array([generate_projections(beta, s) for s in eiv.T])

    else:
        if not (alpha.shape and beta.shape):
            hamilton = total_hamiltonian(BathArray((0,)), magnetic_field,
                                         zfs, others=bath,
                                         other_states=projected_bath_state,
                                         central_spin=central_spin,
                                         central_gyro=gyro_e)

            en, eiv = np.linalg.eigh(hamilton)
            alpha = eiv[:, alpha]
            beta = eiv[:, beta]

        energy_alpha = None
        energy_beta = None
        energies = None

        projections_alpha_all = None
        projections_beta_all = None

    projections_alpha = generate_projections(alpha)
    projections_beta = generate_projections(beta)

    coherence = decorated_coherence_function(bath, clusters, projections_alpha, projections_beta,
                                             magnetic_field, timespace, pulses, as_delay=as_delay,
                                             states=bath_state,
                                             projected_states=projected_bath_state,
                                             parallel=parallel, direct=direct,
                                             energy_alpha=energy_alpha, energy_beta=energy_beta,
                                             energies=energies,
                                             projections_alpha_all=projections_alpha_all,
                                             projections_beta_all=projections_beta_all,
                                             **kwargs
                                             )
    return coherence


@monte_carlo_decorator
def monte_calro_cce(bath, clusters, timespace, pulses, alpha, beta, magnetic_field,
                    central_spin,
                    zfs=None, central_gyro=ELECTRON_GYRO,
                    as_delay=False, bath_state=None,
                    direct=False, parallel=False, second_order=False, level_confidence=0.95,
                    **kwargs):
    r"""
        Compute coherence of the central spin using conventional CCE with Monte-Carlo bath state sampling.
        Note that because the function is decorated, the actual call differs from the one above by virtue of adding
        several additional keywords (see ``monte_carlo_decorator`` for details).

        Args:
            bath (BathArray):
                array of all bath spins.
            clusters (dict):
                clusters included in different CCE orders of structure ``{int order: ndarray([[i,j],[i,j]])}``.

            alpha (int or ndarray with shape (2s+1, )): :math:`\ket{0}` state of the qubit in :math:`S_z`
                basis or the index of eigenstate to be used as one.

            beta (int or ndarray with shape (2s+1, )): :math:`\ket{1}` state of the qubit in :math:`S_z` basis
                or the index of the eigenstate to be used as one.

            timespace (ndarray):
                Time points at which to compute coherence.

            pulses (int):
                number of pulses in CPMG sequence.

            magnetic_field (ndarray):
                Magnetic field of type ``mfield = np.array([Bx, By, Bz])``.

            central_spin (float): Value of the central spin.

            central_gyro (float or ndarray with shape (3,3)):
                Gyromagnetic ratio of the central spin

                **OR**

                tensor corresponding to interaction between magnetic field and
                central spin.

            zfs (ndarray with shape (3,3)): Zero Field Splitting tensor of the central spin.

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

            second_order (bool):
                True if add second order perturbation theory correction to the cluster Hamiltonian in conventional CCE.

                If set to True sets the qubit states as eigenstates of central spin Hamiltonian from the following
                procedure. If qubit states are provided as vectors in :math:`S_z` basis,
                for each qubit state compute the fidelity of the qubit state and
                all eigenstates of the central spin and chose the one with fidelity higher than ``level_confidence``.
                If such state is not found, raises an error.

            level_confidence (float): Maximum fidelity of the qubit state to be considered eigenstate of the
                central spin Hamiltonian.

            **kwargs (any):
                Additional keyword arguments for the ``projected_hamiltonian``.

        Returns:
            ndarray: coherence function of the central spin
        """
    coherence = compute_cce_coherence(bath, clusters, timespace, alpha, beta, magnetic_field, pulses,
                                      central_spin, as_delay=as_delay,
                                      bath_state=bath_state, projected_bath_state=bath_state,
                                      zfs=zfs, gyro_e=central_gyro,
                                      direct=direct, parallel=parallel,
                                      second_order=second_order, level_confidence=level_confidence, **kwargs)

    return coherence


def _close_state_index(state, eiv, level_confidence=0.95):
    """
    Get index of the eigenstate stored in eiv,
    which has fidelity higher than ``level_confidence`` with the provided ``state``.

    Args:
        state (ndarray with shape (2s+1,)): State for which to find the analogous eigen state.
        eiv (ndarray with shape (2s+1, 2s+1)): Matrix of eigenvectors as columns.
        level_confidence (float): Threshold fidelity. Default 0.95.

    Returns:
        int: Index of the eigenstate.
    """
    indexes = np.argwhere((eiv.T @ state) ** 2 > level_confidence).flatten()

    if not indexes.size:
        raise ValueError(f"Initial qubit state is below F = {level_confidence} "
                         f"to the eigenstate of central spin Hamiltonian.\n"
                         f"Qubit level:\n{repr(state)}"
                         f"Eigenstates (rows):\n{repr(eiv.T)}")
    return indexes[0]
