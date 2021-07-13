import numpy as np
from pycce.constants import PI2
from pycce.h import projected_hamiltonian
from pycce.utilities import generate_projections

from .base import RunObject
from .gcce import gen_density_matrix


def _rotmul(rotation, u, **kwargs):
    if rotation is not None:
        u = np.matmul(rotation, u, **kwargs)
    return u


def propagators(timespace, H0, H1, pulses, as_delay=False):
    """
    Function to compute propagators U0 and U1 in conventional CCE.

    Args:
        timespace (ndarray with shape (t, )): Time delay values at which to compute propagators.

        H0 (ndarray with shape (n, n)): Hamiltonian projected on alpha qubit state.

        H1 (ndarray with shape (n, n)): Hamiltonian projected on beta qubit state.

        pulses (int or Sequence): Sequence of pulses.

        as_delay (bool):
            True if time points are delay between pulses.
            False if time points are total time.

    Returns:
        tuple: *tuple* containing:

            * **ndarray with shape (t, n, n)**:
              Matrix representation of the propagator conditioned on the alpha qubit state for each time point.
            * **ndarray with shape (t, n, n)**:
              Matrix representation of the propagator conditioned on the beta qubit state for each time point.

    """
    try:
        number = len(pulses)
        check = (not as_delay) & (pulses.delays is None)
        use_rotations = True
    except TypeError:
        number = pulses
        check = (not as_delay)
        use_rotations = False

    if check and number:
        timespace = timespace / (2 * number)

    eval0, evec0 = np.linalg.eigh(H0 * PI2)
    eval1, evec1 = np.linalg.eigh(H1 * PI2)

    if not use_rotations:
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

        if not number:
            return v0, v1

        V0_HE = np.matmul(v0, v1, dtype=np.complex128)
        V1_HE = np.matmul(v1, v0, dtype=np.complex128)

        if number == 1:
            return V0_HE, V1_HE

        V0 = np.matmul(V0_HE, V1_HE, dtype=np.complex128)  # v0 @ v1 @ v1 @ v0
        V1 = np.matmul(V1_HE, V0_HE, dtype=np.complex128)  # v1 @ v0 @ v0 @ v1

        U0 = np.linalg.matrix_power(V0, number // 2)
        U1 = np.linalg.matrix_power(V1, number // 2)

        if number % 2 == 1:
            U0 = np.matmul(U0, V0_HE)
            U1 = np.matmul(U1, V1_HE)

        return U0, U1

    if pulses.delays is None:

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

        vs = [v0, v1]

        U0 = np.eye(v0.shape[1], dtype=np.complex128)
        U1 = np.eye(v1.shape[1], dtype=np.complex128)
        Us = [U0, U1]

        for p in pulses:
            for i in range(2):
                Us[i] = np.matmul(vs[i], Us[i])
                Us[i] = _rotmul(p.rotation, Us[i])

            if p.flip:
                vs = vs[::-1]

            for i in range(2):
                Us[i] = np.matmul(vs[i], Us[i])

        return Us[0], Us[1]

    U0 = None
    U1 = None

    times = 0

    # for timesteps, rotation in zip(pulses.delays, pulses.rotations):
    evalues = [eval0, eval1]
    evec = [evec0, evec1]

    for p in pulses:
        timesteps = p.delay
        rotation = p.rotation

        eigen_exp0 = np.exp(-1j * np.tensordot(timesteps,
                                               evalues[0], axes=0), dtype=np.complex128)
        eigen_exp1 = np.exp(-1j * np.tensordot(timesteps,
                                               evalues[1], axes=0), dtype=np.complex128)

        u0 = np.matmul(np.einsum('...ij,...j->...ij', evec[0], eigen_exp0, dtype=np.complex128),
                       evec[0].conj().T)

        u1 = np.matmul(np.einsum('...ij,...j->...ij', evec[1], eigen_exp1, dtype=np.complex128),
                       evec[1].conj().T)

        times += timesteps

        if U0 is None:
            U0 = _rotmul(rotation, u0)
            U1 = _rotmul(rotation, u1)

        else:
            U0 = np.matmul(u0, U0)
            U0 = _rotmul(rotation, U0)

            U1 = np.matmul(u1, U1)
            U1 = _rotmul(rotation, U1)

        if p.flip:
            evalues = evalues[::-1]
            evec = evec[::-1]

    if ((timespace - times) >= 0).all() and (timespace - times).any():

        eigen_exp0 = np.exp(-1j * np.tensordot(timespace - times,
                                               evalues[0], axes=0), dtype=np.complex128)
        eigen_exp1 = np.exp(-1j * np.tensordot(timespace - times,
                                               evalues[1], axes=0), dtype=np.complex128)

        u0 = np.matmul(np.einsum('ij,kj->kij', evec[0], eigen_exp0, dtype=np.complex128),
                       evec[0].conj().T)

        u1 = np.matmul(np.einsum('ij,kj->kij', evec[1], eigen_exp1, dtype=np.complex128),
                       evec[1].conj().T)

        U0 = np.matmul(u0, U0)
        U1 = np.matmul(u1, U1)

    elif ((timespace - times) < 0).any():
        raise ValueError(f"Pulse sequence time steps add up to larger than total times"
                         f"{np.argwhere((timespace - times) < 0)} are longer than total time.")

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


class CCE(RunObject):
    """
    Class for running conventional CCE simulations.

    .. note::

        Subclass of the ``RunObject`` abstract class.

    Args:
        *args: Positional arguments of the ``RunObject``.

        pulses (int or Sequence):
            number of pulses in CPMG sequence or instance of Sequence object.
            For now, only CPMG sequences are supported in conventional CCE simulations.

        as_delay (bool):
            True if time points are delay between pulses, False if time points are total time.

        second_order (bool):
            True if add second order perturbation theory correction to the cluster Hamiltonian.
            If set to True sets the qubit states as eigenstates of central spin Hamiltonian from the following
            procedure. If qubit states are provided as vectors in :math:`S_z` basis,
            for each qubit state compute the fidelity of the qubit state and
            all eigenstates of the central spin and chose the one with fidelity higher than ``level_confidence``.
            If such state is not found, raises an error.

        level_confidence (float): Maximum fidelity of the qubit state to be considered eigenstate of the
            central spin Hamiltonian. Default 0.95.

        **kwargs: Keyword arguments of the ``RunObject``.

    """

    def __init__(self, *args, pulses=0, as_delay=False, second_order=False,
                 level_confidence=0.95, **kwargs):

        self.initial_pulses = pulses
        """int or Sequence: Input pulses"""
        self.pulses = None
        """int or Sequence: If input Sequence contains only pi pulses at even delay, stores number of pulses.
        Otherwise stores full ``Sequence``."""
        self.as_delay = as_delay
        """bool: True if time points are delay between pulses, False if time points are total time."""
        self.second_order = second_order
        """bool: True if add second order perturbation theory correction to the cluster hamiltonian."""
        self.level_confidence = level_confidence
        """float: Maximum fidelity of the qubit state to be considered eigenstate of the central spin hamiltonian."""
        self.energy_alpha = None
        """float: Eigen energy of the alpha state in the central spin Hamiltonian."""
        self.energy_beta = None
        """float: Eigen energy of the beta state in the central spin Hamiltonian."""
        self.energies = None
        """ndarray with shape (2s+1, ): All eigen energies of the central spin Hamiltonian."""
        self.projections_alpha_all = None
        r"""ndarray with shape (2s+1, 3): Array of vectors with spin operator matrix elements
        of type :math:`[\bra{0}\hat S_x\ket{i}, \bra{0}\hat S_y\ket{i}, \bra{0}\hat S_z\ket{i}]`, where
        :math:`\ket{0}` is the alpha qubit state,
        :math:`\ket{i}` are all eigenstates of the central spin hamiltonian."""
        self.projections_beta_all = None
        r"""ndarray with shape (2s+1, 3): Array of vectors with spin operator matrix elements
        of type :math:`[\bra{1}\hat S_x\ket{i}, \bra{1}\hat S_y\ket{i}, \bra{1}\hat S_z\ket{i}]`, where
        :math:`\ket{1}` is the beta qubit state,
        :math:`\ket{i}` are all eigenstates of the central spin hamiltonian."""
        self.projections_alpha = None
        r"""ndarray with shape (3,): Vector with spin operator matrix elements
        of type :math:`[\bra{0}\hat S_x\ket{0}, \bra{0}\hat S_y\ket{0}, \bra{0}\hat S_z\ket{0}]`, where
        :math:`\ket{0}` is the alpha qubit state"""
        self.projections_beta = None
        r"""ndarray with shape (3,): Vectors with spin operator matrix elements
        of type :math:`[\bra{1}\hat S_x\ket{1}, \bra{1}\hat S_y\ket{1}, \bra{1}\hat S_z\ket{1}]`, where
        :math:`\ket{1}` is the beta qubit state."""

        self.use_pulses = False
        """bool: True if use full ``Sequence``. False if use only number of pulses."""
        super().__init__(*args, **kwargs)

    def preprocess(self):
        super().preprocess()

        try:
            pulses = self.initial_pulses
            number = len(self.initial_pulses)

            angles = [p.angle for p in pulses]
            c1 = not (None in angles)

            if not c1:
                pure_angles = [value for value in angles if value]
            else:
                pure_angles = angles

            if not all(np.isclose(pure_angles, np.pi)):
                raise ValueError('Only pi-pulses are supported for CCE. Use gCCE for user-defined sequences.')

            c2 = all([p.bath_names is None for p in pulses])
            c3 = all([not p._has_delay for p in pulses])

            if (c1 & c2 & c3) or not number:
                self.pulses = number
            else:
                self.pulses = self.initial_pulses
                self.use_pulses = True

        except TypeError:
            self.pulses = self.initial_pulses

        if self.second_order:
            ai = _close_state_index(self.alpha, self.eigenvectors, level_confidence=self.level_confidence)
            bi = _close_state_index(self.beta, self.eigenvectors, level_confidence=self.level_confidence)

            alpha = self.eigenvectors[:, ai]
            beta = self.eigenvectors[:, bi]

            self.energy_alpha = self.energies[ai]
            self.energy_beta = self.energies[bi]

            self.energies = self.energies

            self.projections_alpha_all = np.array([generate_projections(alpha, s) for s in self.eigenvectors.T])
            self.projections_beta_all = np.array([generate_projections(beta, s) for s in self.eigenvectors.T])

        else:

            self.energy_alpha = None
            self.energy_beta = None
            self.energies = None

            self.projections_alpha_all = None
            self.projections_beta_all = None

        self.projections_alpha = generate_projections(self.alpha)
        self.projections_beta = generate_projections(self.beta)

    def postprocess(self):
        pass

    def generate_hamiltonian(self):
        """
        Using the attributes of the ``self`` object,
        compute the two projected cluster hamiltonians.

        Returns:

            tuple: Tuple containing:

                * **Hamiltonian**:
                  Cluster hamiltonian when qubit in the alpha state.
                * **Hamiltonian**:
                  Cluster hamiltonian when qubit in the alpha state.

        """
        hamil = projected_hamiltonian(self.cluster, self.projections_alpha, self.projections_beta, self.magnetic_field,
                                      others=self.others,
                                      other_states=self.other_states,
                                      projections_beta_all=self.projections_beta_all,
                                      projections_alpha_all=self.projections_alpha_all,
                                      energy_alpha=self.energy_alpha, energy_beta=self.energy_beta)

        if self.use_pulses:
            self.pulses.generate_pulses(dimensions=hamil[0].dimensions, bath=self.cluster,
                                        vectors=hamil[0].vectors, central_spin=False)

        return hamil

    def compute_result(self):
        """
        Using the attributes of the ``self`` object,
        compute the coherence function as overlap in the bath evolution.

        Returns:

            ndarray: Computed coherence.

        """
        return compute_coherence(self.cluster_hamiltonian[0], self.cluster_hamiltonian[1],
                                 self.timespace, self.pulses, as_delay=self.as_delay,
                                 states=self.states)

    # def kernel(self, cluster, **kwargs):
    #     """
    #     Inner kernel function to compute coherence function in conventional CCE.
    #
    #     Args:
    #         cluster (ndarray): Indexes of the bath spins in the given cluster.
    #
    #     Returns:
    #         ndarray: Coherence function of the central spin.
    #     """
    #     nspin = self.bath[cluster]
    #     states, others, other_states = _check_projected_states(cluster, self.bath, self.bath_state,
    #                                                            self.projected_bath_state)
    #     # print(other_states)
    #     H0, H1 = projected_hamiltonian(nspin, self.projections_alpha, self.projections_beta, self.magnetic_field,
    #                                    others=others,
    #                                    other_states=other_states,
    #                                    projections_beta_all=self.projections_beta_all,
    #                                    projections_alpha_all=self.projections_alpha_all,
    #                                    energy_alpha=self.energy_alpha, energy_beta=self.energy_beta,
    #                                    **kwargs)
    #
    #     coherence = compute_coherence(H0, H1, self.timespace, self.pulses, as_delay=self.as_delay, states=states)
    #
    #     return coherence

    # def interlaced_kernel(self, cluster, supercluster, *args, **kwargs):
    #     """
    #     Inner kernel function to compute coherence function in conventional CCE with interlaced averaging.
    #
    #     Args:
    #         cluster (ndarray): Indexes of the bath spins in the given cluster.
    #         cluster (ndarray): Indexes of the bath spins in the supercluster of the given cluster.
    #
    #     Returns:
    #         ndarray: Coherence function of the central spin.
    #     """
    #     nspin = self.bath[cluster]
    #
    #     _, others, other_states = _check_projected_states(supercluster, self.bath, self.bath_state,
    #                                                       self.projected_bath_state)
    #
    #     H0, H1 = projected_hamiltonian(nspin, self.projections_alpha, self.projections_beta, self.magnetic_field,
    #                                    others=others,
    #                                    other_states=other_states,
    #                                    projections_beta_all=self.projections_beta_all,
    #                                    projections_alpha_all=self.projections_alpha_all,
    #                                    energy_alpha=self.energy_alpha, energy_beta=self.energy_beta,
    #                                    **kwargs)
    #
    #     sc_mask = ~np.isin(supercluster, cluster)
    #
    #     outer_indexes = supercluster[sc_mask]
    #     outer_spin = self.bath[outer_indexes]
    #
    #     initial_h0 = H0.data
    #     initial_h1 = H1.data
    #
    #     coherence = 0
    #     i = 0
    #
    #     for i, state in enumerate(generate_supercluser_states(self, supercluster)):
    #
    #         cluster_states = state[~sc_mask]
    #         outer_states = state[sc_mask]
    #
    #         if outer_spin.size > 0:
    #             addition = 0
    #
    #             for ivec, n in zip(H0.vectors, nspin):
    #                 addition += overhauser_bath(ivec, n['xyz'], n.gyro, outer_spin.gyro,
    #                                             outer_spin['xyz'], outer_states)
    #
    #             H0.data = initial_h0 + addition
    #             H1.data = initial_h1 + addition
    #
    #         coherence += compute_coherence(H0, H1, self.timespace, self.pulses, as_delay=self.as_delay,
    #                                        states=cluster_states)
    #
    #     coherence /= i + 1
    #
    #     return coherence
