import numpy as np
from numpy import ma as ma
from pycce.bath.array import BathArray
from pycce.constants import PI2
from pycce.h import total_hamiltonian
from pycce.run.base import RunObject, generate_initial_state, simple_propagator
from pycce.utilities import shorten_dimensions, outer


def rotation_propagator(u, rotations):
    """
    Generate the propagator from the simple propagator and set of :math:`2\tau` equispaced rotation operators.

    .. note::

        While the spacing between rotation operators is assumed to be :math:`2\tau`, the spacing before and after
        the first and the last rotation respectively is assumed to be :math:`\tau`.

    Args:
        u (ndarray with shape (n, N, N)): Simple propagator.
        rotations (ndarray with shape (x, N, N)): Array of rotation operators.

    Returns:
        ndarray with shape (n, N, N): Full propagator.

    """
    full_u = np.eye(u.shape[1], dtype=np.complex128)

    for rotation in rotations:
        full_u = np.matmul(u, full_u)

        if rotation is not None:
            full_u = np.matmul(rotation, full_u)

        full_u = np.matmul(u, full_u)

    return full_u


def _get_state(state, center):
    if callable(state):
        return state

    state = np.asarray(state)
    if state.size == 1:
        return center.eigenvectors[:, int(state)]
    else:
        return state.astype(np.complex128)


class gCCE(RunObject):
    """
    Class for running generalized CCE simulations.

    .. note::

        Subclass of the ``RunObject`` abstract class.

    Args:
        *args: Positional arguments of the ``RunObject``.

        pulses (Sequence): Sequence object, containing series of pulses, applied to the system.

        fulldm (bool):
            True if return full density matrix. Default False.

        **kwargs: Keyword arguments of the ``RunObject``.

    """

    def __init__(self, *args, i=None, j=None, fulldm=False, normalized=True, **kwargs):

        self.dm0 = None
        """ ndarray with shape (2s+1, 2s+1): Initial density matrix of the central spin."""
        self.normalization = None
        """ float: Coherence at time 0."""
        self.zero_cluster = None
        """ ndarray with shape (n,): Coherence computed for the isolated central spin."""
        self.i = i
        self.j = j
        self.alpha = None
        self.beta = None
        self.normalized = normalized
        self.fulldm = fulldm
        """ bool: True if return full density matrix."""

        super().__init__(*args, **kwargs)

    def preprocess(self):
        super().preprocess()

        # Emulate kernel
        self.base_hamiltonian = self.center.generate_hamiltonian(magnetic_field=self.magnetic_field)
        self.cluster = BathArray((0,))
        self.cluster_indexes = np.array([], dtype=int)

        # if self.has_states:
        #     self.others = self.bath
        #     self.others_mask = np.ones(self.bath.size, dtype=bool)

        self._check_hamiltonian()

        self.dm0 = self.center.state
        self.normalization = outer(self.center.state, self.center.state)

        if self.i is None:
            alpha = self.center.alpha
        else:
            alpha = _get_state(self.i, self.center)

        if self.j is None:
            beta = self.center.beta
        else:
            beta = _get_state(self.j, self.center)

        if self.pulses is not None:
            self.center.generate_sigma()
            self.generate_pulses()

            for rotation in self.rotations:
                if rotation is not None:
                    if self.i is None:
                        alpha = rotation @ alpha
                    if self.j is None:
                        beta = rotation @ beta

                    self.normalization = rotation @ self.normalization @ rotation.T.conj()

        self.alpha = alpha
        self.beta = beta
        self.zero_cluster = 1  # For compute result to work

        self.zero_cluster = self.compute_result()

        if self.fulldm:
            self.zero_cluster = ma.array(self.zero_cluster, mask=np.isclose(self.zero_cluster, 0), fill_value=0j)
        elif self.normalized:
            self.normalization = self.process_dm(self.normalization)

        # else:
        #     density_matrix = self.center.eigenvectors.conj().T @ density_matrix @ self.center.eigenvectors

    def process_dm(self, density_matrix):
        """
        Obtain the result from the density matrices.

        Args:
            density_matrix (ndarray with shape (n, N, N)): Array of the density matrices.

        Returns:
            ndarray:
                Depending on the parameters,
                returns the off diagonal element of the density matrix or full matrix.
        """
        if self.fulldm:
            return density_matrix

        if callable(self.alpha):
            result = self.alpha(density_matrix)
        elif callable(self.beta):
            result = self.beta(density_matrix)
        else:
            result = self.alpha.conj() @ density_matrix @ self.beta

        return result

    def postprocess(self):
        self.result = self.zero_cluster * self.result
        if self.fulldm:
            self.result = self.result.filled()
        elif self.normalized:
            self.result = self.result / self.normalization

        super().postprocess()

        # else:
        #     self.result = self.center.eigenvectors @ self.result @ self.center.eigenvectors.conj().T

    def generate_hamiltonian(self):
        """
        Using the attributes of the ``self`` object,
        compute the cluster hamiltonian including the central spin.

        Returns:
            Hamiltonian: Cluster hamiltonian.

        """
        ham = total_hamiltonian(self.cluster, self.center, self.magnetic_field)

        return ham

    def compute_result(self):
        """
        Using the attributes of the ``self`` object,
        compute the coherence function of the central spin.

        Returns:

            ndarray: Computed coherence.

        """

        dimensions = shorten_dimensions(self.base_hamiltonian.dimensions, self.center.size)

        initial_state = generate_initial_state(dimensions, states=self.states, central_state=self.dm0)

        unitary_evolution = self.propagator()

        if initial_state.ndim > 1:
            # rho U^\dagger
            dm_udagger = np.matmul(initial_state, unitary_evolution.conj().transpose(0, 2, 1))

            # U rho U^\dagger
            result = np.matmul(unitary_evolution, dm_udagger)
            if self.store_states:
                self.cluster_evolved_states = result.copy()
        else:
            # |dm> = U|dm>
            result = unitary_evolution @ initial_state
            if self.store_states:
                self.cluster_evolved_states = result
            # |dm><dm|
            result = np.einsum('ki,kj->kij', result, result.conj())

        initial_shape = result.shape
        result.shape = (initial_shape[0], *dimensions, *dimensions)

        for d in range(len(dimensions) + 1, 2, -1):  # The last one is el spin
            result = np.trace(result, axis1=1, axis2=d)

            if result.shape[1:] == self.dm0.shape:  # break if shape is the same
                break

        result = self.process_dm(result)

        return result / self.zero_cluster

    def propagator(self):
        """
        Function to compute time propagator U.

        Returns:
            ndarray with shape (t, n, n): Array of propagators, evaluated at each time point in ``self.timespace``.
        """
        if not self.pulses:
            return simple_propagator(self.timespace, self.hamiltonian)

        if self.delays is None:
            if self.projected_states is None:
                return self._no_delays_no_ps()
            # proj_states is not None - there are bath rotations alas
            return self._no_delays_ps()

        # There are delays but no bath flips
        if self.projected_states is None:
            return self._delays_no_ps()

        # The most complicated case - both projected_states is not None and delays is not None
        return self._delays_ps()

    def _no_delays_no_ps(self):

        delays = self.timespace if self.as_delay else self.timespace / (2 * len(self.pulses))

        # Same propagator for all parts
        u = simple_propagator(delays, self.hamiltonian)

        return rotation_propagator(u, self.rotations)

    def _no_delays_ps(self):
        delays = self.timespace if self.as_delay else self.timespace / (2 * len(self.pulses))

        self.get_hamiltonian_variable_bath_state(0)
        u = simple_propagator(delays, self.hamiltonian)

        full_u = np.eye(self.base_hamiltonian.data.shape[0], dtype=np.complex128)

        ps_counter = 0

        for p, rotation in zip(self.pulses, self.rotations):

            full_u = np.matmul(u, full_u)

            if rotation is not None:
                full_u = np.matmul(rotation, full_u)

                if p.bath_names is not None:
                    ps_counter += 1
                    self.get_hamiltonian_variable_bath_state(ps_counter)
                    u = simple_propagator(delays, self.hamiltonian)

            full_u = np.matmul(u, full_u)

        return full_u

    def _delays_no_ps(self):

        evalues, evec = np.linalg.eigh(self.hamiltonian * PI2)

        full_u = np.eye(self.base_hamiltonian.data.shape[0], dtype=np.complex128)
        times = 0

        for delay, rotation in zip(self.delays, self.rotations):
            times += delay

            eigexp = np.exp(-1j * np.outer(delay, evalues),
                            dtype=np.complex128)

            u = np.matmul(np.einsum('ij,kj->kij', evec, eigexp, dtype=np.complex128),
                          evec.conj().T)

            full_u = np.matmul(u, full_u)

            if rotation is not None:
                full_u = np.matmul(rotation, full_u)

        which = np.isclose(self.timespace, times)

        if ((self.timespace - times)[~which] >= 0).all():
            u = simple_propagator(self.timespace - times, self.hamiltonian)

            full_u = np.matmul(u, full_u)

        elif not which.all():
            raise ValueError(f"Pulse sequence time steps add up to larger than total times. Delays at"
                             f"{self.timespace[(self.timespace - times) < 0]} ms are longer than total time.")

        return full_u

    def _delays_ps(self):
        self.get_hamiltonian_variable_bath_state(0)

        full_u = np.eye(self.hamiltonian.shape[0], dtype=np.complex128)

        ps_counter = 0
        times = 0

        for p, rotation, delay in zip(self.pulses, self.rotations, self.delays):
            times += delay
            u = simple_propagator(delay, self.hamiltonian)
            full_u = np.matmul(u, full_u)

            if rotation is not None:
                full_u = np.matmul(rotation, full_u)

                if p.bath_names is not None:
                    ps_counter += 1
                    self.get_hamiltonian_variable_bath_state(ps_counter)

        which = np.isclose(self.timespace, times)

        if ((self.timespace - times)[~which] >= 0).all():
            u = simple_propagator(self.timespace - times, self.hamiltonian)

            full_u = np.matmul(u, full_u)

        elif not which.all():
            raise ValueError(f"Pulse sequence time steps add up to larger than total times. Delays at"
                             f"{self.timespace[(self.timespace - times) < 0]} ms are longer than total time.")

        return full_u

# def propagator(timespace, hamiltonian,
#                pulses=None, as_delay=False):

#     evalues, evec = np.linalg.eigh(hamiltonian * PI2)
#
#     if not pulses:
#
#         eigexp = np.exp(-1j * np.outer(timespace, evalues),
#                         dtype=np.complex128)
#
#         u = np.matmul(np.einsum('ij,kj->kij', evec, eigexp, dtype=np.complex128),
#                       evec.conj().T)
#
#         return u
#
#     else:
#
#         if pulses.delays is None:
#             if not as_delay:
#                 n = len(pulses)
#                 timespace = timespace / (2 * n)
#
#             eigexp = np.exp(-1j * np.outer(timespace, evalues),
#                             dtype=np.complex128)
#
#             u = np.matmul(np.einsum('ij,kj->kij', evec, eigexp, dtype=np.complex128),
#                           evec.conj().T)
#             U = np.eye(u.shape[1], dtype=np.complex128)
#
#             for rotation in pulses.rotations:
#                 U = np.matmul(u, U)
#                 if rotation is not None:
#                     U = np.matmul(rotation, U)
#                 U = np.matmul(u, U)
#
#             return U
#
#         U = None
#
#         times = 0
#         for timesteps, rotation in zip(pulses.delays, pulses.rotations):
#
#             eigexp = np.exp(-1j * np.outer(timesteps, evalues),
#                             dtype=np.complex128)
#
#             u = np.matmul(np.einsum('...ij,...j->...ij', evec, eigexp, dtype=np.complex128),
#                           evec.conj().T)
#             times += timesteps
#
#             if U is None:
#                 if rotation is not None:
#                     U = np.matmul(rotation, u)
#                 else:
#                     U = u
#
#             else:
#                 U = np.matmul(u, U)
#                 if rotation is not None:
#                     U = np.matmul(rotation, U)
#
#         which = np.isclose(timespace, times)
#
#         if ((timespace - times)[~which] >= 0).all():
#             eigexp = np.exp(-1j * np.outer(timespace - times, evalues),
#                             dtype=np.complex128)
#
#             u = np.matmul(np.einsum('ij,kj->kij', evec, eigexp, dtype=np.complex128),
#                           evec.conj().T)
#
#             U = np.matmul(u, U)
#         elif not which.all():
#             raise ValueError(f"Pulse sequence time steps add up to larger than total times. Delays at"
#                              f"{timespace[(timespace - times) < 0]} ms are longer than total time.")
#     return U
#
#
# def jit_propagator_advanced(timespace, hamiltonian,
#                             total_delays, total_rotations):
#     evalues, evec = np.linalg.eigh(hamiltonian * PI2)
#
#     times = np.zeros(timespace.shape)
#     propagators = np.zeros(timespace.shape + hamiltonian.shape)
#     eye = np.eye(hamiltonian.shape[0], dtype=np.complex128)
#
#     for index in range(timespace.size):
#         u = eye
#
#         total_time = timespace[index]
#         delays = total_delays[:, index]
#
#         applicable_rotations = delays <= total_time
#         delays = delays[applicable_rotations]
#         rotations = total_rotations[applicable_rotations]
#         order = np.argsort(delays)
#         passed_time = 0
#
#         for i in order:
#             rotation = rotations[i]
#             t = delays[i] - passed_time
#             if t:
#                 eigexp = np.exp(-1j * t * evalues, dtype=np.complex128)
#                 u = (evec @ np.diag(eigexp) @ evec.conj().T) @ u
#                 passed_time = delays[i]
#
#             if not (rotation == eye).all():
#                 u = rotation @ u
#
#         propagators[index] = u
#
#     return propagators

#
# def compute_dm(initial_state, H, timespace, pulse_sequence=None, as_delay=False, states=None, ncenters=1):
#     """
#     Function to compute density matrix of the central spin, given Hamiltonian H.
#
#     Args:
#         initial_state (ndarray): Initial density matrix of central spin.
#
#         H (ndarray): Cluster Hamiltonian.
#
#         timespace (ndarray): Time points at which to compute density matrix.
#
#         pulse_sequence (Sequence): Sequence of pulses.
#
#         as_delay (bool):
#             True if time points are delay between pulses, False if time points are total time.
#
#         states (ndarray): ndarray of bath states in any accepted format.
#
#         ncenters (int): Number of central spins.
#
#     Returns:
#         ndarray: Array of density matrices evaluated at all time points in timespace.
#     """
#     center_shape = initial_state.shape
#     dimensions = shorten_dimensions(H.dimensions, ncenters)
#
#     initial_state = generate_initial_state(dimensions, states=states, central_state=initial_state)
#     dm = full_dm(initial_state, H, timespace, pulse_sequence=pulse_sequence, as_delay=as_delay)
#     initial_shape = dm.shape
#     dm.shape = (initial_shape[0], *dimensions, *dimensions)
#     for d in range(len(dimensions) + 1, 2, -1):  # The last one is el spin
#         dm = np.trace(dm, axis1=1, axis2=d)
#         if dm.shape[1:] == center_shape:  # break if shape is the same
#             break
#     return dm
#
#
# def full_dm(dm0, H, timespace, pulse_sequence=None, as_delay=False):
#     """
#     A function to compute density matrix of the cluster, using hamiltonian H
#     from the initial density matrix of the cluster.
#
#     Args:
#         dm0 (ndarray):
#             Initial density matrix of the cluster
#         H (ndarray):
#             Cluster Hamiltonian
#         timespace (ndarray): Time points at which to compute coherence function.
#         pulse_sequence (Sequence): Sequence of pulses.
#
#         as_delay (bool):
#             True if time points are delay between pulses, False if time points are total time. Ignored if delays
#             are provided in the ``pulse_sequence``.
#
#     Returns:
#         ndarray: Array of density matrices of the cluster, evaluated at the time points from timespace.
#     """
#     dm = 0
#     return dm
