import numpy as np
from pycce.constants import PI2
from pycce.h.functions import projected_addition
from pycce.h.total import bath_hamiltonian
from pycce.run.base import RunObject, simple_propagator, generate_initial_state


def _rotmul(rotation, u, **kwargs):
    if rotation is not None and u is not None:
        u = np.matmul(rotation, u, **kwargs)
    elif rotation is not None:
        u = rotation

    return u


def simple_propagators(delays, hamiltonian_alpha, hamiltonian_beta):
    r"""
    Generate two simple propagators :math:`U=\exp[-\frac{i}{\hbar} \hat H]` from the Hamiltonians, conditioned
    on two qubit levels.

    Args:
        delays (ndarray with shape (n, )): Time points at which to evaluate the propagator.
        hamiltonian_alpha (ndarray with shape (N, N)): Hamiltonian of the bath spins with qubit in alpha state.
        hamiltonian_beta (ndarray with shape (N, N)): Hamiltonian of the bath spins with qubit in beta state.

    Returns:
        tuple:
            * **ndarray with shape (n, N, N)**:
              Matrix representation of the propagator conditioned on the alpha qubit state for each time point.
            * **ndarray with shape (n, N, N)**:
              Matrix representation of the propagator conditioned on the beta qubit state for each time point.

    """
    u0 = simple_propagator(delays, hamiltonian_alpha)
    u1 = simple_propagator(delays, hamiltonian_beta)

    return u0, u1


def propagate_propagators(v0, v1, number):
    """
    From two simple propagators and number of pulses in CPMG sequence generate two full propagators.
    Args:
        v0 (ndarray with shape (n, N, N)): Propagator conditioned on the alpha qubit state for each time point.
        v1 (ndarray with shape (n, N, N)): Propagator conditioned on the beta qubit state for each time point.
        number (int): Number of pulses.

    Returns:
        tuple:
            * **ndarray with shape (n, N, N)**:
              Matrix representation of the propagator conditioned on the alpha qubit state for each time point.
            * **ndarray with shape (n, N, N)**:
              Matrix representation of the propagator conditioned on the beta qubit state for each time point.

    """
    v0_he = np.matmul(v0, v1, dtype=np.complex128)
    v1_he = np.matmul(v1, v0, dtype=np.complex128)

    if number == 1:
        return v0_he, v1_he

    v0_cp = np.matmul(v0_he, v1_he, dtype=np.complex128)  # v0 @ v1 @ v1 @ v0
    v1_cp = np.matmul(v1_he, v0_he, dtype=np.complex128)  # v1 @ v0 @ v0 @ v1

    unitary_0 = np.linalg.matrix_power(v0_cp, number // 2)
    unitary_1 = np.linalg.matrix_power(v1_cp, number // 2)

    if number % 2 == 1:
        unitary_0 = np.matmul(unitary_0, v0_he)
        unitary_1 = np.matmul(unitary_1, v1_he)

    return unitary_0, unitary_1


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

    def __init__(self, *args, second_order=False,
                 level_confidence=0.95, **kwargs):
        self.key_alpha = None
        self.key_beta = None

        self.initial_pulses = None
        """int or Sequence: Input pulses"""
        self.pulses = None
        """int or Sequence: If input Sequence contains only pi pulses at even delay, stores number of pulses.
        Otherwise stores full ``Sequence``."""
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
        self.initial_pulses = self.pulses
        """ Sequence: Sequence object, containing series of pulses, applied to the system."""
        try:
            pulses = self.initial_pulses
            number = len(self.initial_pulses)

            angles = np.array([p._angles for p in pulses])
            naxes = [p.naxes for p in pulses]

            if not number:
                self.pulses = number

            else:
                c0 = all(p.which is None for p in pulses)
                c1 = all(n <= 1 for n in naxes)
                c2 = all([not p.keys() for p in pulses])
                c3 = all([not p._has_delay for p in pulses])
                c4 = np.isclose(angles, np.pi).any(axis=1).all()

                if c0 & c1 & c2 & c3 & c4:
                    self.pulses = number
                else:
                    self.pulses = self.initial_pulses
                    self.use_pulses = True

                if not c1 or not (np.isclose(angles, np.pi) | np.isclose(angles, 0)).all():
                    raise ValueError('Only pi-pulses are supported in CCE. Use gCCE for user-defined sequences.')

                if not c0 and self.second_order:
                    raise ValueError('Only full flip pulses are supported in CCE with second order corrections')

        except TypeError:
            self.pulses = self.initial_pulses

        self.center.generate_projections(second_order=self.second_order, level_confidence=self.level_confidence)

        if self.center.state_index is not None and isinstance(self.center.state_index, np.ndarray):
            self.key_alpha = self.center.state_index[:, 0]
            self.key_beta = self.center.state_index[:, 1]
        else:
            self.key_alpha = np.ones(len(self.center), dtype=bool)
            self.key_beta = np.zeros(len(self.center), dtype=bool)

    def postprocess(self):
        super().postprocess()
        self.h = None
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
        hamil = bath_hamiltonian(self.cluster, self.magnetic_field)

        return hamil

    def compute_result(self):
        """
        Using the attributes of the ``self`` object,
        compute the coherence function as overlap in the bath evolution.

        Returns:

            ndarray: Computed coherence.

        """
        unitary_0, unitary_1 = self.propagators()

        if self.states is None:
            coherence_function = np.einsum('zij,zij->z', unitary_0, unitary_1.conj()) / unitary_0.shape[1]
            if self.store_states:
                self.cluster_evolved_states = unitary_0 / unitary_0.shape[1], unitary_1 / unitary_1.shape[1]
        else:
            dm = generate_initial_state(self.base_hamiltonian.dimensions, states=self.states)
            # tripple einsum is slow
            # coherence_function = np.einsum('zli,ij,zlj->z', U0, dm, U1.conj())
            if len(dm.shape) > 1:

                dm_udagger = np.matmul(dm, unitary_1.conj().transpose(0, 2, 1))
                coherence_function = np.trace(np.matmul(unitary_0, dm_udagger), axis1=1, axis2=2)
                if self.store_states:
                    dm0_udagger = np.matmul(dm, unitary_0.conj().transpose(0, 2, 1))
                    dm0 = np.matmul(unitary_0, dm0_udagger)

                    dm1_udagger = np.matmul(dm, unitary_1.conj().transpose(0, 2, 1))
                    dm1 = np.matmul(unitary_1, dm1_udagger)
                    self.cluster_evolved_states = dm0, dm1
            else:

                rightside = unitary_1 @ dm
                leftside = unitary_0 @ dm

                coherence_function = np.einsum('ki,ki->k', leftside.conj(), rightside)
                if self.store_states:
                    self.cluster_evolved_states = leftside, rightside

        return coherence_function

    def propagators(self):
        """
        Generate two propagators, conditioned on the qubit state.

        Returns:
            tuple: *tuple* containing:

                * **ndarray with shape (t, n, n)**:
                  Matrix representation of the propagator conditioned on the alpha qubit state for each time point.
                * **ndarray with shape (t, n, n)**:
                  Matrix representation of the propagator conditioned on the beta qubit state for each time point.

        """
        if not self.use_pulses:
            return self._no_pulses()

        if self.delays is None:
            return self._no_delays()

        return self._delays()

    def _proj_ham(self, index=0, alpha=True, beta=False):
        self.get_hamiltonian_variable_bath_state(index)

        ha = self.hamiltonian + projected_addition(self.base_hamiltonian.vectors,
                                                   self.cluster, self.center, alpha)

        hb = self.hamiltonian + projected_addition(self.base_hamiltonian.vectors,
                                                   self.cluster, self.center, beta)

        return ha, hb

    def _no_pulses(self):
        ha, hb = self._proj_ham()
        delays = self.timespace / (2 * self.pulses) if ((not self.as_delay) and self.pulses) else self.timespace
        v0, v1 = simple_propagators(delays, ha, hb)

        if not self.pulses:
            return v0, v1

        else:
            return propagate_propagators(v0, v1, self.pulses)

    def _no_delays(self):
        delays = self.timespace if self.as_delay else self.timespace / (2 * len(self.pulses))

        key_alpha = list(self.key_alpha)
        key_beta = list(self.key_beta)

        ha, hb = self._proj_ham(alpha=key_alpha, beta=key_beta)

        v0, v1 = simple_propagators(delays, ha, hb)
        vs = {}

        vs[tuple(key_alpha)] = v0
        vs[tuple(key_beta)] = v1

        unitary_0 = np.eye(v0.shape[1], dtype=np.complex128)
        unitary_1 = np.eye(v1.shape[1], dtype=np.complex128)

        ps_counter = 0

        for p, rotation in zip(self.pulses, self.rotations):
            unitary_0 = np.matmul(v0, unitary_0)
            unitary_1 = np.matmul(v1, unitary_1)

            unitary_0 = _rotmul(rotation, unitary_0)
            unitary_1 = _rotmul(rotation, unitary_1)

            if p.bath_names is not None:
                ps_counter += 1
                vs.clear()

            key_alpha, key_beta = _gen_key(p, key_alpha, key_beta)

            try:
                v0, v1 = vs[tuple(key_alpha)], vs[tuple(key_beta)]

            except KeyError:

                ha, hb = self._proj_ham(index=ps_counter, alpha=key_alpha, beta=key_beta)
                v0, v1 = simple_propagators(delays, ha, hb)

                vs[tuple(key_alpha)] = v0
                vs[tuple(key_beta)] = v1

            unitary_0 = np.matmul(v0, unitary_0)
            unitary_1 = np.matmul(v1, unitary_1)

        return unitary_0, unitary_1

    def _delays(self):

        times = 0
        key_alpha = list(self.key_alpha)
        key_beta = list(self.key_beta)

        ha, hb = self._proj_ham(alpha=key_alpha, beta=key_beta)

        eval0, evec0 = np.linalg.eigh(ha * PI2)
        eval1, evec1 = np.linalg.eigh(hb * PI2)

        # for timesteps, rotation in zip(pulses.delays, pulses.rotations):
        eval_evec = {}
        eval_evec[tuple(key_alpha)] = eval0, evec0
        eval_evec[tuple(key_beta)] = eval1, evec1

        unitary_0 = None
        unitary_1 = None
        ps_counter = 0
        for p, delay, rotation in zip(self.pulses, self.delays, self.rotations):
            if np.any(delay):
                eigen_exp0 = np.exp(-1j * np.outer(delay, eval0), dtype=np.complex128)

                eigen_exp1 = np.exp(-1j * np.outer(delay, eval1), dtype=np.complex128)

                u0 = np.matmul(np.einsum('...ij,...j->...ij', evec0, eigen_exp0, dtype=np.complex128), evec0.conj().T)

                u1 = np.matmul(np.einsum('...ij,...j->...ij', evec1, eigen_exp1, dtype=np.complex128), evec1.conj().T)

                times += delay

                unitary_0 = _rotmul(rotation, u0) if unitary_0 is None else np.matmul(u0, _rotmul(rotation, unitary_0))
                unitary_1 = _rotmul(rotation, u1) if unitary_1 is None else np.matmul(u1, _rotmul(rotation, unitary_1))
            else:
                unitary_0 = _rotmul(rotation, unitary_0)
                unitary_1 = _rotmul(rotation, unitary_1)

            if p.bath_names is not None:
                ps_counter += 1
                eval_evec.clear()

            key_alpha, key_beta = _gen_key(p, key_alpha, key_beta)

            try:
                eval0, evec0 = eval_evec[tuple(key_alpha)]
                eval1, evec1 = eval_evec[tuple(key_beta)]
            except KeyError:
                ha, hb = self._proj_ham(index=ps_counter, alpha=key_alpha, beta=key_beta)
                eval0, evec0 = np.linalg.eigh(ha * PI2)
                eval1, evec1 = np.linalg.eigh(hb * PI2)

                eval_evec[tuple(key_alpha)] = eval0, evec0
                eval_evec[tuple(key_beta)] = eval1, evec1

        which = np.isclose(self.timespace, times)
        if ((self.timespace - times)[~which] >= 0).all():

            eigen_exp0 = np.exp(-1j * np.outer(self.timespace - times,
                                               eval0), dtype=np.complex128)
            eigen_exp1 = np.exp(-1j * np.outer(self.timespace - times,
                                               eval1), dtype=np.complex128)

            u0 = np.matmul(np.einsum('ij,kj->kij', evec0, eigen_exp0, dtype=np.complex128),
                           evec0.conj().T)

            u1 = np.matmul(np.einsum('ij,kj->kij', evec1, eigen_exp1, dtype=np.complex128),
                           evec1.conj().T)

            unitary_0 = np.matmul(u0, unitary_0)
            unitary_1 = np.matmul(u1, unitary_1)

        elif not which.all():
            raise ValueError(f"Pulse sequence time steps add up to larger than total times"
                             f"{np.argwhere((self.timespace - times) < 0)} are longer than total time.")

        return unitary_0, unitary_1


def _gen_key(p, key_alpha, key_beta):
    if p.flip:
        if p.which is None:
            # key_alpha, key_beta = key_beta, key_alpha
            key_alpha = [not k for k in key_alpha]
            key_beta = [not k for k in key_beta]
        else:
            for index in p.which:
                key_alpha[index] = not key_alpha[index]
                key_beta[index] = not key_beta[index]
    return key_alpha, key_beta

# def compute_coherence(H0, H1, timespace, N, as_delay=False, states=None):
#     """
#     Function to compute cluster coherence function in conventional CCE.
#
#     Args:
#         H0 (ndarray): Hamiltonian projected on alpha qubit state.
#         H1 (ndarray): Hamiltonian projected on beta qubit state.
#         timespace (ndarray): Time points at which to compute coherence function.
#         N (int): Number of pulses in CPMG.
#         as_delay (bool):
#             True if time points are delay between pulses,
#             False if time points are total time.
#         states (ndarray): ndarray of bath states in any accepted format.
#
#     Returns:
#         ndarray: Coherence function of the central spin.
#
#     """
#     # if timespace was given not as delay between pulses,
#     # divide to obtain the delay
#     coherence_function = None
#     return coherence_function

# def propagators(timespace, H0, H1, pulses, as_delay=False):
#     """
#     Function to compute propagators U0 and U1 in conventional CCE.
#
#     Args:
#         timespace (ndarray with shape (t, )): Time delay values at which to compute propagators.
#
#         H0 (ndarray with shape (n, n)): Hamiltonian projected on alpha qubit state.
#
#         H1 (ndarray with shape (n, n)): Hamiltonian projected on beta qubit state.
#
#         pulses (int or Sequence): Sequence of pulses.
#
#         as_delay (bool):
#             True if time points are delay between pulses.
#             False if time points are total time.
#
#     Returns:
#         tuple: *tuple* containing:
#
#             * **ndarray with shape (t, n, n)**:
#               Matrix representation of the propagator conditioned on the alpha qubit state for each time point.
#             * **ndarray with shape (t, n, n)**:
#               Matrix representation of the propagator conditioned on the beta qubit state for each time point.
#
#     """
#     try:
#         number = len(pulses)
#         check = (not as_delay) & (pulses.delays is None)
#         use_rotations = True
#     except TypeError:
#         number = pulses
#         check = (not as_delay)
#         use_rotations = False
#
#     if check and number:
#         timespace = timespace / (2 * number)
#
#     if not use_rotations:
#         v0, v1 = simple_propagators(timespace, H0, H1)
#
#         if not number:
#             return v0, v1
#         else:
#             return propagate_propagators(v0, v1, number)
#
#     if pulses.delays is None:
#
#         v0, v1 = simple_propagators(timespace, H0, H1)
#
#         vs = [v0, v1]
#
#         U0 = np.eye(v0.shape[1], dtype=np.complex128)
#         U1 = np.eye(v1.shape[1], dtype=np.complex128)
#
#         Us = [U0, U1]
#
#         for p in pulses:
#             for i in range(2):
#                 Us[i] = np.matmul(vs[i], Us[i])
#                 Us[i] = _rotmul(p.rotation, Us[i])
#
#             if p.flip:
#                 vs = vs[::-1]
#
#             for i in range(2):
#                 Us[i] = np.matmul(vs[i], Us[i])
#
#         return Us[0], Us[1]
#
#     U0 = None
#     U1 = None
#
#     times = 0
#
#     eval0, evec0 = np.linalg.eigh(H0 * PI2)
#     eval1, evec1 = np.linalg.eigh(H1 * PI2)
#
#     # for timesteps, rotation in zip(pulses.delays, pulses.rotations):
#     evalues = [eval0, eval1]
#     evec = [evec0, evec1]
#
#     for p in pulses:
#         timesteps = p.delay
#         rotation = p.rotation
#         eigen_exp0 = np.exp(-1j * np.outer(timesteps,
#                                            evalues[0]), dtype=np.complex128)
#         eigen_exp1 = np.exp(-1j * np.outer(timesteps,
#                                            evalues[1]), dtype=np.complex128)
#
#         u0 = np.matmul(np.einsum('...ij,...j->...ij', evec[0], eigen_exp0, dtype=np.complex128),
#                        evec[0].conj().T)
#
#         u1 = np.matmul(np.einsum('...ij,...j->...ij', evec[1], eigen_exp1, dtype=np.complex128),
#                        evec[1].conj().T)
#
#         times += timesteps
#
#         if U0 is None:
#             U0 = _rotmul(rotation, u0)
#             U1 = _rotmul(rotation, u1)
#
#         else:
#             U0 = np.matmul(u0, U0)
#             U0 = _rotmul(rotation, U0)
#
#             U1 = np.matmul(u1, U1)
#             U1 = _rotmul(rotation, U1)
#
#         if p.flip:
#             evalues = evalues[::-1]
#             evec = evec[::-1]
#
#     which = np.isclose(timespace, times)
#     if ((timespace - times)[~which] >= 0).all():
#
#         eigen_exp0 = np.exp(-1j * np.outer(timespace - times,
#                                            evalues[0]), dtype=np.complex128)
#         eigen_exp1 = np.exp(-1j * np.outer(timespace - times,
#                                            evalues[1]), dtype=np.complex128)
#
#         u0 = np.matmul(np.einsum('ij,kj->kij', evec[0], eigen_exp0, dtype=np.complex128),
#                        evec[0].conj().T)
#
#         u1 = np.matmul(np.einsum('ij,kj->kij', evec[1], eigen_exp1, dtype=np.complex128),
#                        evec[1].conj().T)
#
#         U0 = np.matmul(u0, U0)
#         U1 = np.matmul(u1, U1)
#
#     elif not which.all():
#         raise ValueError(f"Pulse sequence time steps add up to larger than total times"
#                          f"{np.argwhere((timespace - times) < 0)} are longer than total time.")
#
#     return U0, U1
#
