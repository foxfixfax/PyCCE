import numpy as np
import scipy
from numpy import ma as ma
from pycce.bath.array import BathArray
from pycce.constants import PI2
from pycce.h import total_hamiltonian, projected_addition
from pycce.run.base import RunObject, generate_initial_state, simple_propagator
from pycce.utilities import shorten_dimensions, outer, expand
from pycce.run.gcce import gCCE, rotation_propagator
from pycce.run.cce import CCE, _rotmul, _gen_key

from pycce.sm import _smc


def simple_incoherent_propagator(timespace, lindbladian):
    r"""
    Generate a simple propagator :math:`U=\exp[ \mathcal{L}]` from the Lindbladian superoperator.

    Args:

        timespace (ndarray with shape (n, )): Time points at which to evaluate the propagator.
        lindbladian (ndarray with shape (N, N)): Lindbladian superoperator of the system in matrix form.

    Returns:
        ndarray with shape (n, N, N): Master equation propagators, evaluated at each timepoint. Use with vector form
            of density matrix.
    """
    return scipy.linalg.expm(timespace[:, np.newaxis, np.newaxis] * lindbladian[np.newaxis, :] * PI2)

    # evalues, evec = np.linalg.eig(lindbladian * PI2)
    #
    # eigexp = np.exp(np.outer(timespace, evalues),
    #                 dtype=np.complex128)
    #
    # return np.matmul(np.einsum('...ij,...j->...ij', evec, eigexp, dtype=np.complex128),
    #                  evec.conj().T)


def collapse_superoperator(superoperators, index, dims):
    sm = _smc[(dims[index] - 1) / 2]
    full_lindb = 0
    eye = np.eye(dims.prod(), dtype=np.complex128)
    for key in superoperators:
        if isinstance(key, str):
            collapse = None
            for sym in key:
                collapse = superoperators[key] * getattr(sm, sym) if collapse is None else np.matmul(collapse,
                                                                                                     getattr(sm, sym))
        else:
            collapse = superoperators[key] * sm.stev(*key)
        cn = expand(collapse, index, dims)
        cn_rho_cndag = op_to_supop(cn, cn.conj().T)
        rho_cndag_cn = op_to_supop(eye, cn.conj().T @ cn)
        cndag_cn_rho = op_to_supop(cn.conj().T @ cn, eye)
        lindb = 1 / 2 * (2 * cn_rho_cndag - rho_cndag_cn - cndag_cn_rho)
        full_lindb += lindb
    return full_lindb


def incoherent_superoperator(spins, dims=None, offset=0):
    if dims is None:
        dims = spins.dim
    add = 0
    for index, b in enumerate(spins):
        if b.so:
            add += collapse_superoperator(b.so, index + offset, dims)

    return add


def coherent_superoperator(hamiltonian):
    eye = np.eye(hamiltonian.shape[0], dtype=np.complex128)
    return -1j * (op_to_supop(hamiltonian, eye) - op_to_supop(eye, hamiltonian))


def projected_coherent_superoperator(hamiltonian0, hamiltonian1):
    eye = np.eye(hamiltonian0.shape[0], dtype=np.complex128)
    return -1j * (op_to_supop(hamiltonian0, eye) - op_to_supop(eye, hamiltonian1))


def mat_to_vec(operator):
    return operator.reshape(np.prod(operator.shape))


def vec_to_mat(vector):
    side = np.sqrt(vector.shape[-1])
    if not int(side) == side:
        raise ValueError('Unsupported vector shape')
    return vector.reshape(*vector.shape[:-1], int(side), int(side))


def op_to_supop(left_operator, right_operator):
    return np.kron(left_operator, right_operator.T)


class LindbladgCCE(gCCE):
    def __init__(self, *args, **kwargs):
        self.superoperator = None
        super().__init__(*args, **kwargs)

    def preprocess(self):
        super().preprocess()

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

    def get_superoperator(self):
        self.superoperator = coherent_superoperator(self.hamiltonian)
        self.superoperator += incoherent_superoperator(self.cluster, self.base_hamiltonian.dimensions)
        self.superoperator += incoherent_superoperator(self.center, self.base_hamiltonian.dimensions,
                                                       offset=self.cluster.size)

    def super_propagator(self):

        if not self.pulses:
            return simple_incoherent_propagator(self.timespace, self.superoperator)

        if self.delays is None:
            if self.projected_states is None:
                return self._no_delays_no_ps_super()
            # proj_states is not None - there are bath rotations alas
            return self._no_delays_ps_super()

        # There are delays but no bath flips
        if self.projected_states is None:
            return self._delays_no_ps_super()

        # The most complicated case - both projected_states is not None and delays is not None
        return self._delays_ps_super()

    def _no_delays_no_ps_super(self):

        delays = self.timespace if self.as_delay else self.timespace / (2 * len(self.pulses))
        rotations = [op_to_supop(rot, rot.conj().T) if rot is not None else None for rot in self.rotations]

        # Same propagator for all parts
        u = simple_incoherent_propagator(delays, self.superoperator)

        return rotation_propagator(u, rotations)

    def _no_delays_ps_super(self):
        delays = self.timespace if self.as_delay else self.timespace / (2 * len(self.pulses))
        rotations = [op_to_supop(rot, rot.conj().T) if rot is not None else None for rot in self.rotations]

        self.get_hamiltonian_variable_bath_state(0)
        self.get_superoperator()

        u = simple_incoherent_propagator(delays, self.superoperator)

        full_u = np.eye(self.superoperator.shape[0], dtype=np.complex128)

        ps_counter = 0

        for p, rotation in zip(self.pulses, rotations):

            full_u = np.matmul(u, full_u)

            if rotation is not None:
                full_u = np.matmul(rotation, full_u)

                if p.bath_names is not None:
                    ps_counter += 1
                    self.get_hamiltonian_variable_bath_state(ps_counter)
                    u = simple_incoherent_propagator(delays, self.superoperator)

            full_u = np.matmul(u, full_u)

        return full_u

    def _delays_no_ps_super(self):

        evalues, evec = np.linalg.eigh(self.superoperator * PI2)
        rotations = [op_to_supop(rot, rot.conj().T) if rot is not None else None for rot in self.rotations]
        full_u = np.eye(self.superoperator.shape[0], dtype=np.complex128)
        times = 0
        for delay, rotation in zip(self.delays, rotations):
            times += delay

            eigexp = np.exp(np.outer(delay, evalues),
                            dtype=np.complex128)

            u = np.matmul(np.einsum('ij,kj->kij', evec, eigexp, dtype=np.complex128),
                          evec.conj().T)

            full_u = np.matmul(u, full_u)

            if rotation is not None:
                full_u = np.matmul(rotation, full_u)

        which = np.isclose(self.timespace, times)

        if ((self.timespace - times)[~which] >= 0).all():
            u = simple_incoherent_propagator(self.timespace - times, self.superoperator)

            full_u = np.matmul(u, full_u)

        elif not which.all():
            raise ValueError(f"Pulse sequence time steps add up to larger than total times. Delays at"
                             f"{self.timespace[(self.timespace - times) < 0]} ms are longer than total time.")

        return full_u

    def _delays_ps_super(self):
        self.get_hamiltonian_variable_bath_state(0)
        self.get_superoperator()
        rotations = [op_to_supop(rot, rot.conj().T) if rot is not None else None for rot in self.rotations]
        full_u = np.eye(self.superoperator.shape[0], dtype=np.complex128)

        ps_counter = 0
        times = 0

        for p, rotation, delay in zip(self.pulses, rotations, self.delays):
            times += delay
            u = simple_incoherent_propagator(delay, self.superoperator)
            full_u = np.matmul(u, full_u)

            if rotation is not None:
                full_u = np.matmul(rotation, full_u)

                if p.bath_names is not None:
                    ps_counter += 1
                    self.get_hamiltonian_variable_bath_state(ps_counter)

        which = np.isclose(self.timespace, times)

        if ((self.timespace - times)[~which] >= 0).all():
            u = simple_incoherent_propagator(self.timespace - times, self.superoperator)

            full_u = np.matmul(u, full_u)

        elif not which.all():
            raise ValueError(f"Pulse sequence time steps add up to larger than total times. Delays at"
                             f"{self.timespace[(self.timespace - times) < 0]} ms are longer than total time.")

        return full_u

    def compute_result(self):
        """
        Using the attributes of the ``self`` object,
        compute the coherence function of the central spin.

        Returns:

            ndarray: Computed coherence.

        """
        self.get_superoperator()

        dimensions = shorten_dimensions(self.base_hamiltonian.dimensions, self.center.size)

        initial_state = generate_initial_state(dimensions, states=self.states, central_state=self.dm0)

        if initial_state.ndim == 1:
            initial_state = outer(initial_state, initial_state)

        initial_state = mat_to_vec(initial_state)
        non_unitary_evolution = self.super_propagator()

        result = non_unitary_evolution @ initial_state
        result = vec_to_mat(result)

        if self.store_states:
            self.cluster_evolved_states = result.copy()

        initial_shape = result.shape
        result.shape = (initial_shape[0], *dimensions, *dimensions)

        for d in range(len(dimensions) + 1, 2, -1):  # The last one is el spin
            result = np.trace(result, axis1=1, axis2=d)

            if result.shape[1:] == self.dm0.shape:  # break if shape is the same
                break

        result = self.process_dm(result)

        return result / self.zero_cluster


def propagate_superpropagators(u_before_pi, u_after_pi, number):
    v_he = np.matmul(u_after_pi, u_before_pi, dtype=np.complex128)

    if number == 1:
        return v_he

    v_he_reversed = np.matmul(u_before_pi, u_after_pi, dtype=np.complex128)
    v_cp = np.matmul(v_he_reversed, v_he, dtype=np.complex128)  # v0 @ v1 @ v1 @ v0

    if number == 2:
        return v_cp

    nonunitary = np.linalg.matrix_power(v_cp, number // 2)

    if number % 2 == 1:
        nonunitary = np.matmul(v_he, nonunitary)

    return nonunitary


class LindbladCCE(CCE):
    def __init__(self, *args, **kwargs):
        self.superoperator = None
        super().__init__(*args, **kwargs)

    def preprocess(self):
        super().preprocess()

    def postprocess(self):
        super().postprocess()

        # else:
        #     self.result = self.center.eigenvectors @ self.result @ self.center.eigenvectors.conj().T

    def get_superoperator(self, alpha=True, beta=False, return_two=False, index=0):
        self.get_hamiltonian_variable_bath_state(index)

        ha = self.hamiltonian + projected_addition(self.base_hamiltonian.vectors,
                                                   self.cluster, self.center, alpha)

        hb = self.hamiltonian + projected_addition(self.base_hamiltonian.vectors,
                                                   self.cluster, self.center, beta)

        addition = (incoherent_superoperator(self.cluster, self.base_hamiltonian.dimensions) +
                    incoherent_superoperator(self.center, self.base_hamiltonian.dimensions,
                                             offset=self.cluster.size))
        self.superoperator = projected_coherent_superoperator(ha, hb) + addition

        if return_two:
            flipped_superoperator = projected_coherent_superoperator(hb, ha) + addition
            return self.superoperator, flipped_superoperator

    def super_propagator(self):

        if not self.use_pulses:
            return self._no_pulses_super()

        if self.delays is None:
            return self._no_delays_super()
        
        return self._delays_super()

    def _no_pulses_super(self):
        delays = self.timespace / (2 * self.pulses) if ((not self.as_delay) and self.pulses) else self.timespace
        if not self.pulses:
            u = simple_incoherent_propagator(delays, self.superoperator)
            return u

        i1, i2 = self.get_superoperator(return_two=True)
        u0, u1 = (simple_incoherent_propagator(delays, isup) for isup in (i1, i2))
        return propagate_superpropagators(u0, u1, self.pulses)

    def _no_delays_super(self):
        delays = self.timespace if self.as_delay else self.timespace / (2 * len(self.pulses))
        rotations = [op_to_supop(rot, rot.conj().T) if rot is not None else None for rot in self.rotations]

        key_alpha = list(self.key_alpha)
        key_beta = list(self.key_beta)
        ps_counter = 0

        self.get_superoperator(alpha=key_alpha, beta=key_beta, index=ps_counter)

        v01 = simple_incoherent_propagator(delays, self.superoperator)
        vs = {(tuple(key_alpha), tuple(key_beta)): v01}
        nonunitary = np.eye(v01.shape[1], dtype=np.complex128)

        for p, rotation in zip(self.pulses, rotations):
            nonunitary = np.matmul(v01, nonunitary)
            nonunitary = _rotmul(rotation, nonunitary)

            if p.bath_names is not None:
                ps_counter += 1
                vs.clear()

            key_alpha, key_beta = _gen_key(p, key_alpha, key_beta)

            try:
                v01 = vs[(tuple(key_alpha), tuple(key_beta))]

            except KeyError:
                self.get_superoperator(alpha=key_alpha, beta=key_beta, index=ps_counter)
                v01 = simple_incoherent_propagator(delays, self.superoperator)

                vs[(tuple(key_alpha), tuple(key_beta))] = v01

            nonunitary = np.matmul(v01, nonunitary)

        return nonunitary

    def _delays_super(self):

        times = 0
        key_alpha = list(self.key_alpha)
        key_beta = list(self.key_beta)
        ps_counter = 0

        self.get_superoperator(alpha=key_alpha, beta=key_beta, index=ps_counter)
        rotations = [op_to_supop(rot, rot.conj().T) if rot is not None else None for rot in self.rotations]

        eval01, evec01 = np.linalg.eigh(self.superoperator * PI2)

        # for timesteps, rotation in zip(pulses.delays, pulses.rotations):
        eval_evec = {(tuple(key_alpha), tuple(key_beta)): (eval01, evec01)}

        nonunitary = None

        for p, delay, rotation in zip(self.pulses, self.delays, rotations):
            if np.any(delay):
                eigen_exp = np.exp(-1j * np.outer(delay, eval01), dtype=np.complex128)
                u01 = np.matmul(np.einsum('...ij,...j->...ij', evec01, eigen_exp, dtype=np.complex128), evec01.conj().T)

                times += delay

                nonunitary = _rotmul(rotation, u01) if nonunitary is None else np.matmul(u01, _rotmul(rotation, nonunitary))
            else:
                nonunitary = _rotmul(rotation, nonunitary)

            if p.bath_names is not None:
                ps_counter += 1
                eval_evec.clear()

            key_alpha, key_beta = _gen_key(p, key_alpha, key_beta)

            try:
                eval01, evec01 = eval_evec[(tuple(key_alpha), tuple(key_beta))]
            except KeyError:
                self.get_superoperator(alpha=key_alpha, beta=key_beta, index=ps_counter)
                eval01, evec01 = np.linalg.eigh(self.superoperator * PI2)

                eval_evec[(tuple(key_alpha), tuple(key_beta))] = eval01, evec01

        which = np.isclose(self.timespace, times)
        if ((self.timespace - times)[~which] >= 0).all():

            eigen_exp = np.exp(-1j * np.outer(self.timespace - times, eval01), dtype=np.complex128)
            u01 = np.matmul(np.einsum('...ij,...j->...ij', evec01, eigen_exp, dtype=np.complex128), evec01.conj().T)

            nonunitary = np.matmul(u01, nonunitary)

        elif not which.all():
            raise ValueError(f"Pulse sequence time steps add up to larger than total times"
                             f"{np.argwhere((self.timespace - times) < 0)} are longer than total time.")

        return nonunitary

    def compute_result(self):
        """
        Using the attributes of the ``self`` object,
        compute the coherence function of the central spin.

        Returns:

            ndarray: Computed coherence.

        """
        self.get_superoperator()

        dimensions = shorten_dimensions(self.base_hamiltonian.dimensions, self.center.size)
        initial_state = generate_initial_state(dimensions, states=self.states)

        if initial_state.ndim == 1:
            initial_state = outer(initial_state, initial_state)

        initial_state = mat_to_vec(initial_state)
        non_unitary_evolution = self.super_propagator()

        result = non_unitary_evolution @ initial_state
        result = vec_to_mat(result)

        if self.store_states:
            self.cluster_evolved_states = result.copy()

        result = np.trace(result, axis1=1, axis2=2)

        return result
