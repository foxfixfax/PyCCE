import numpy as np
from numpy import ma as ma
from pycce.bath.array import BathArray
from pycce.constants import PI2
from pycce.h import total_hamiltonian, expand

from .base import RunObject


def propagator(timespace, hamiltonian,
               pulses=None, as_delay=False):
    """
    Function to compute time propagator U.

    Args:
        timespace (ndarray with shape (t, )):
            Time points at which to compute propagators.

        hamiltonian (ndarray with shape (n, n)):
            Matrix representation of the cluster hamiltonian.

        pulses (Sequence):
            Pulses as an instance of ``Sequence`` class with rotations already generated. Default is None.

        as_delay (bool):
            True if time points are delay between pulses, False if time points are total time. Default is False.

    Returns:
        ndarray with shape (t, n, n): Array of propagators, evaluated at each time point in timespace.
    """
    evalues, evec = np.linalg.eigh(hamiltonian * PI2)

    if not pulses:

        eigexp = np.exp(-1j * np.tensordot(timespace, evalues, axes=0),
                        dtype=np.complex128)

        u = np.matmul(np.einsum('ij,kj->kij', evec, eigexp, dtype=np.complex128),
                      evec.conj().T)

        return u

    else:

        if pulses.delays is None:
            if not as_delay:
                N = len(pulses)
                timespace = timespace / (2 * N)

            eigexp = np.exp(-1j * np.tensordot(timespace, evalues, axes=0),
                            dtype=np.complex128)

            u = np.matmul(np.einsum('ij,kj->kij', evec, eigexp, dtype=np.complex128),
                          evec.conj().T)
            U = np.eye(u.shape[1], dtype=np.complex128)

            for rotation in pulses.rotations:
                U = np.matmul(u, U)
                U = np.matmul(rotation, U)
                U = np.matmul(u, U)

            return U

        U = None

        times = 0

        for timesteps, rotation in zip(pulses.delays, pulses.rotations):

            eigexp = np.exp(-1j * np.tensordot(timesteps, evalues, axes=0),
                            dtype=np.complex128)

            u = np.matmul(np.einsum('...ij,...j->...ij', evec, eigexp, dtype=np.complex128),
                          evec.conj().T)
            times += timesteps

            if U is None:
                U = np.matmul(rotation, u)

            else:
                U = np.matmul(u, U)
                U = np.matmul(rotation, U)

        if ((timespace - times) >= 0).all() and (timespace - times).any():
            eigexp = np.exp(-1j * np.tensordot(timespace - times, evalues, axes=0),
                            dtype=np.complex128)

            u = np.matmul(np.einsum('ij,kj->kij', evec, eigexp, dtype=np.complex128),
                          evec.conj().T)

            U = np.matmul(u, U)
        elif ((timespace - times) < 0).any():
            raise ValueError(f"Pulse sequence time steps add up to larger than total times"
                             f"{np.argwhere((timespace - times) < 0)} are longer than total time.")
    return U


def compute_dm(dm0, H, timespace, pulse_sequence=None, as_delay=False, states=None):
    """
    Function to compute density matrix of the central spin, given Hamiltonian H.

    Args:
        dm0 (ndarray): Initial density matrix of central spin.

        H (ndarray): Cluster Hamiltonian.

        timespace (ndarray): Time points at which to compute density matrix.

        pulse_sequence (Sequence): Sequence of pulses.

        as_delay (bool):
            True if time points are delay between pulses, False if time points are total time.

        states (ndarray): ndarray of bath states in any accepted format.

    Returns:
        ndarray: Array of density matrices evaluated at all time points in timespace.
    """

    dm0 = generate_dm0(dm0, H.dimensions, states)
    dm = full_dm(dm0, H, timespace, pulse_sequence=pulse_sequence, as_delay=as_delay)
    initial_shape = dm.shape

    dm.shape = (initial_shape[0], *H.dimensions, *H.dimensions)
    for d in range(len(H.dimensions) + 1, 2, -1):  # The last one is el spin
        dm = np.trace(dm, axis1=1, axis2=d)
    return dm


def full_dm(dm0, H, timespace, pulse_sequence=None, as_delay=False):
    """
    A function to compute density matrix of the cluster, using hamiltonian H
    from the initial density matrix of the cluster.

    Args:
        dm0 (ndarray):
            Initial density matrix of the cluster
        H (ndarray):
            Cluster Hamiltonian
        timespace (ndarray): Time points at which to compute coherence function.
        pulse_sequence (Sequence): Sequence of pulses.

        as_delay (bool):
            True if time points are delay between pulses, False if time points are total time. Ignored if delays
            are provided in the ``pulse_sequence``.

    Returns:
        ndarray: Array of density matrices of the cluster, evaluated at the time points from timespace.
    """
    U = propagator(timespace, H.data, pulse_sequence, as_delay=as_delay)
    if len(dm0.shape) > 1:
        dmUdagger = np.matmul(dm0, np.transpose(U.conj(), axes=(0, 2, 1)))
        dm = np.matmul(U, dmUdagger)
    else:
        dm = U @ dm0
        dm = np.einsum('ki,kj->kij', dm, dm.conj())

    return dm


def generate_dm0(dm0, dimensions, states=None):
    """
    A function to generate initial density matrix or statevector of the cluster.
    Args:
        dm0 (ndarray):
            Initial density matrix of the central spin.
        dimensions (ndarray):
            ndarray of bath spin dimensions. Last entry - electron spin dimensions.
        states (ndarray):
            ndarray of bath states in any accepted format.

    Returns:
        ndarray:
            Initial density matrix of the cluster
            **OR** statevector if dm0 is vector and ``states`` are provided as list of pure states.
    """

    if states is None:
        dmtotal0 = expand(dm0, len(dimensions) - 1, dimensions) / np.prod(dimensions[:-1])
    elif len(dm0.shape) == 1:
        dmtotal0 = generate_pure_initial_state(dm0, dimensions, states)

    else:
        dmtotal0 = gen_density_matrix(states, dimensions[:-1])
        dmtotal0 = np.kron(dmtotal0, dm0)

    return dmtotal0


def generate_pure_initial_state(state0, dimensions, states):
    """
    A function to generate initial state vector of the cluster with central spin.

    Args:
        state0 (ndarray):
            Initial state of the central spin.
        dimensions (ndarray):
            ndarray of bath spin dimensions. Last entry - electron spin dimensions.
        states (ndarray):
            ndarray of bath states in any accepted format.

    Returns:
        ndarray: Initial state vector of the cluster.
    """

    cluster_state = 1

    for i, s in enumerate(states):
        d = dimensions[i]
        n = int(round((d - 1) / 2 - s))

        state = np.zeros(d)
        state[n] = 1
        cluster_state = np.kron(cluster_state, state)

    with_central_spin = np.kron(cluster_state, state0)

    return with_central_spin


def gen_density_matrix(states=None, dimensions=None):
    r"""
    Generate density matrix from the ndarray of states.

    Args:
        states (ndarray):
            Array of bath spin states. If None, assume completely random state.
            Can have the following forms:

                - array of the :math:`\hat{I}_z` projections for each spin.
                  Assumes that each bath spin is in the pure eigenstate of :math:`\hat{I}_z`.

                - array of the diagonal elements of the density matrix for each spin.
                  Assumes mixed state and the density matrix for each bath spin
                  is diagonal in :math:`\hat{I}_z` basis.

                - array of the density matrices of the bath spins.

        dimensions (ndarray):
            array of bath spin dimensions. Last entry - electron spin dimensions.

    Returns:
        ndarray: Density matrix of the system.
    """
    if states is None:
        tdim = np.prod(dimensions)
        dmtotal0 = np.eye(tdim) / tdim

        return dmtotal0

    dmtotal0 = np.eye(1, dtype=np.complex128)

    for i, s in enumerate(states):

        if not hasattr(s, "__len__"):
            # assume s is int or float showing the spin projection in the pure state
            d = dimensions[i]
            dm_nucleus = np.zeros((d, d), dtype=np.complex128)
            state_number = int(round((d - 1) / 2 - s))
            dm_nucleus[state_number, state_number] = 1

        else:
            if s.shape.__len__() == 1:
                d = dimensions[i]
                dm_nucleus = np.zeros((d, d), dtype=np.complex128)
                np.fill_diagonal(dm_nucleus, s)

            else:
                dm_nucleus = s

        dmtotal0 = np.kron(dmtotal0, dm_nucleus)

    return dmtotal0


class gCCE(RunObject):
    """
    Class for running generalized CCE simulations.

    .. note::

        Subclass of the ``RunObject`` abstract class.

    Args:
        *args: Positional arguments of the ``RunObject``.

        pulses (Sequence): Sequence object, containing series of pulses, applied to the system.

        as_delay (bool):
            True if time points are delay between pulses, False if time points are total time.

        **kwargs: Keyword arguments of the ``RunObject``.

    """

    def __init__(self, *args, as_delay=False, pulses=None, **kwargs):
        self.pulses = pulses
        """ Sequence: Sequence object, containing series of pulses, applied to the system."""
        self.as_delay = as_delay
        """ bool: True if time points are delay between pulses, False if time points are total time."""

        self.dm0 = None
        """ ndarray with shape (2s+1, 2s+1): Initial density matrix of the central spin."""
        self.normalization = None
        """ float: Coherence at time 0."""
        self.zero_cluster = None
        """ ndarray with shape (n,): Coherence computed for the isolated central spin."""
        super().__init__(*args, **kwargs)

    def preprocess(self):
        super().preprocess()

        check = False
        if self.projected_bath_state is not None:
            try:
                check = all(self.projected_bath_state == self.bath_state)
            except TypeError:
                check = False

        if check:
            self.dm0 = self.state
        else:
            self.dm0 = np.tensordot(self.state, self.state, axes=0)

        if self.pulses is not None:
            self.pulses.set_central_spin(self.alpha, self.beta)
            self.pulses.generate_pulses(dimensions=self.hamiltonian.dimensions,
                                        bath=BathArray(0, ), vectors=self.hamiltonian.vectors)

        res = full_dm(self.dm0, self.hamiltonian, self.timespace,
                      pulse_sequence=self.pulses, as_delay=self.as_delay)

        res = self.alpha.conj() @ res @ self.beta

        if len(self.dm0.shape) > 1:
            self.normalization = (self.alpha.conj() @ self.dm0 @ self.beta)
        else:
            self.normalization = np.inner(self.alpha.conj(), self.dm0) * np.inner(self.dm0.conj(), self.beta)

        self.zero_cluster = ma.masked_array(res, mask=(np.isclose(np.abs(res), 0)), dtype=np.complex128)

    def postprocess(self):
        self.result = self.zero_cluster * self.result.filled(0) / self.normalization

    def generate_hamiltonian(self):
        """
        Using the attributes of the ``self`` object,
        compute the cluster hamiltonian including the central spin.

        Returns:
            Hamiltonian: Cluster hamiltonian.

        """
        ham = total_hamiltonian(self.cluster, self.magnetic_field, self.zfs, others=self.others,
                                other_states=self.other_states, central_gyro=self.gyro, central_spin=self.spin)

        if self.pulses is not None:
            self.pulses.generate_pulses(dimensions=ham.dimensions, bath=self.cluster, vectors=ham.vectors)

        return ham

    def compute_result(self):
        """
        Using the attributes of the ``self`` object,
        compute the coherence function of the central spin.

        Returns:

            ndarray: Computed coherence.

        """
        result = compute_dm(self.dm0, self.cluster_hamiltonian, self.timespace, self.pulses,
                            as_delay=self.as_delay, states=self.states)

        result = self.alpha.conj() @ result @ self.beta / self.zero_cluster

        return result
