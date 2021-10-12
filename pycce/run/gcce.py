import numpy as np
from pycce.bath.array import BathArray
from pycce.constants import PI2
from pycce.h import total_hamiltonian, expand
from pycce.run.base import RunObject
from pycce.utilities import shorten_dimensions


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
                n = len(pulses)
                timespace = timespace / (2 * n)

            eigexp = np.exp(-1j * np.tensordot(timespace, evalues, axes=0),
                            dtype=np.complex128)

            u = np.matmul(np.einsum('ij,kj->kij', evec, eigexp, dtype=np.complex128),
                          evec.conj().T)
            U = np.eye(u.shape[1], dtype=np.complex128)

            for rotation in pulses.rotations:
                U = np.matmul(u, U)
                if rotation is not None:
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
                if rotation is not None:
                    U = np.matmul(rotation, u)
                else:
                    U = u

            else:
                U = np.matmul(u, U)
                if rotation is not None:
                    U = np.matmul(rotation, U)

        which = np.isclose(timespace, times)

        if ((timespace - times)[~which] >= 0).all():
            eigexp = np.exp(-1j * np.tensordot(timespace - times, evalues, axes=0),
                            dtype=np.complex128)

            u = np.matmul(np.einsum('ij,kj->kij', evec, eigexp, dtype=np.complex128),
                          evec.conj().T)

            U = np.matmul(u, U)
        elif not which.all():
            raise ValueError(f"Pulse sequence time steps add up to larger than total times. Delays at"
                             f"{timespace[(timespace - times) < 0]} ms are longer than total time.")
    return U


def compute_dm(dm0, H, timespace, pulse_sequence=None, as_delay=False, states=None, ncenters=1):
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

        ncenters (int): Number of central spins.

    Returns:
        ndarray: Array of density matrices evaluated at all time points in timespace.
    """
    center_shape = dm0.shape
    dimensions = shorten_dimensions(H.dimensions, ncenters)

    dm0 = generate_dm0(dm0, dimensions, states)
    dm = full_dm(dm0, H, timespace, pulse_sequence=pulse_sequence, as_delay=as_delay)
    initial_shape = dm.shape
    dm.shape = (initial_shape[0], *dimensions, *dimensions)
    for d in range(len(dimensions) + 1, 2, -1):  # The last one is el spin
        dm = np.trace(dm, axis1=1, axis2=d)
        if dm.shape[1:] == center_shape:  # break if shape is the same
            break
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
        dmUdagger = np.matmul(dm0, U.conj().transpose(0, 2, 1))
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

        fulldm (bool):
            True if return full density matrix. Default False.

        **kwargs: Keyword arguments of the ``RunObject``.

    """

    def __init__(self, *args, as_delay=False, pulses=None, fulldm=False, **kwargs):
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
        self.alpha = None
        self.beta = None
        self.fulldm = fulldm
        """ bool: True if return full density matrix."""

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
            self.dm0 = self.center.state
            self.normalization = np.outer(self.center.state, self.center.state)
        else:
            self.dm0 = np.outer(self.center.state, self.center.state)
            self.normalization = self.dm0

        self.alpha = self.center.alpha
        self.beta = self.center.beta

        if self.pulses is not None:
            self.center.generate_sigma()
            self.pulses.generate_pulses(dimensions=self.center.hamiltonian.dimensions,
                                        bath=BathArray((0,)), vectors=self.center.hamiltonian.vectors,
                                        central_spin=self.center)
            for p in self.pulses:
                if p.rotation is not None:
                    self.alpha = p.rotation @ self.alpha
                    self.beta = p.rotation @ self.beta
                    self.normalization = p.rotation @ self.normalization @ p.rotation.conj()

        res = full_dm(self.dm0, self.center.hamiltonian, self.timespace,
                      pulse_sequence=self.pulses, as_delay=self.as_delay)

        if not self.fulldm:
            res = self.alpha.conj() @ res @ self.beta
            self.normalization = (self.alpha.conj() @ self.normalization @ self.beta)

        self.zero_cluster = res

    def postprocess(self):
        self.result = self.zero_cluster * self.result
        if not self.fulldm:
            self.result = self.result / self.normalization

    def generate_hamiltonian(self):
        """
        Using the attributes of the ``self`` object,
        compute the cluster hamiltonian including the central spin.

        Returns:
            Hamiltonian: Cluster hamiltonian.

        """
        ham = total_hamiltonian(self.cluster, self.center, self.magnetic_field, others=self.others,
                                other_states=self.other_states)

        if self.pulses is not None:
            self.pulses.generate_pulses(dimensions=ham.dimensions, bath=self.cluster, vectors=ham.vectors,
                                        central_spin=self.center)

        return ham

    def compute_result(self):
        """
        Using the attributes of the ``self`` object,
        compute the coherence function of the central spin.

        Returns:

            ndarray: Computed coherence.

        """
        result = compute_dm(self.dm0, self.cluster_hamiltonian, self.timespace, self.pulses,
                            as_delay=self.as_delay, states=self.states, ncenters=self.center.size)
        if self.fulldm:
            result = result / self.zero_cluster
        else:
            result = (self.alpha.conj() @ result @ self.beta) / self.zero_cluster

        return result
