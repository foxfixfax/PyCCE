import numpy as np
from pycce.constants import PI2
from pycce.h import total_hamiltonian
from pycce.run.base import RunObject
from pycce.utilities import shorten_dimensions, generate_initial_state, outer


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

        eigexp = np.exp(-1j * np.outer(timespace, evalues),
                        dtype=np.complex128)

        u = np.matmul(np.einsum('ij,kj->kij', evec, eigexp, dtype=np.complex128),
                      evec.conj().T)

        return u

    else:

        if pulses.delays is None:
            if not as_delay:
                n = len(pulses)
                timespace = timespace / (2 * n)

            eigexp = np.exp(-1j * np.outer(timespace, evalues),
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

            eigexp = np.exp(-1j * np.outer(timesteps, evalues),
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
            eigexp = np.exp(-1j * np.outer(timespace - times, evalues),
                            dtype=np.complex128)

            u = np.matmul(np.einsum('ij,kj->kij', evec, eigexp, dtype=np.complex128),
                          evec.conj().T)

            U = np.matmul(u, U)
        elif not which.all():
            raise ValueError(f"Pulse sequence time steps add up to larger than total times. Delays at"
                             f"{timespace[(timespace - times) < 0]} ms are longer than total time.")
    return U


def compute_dm(initial_state, H, timespace, pulse_sequence=None, as_delay=False, states=None, ncenters=1):
    """
    Function to compute density matrix of the central spin, given Hamiltonian H.

    Args:
        initial_state (ndarray): Initial density matrix of central spin.

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
    center_shape = initial_state.shape
    dimensions = shorten_dimensions(H.dimensions, ncenters)

    initial_state = generate_initial_state(dimensions, states=states, central_state=initial_state)
    dm = full_dm(initial_state, H, timespace, pulse_sequence=pulse_sequence, as_delay=as_delay)
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
    if dm0.ndim > 1:
        dmUdagger = np.matmul(dm0, U.conj().transpose(0, 2, 1))
        dm = np.matmul(U, dmUdagger)
    else:
        # |dm> = U|dm>
        dm = U @ dm0
        # |dm><dm|
        dm = np.einsum('ki,kj->kij', dm, dm.conj())

    return dm


def _get_state(state, center):
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

        as_delay (bool):
            True if time points are delay between pulses, False if time points are total time.

        fulldm (bool):
            True if return full density matrix. Default False.

        **kwargs: Keyword arguments of the ``RunObject``.

    """

    def __init__(self, *args, i=None, j=None, as_delay=False, pulses=None, fulldm=False, normalized=True, **kwargs):
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

        check = True  # False
        # if self.projected_bath_state is not None:
        #     try:
        #         check = all(self.projected_bath_state == self.bath_state)
        #     except TypeError:
        #         check = False

        if check:
            self.dm0 = self.center.state
            self.normalization = outer(self.center.state, self.center.state)
        else:
            self.dm0 = outer(self.center.state, self.center.state)
            self.normalization = self.dm0

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
            self.pulses.generate_pulses(dimensions=self.center.hamiltonian.dimensions,
                                        bath=None, vectors=self.center.hamiltonian.vectors,
                                        central_spin=self.center)
            for p in self.pulses:
                if p.rotation is not None:
                    if self.i is None:
                        alpha = p.rotation @ alpha
                    if self.j is None:
                        beta = p.rotation @ beta

                    self.normalization = p.rotation @ self.normalization @ p.rotation.T.conj()

        self.alpha = alpha
        self.beta = beta

        res = full_dm(self.dm0, self.center.hamiltonian, self.timespace,
                      pulse_sequence=self.pulses, as_delay=self.as_delay)

        if not self.fulldm:
            res = self.alpha.conj() @ res @ self.beta
            self.normalization = (self.alpha.conj() @ self.normalization @ self.beta)
        # else:
        #     res = self.center.eigenvectors.conj().T @ res @ self.center.eigenvectors

        self.zero_cluster = res

    def postprocess(self):
        self.result = self.zero_cluster * self.result

        if self.normalized:
            self.result = self.result / self.normalization

        # else:
        #     self.result = self.center.eigenvectors @ self.result @ self.center.eigenvectors.conj().T

    def generate_hamiltonian(self):
        """
        Using the attributes of the ``self`` object,
        compute the cluster hamiltonian including the central spin.

        Returns:
            Hamiltonian: Cluster hamiltonian.

        """
        ham = total_hamiltonian(self.cluster, self.center, self.magnetic_field, others=self.others)

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

        if not self.fulldm:
            result = (self.alpha.conj() @ result @ self.beta)

        # else self.fulldm:
        #     result = self.center.eigenvectors.conj().T @ result @ self.center.eigenvectors

        return result / self.zero_cluster
