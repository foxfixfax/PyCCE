import numpy as np
import numpy.ma as ma
import scipy.linalg

from pycce.bath.array import BathArray
from pycce.cluster_expansion import cluster_expansion_decorator
from pycce.hamiltonian import total_hamiltonian, expand, eta_hamiltonian

hbar = 1.05457172  # When everything else in rad, kHz, ms, G, A


def propagator_dm(timespace, H, pulse_sequence, alpha, beta, dimensions):
    """
    Function to compute bath propagator U in gCCE

    @param beta:
    @param timespace: ndarray
        Time delay values at which to compute propagators
    @param H: ndarray
        Cluster Hamiltonian
    @param pulse_sequence: list
    pulse_sequence should have format of list with tuples,
       each tuple contains two entries:
       first: axis the rotation is about;
       second: angle of rotation. E.g. for Hahn-Echo [('x', np.pi/2)]
    @param alpha: QSpinMatrix
        QSpinMatrix of the central spin
    @param dimensions: list
        list of nuclear spin dimensions. Last entry - electron spin dimensions
    @return: U
    """
    evalues, evec = np.linalg.eigh(H)

    eigexp = np.exp(-1j * np.tensordot(timespace, evalues, axes=0),
                    dtype=np.complex128)

    u = np.matmul(np.einsum('ij,kj->kij', evec, eigexp, dtype=np.complex128),
                  evec.conj().T)

    if not pulse_sequence:
        return u

    else:
        alpha_x_alpha = np.tensordot(alpha, alpha, axes=0)
        beta_x_beta = np.tensordot(beta, beta, axes=0)
        alpha_x_beta = np.tensordot(alpha, beta, axes=0)
        beta_x_alpha = np.tensordot(beta, alpha, axes=0)

        sigmax = alpha_x_beta + beta_x_alpha
        sigmay = -1j * alpha_x_beta + 1j * beta_x_alpha
        sigmaz = alpha_x_alpha - beta_x_beta

        sigma = {'x': expand(sigmax, len(dimensions) - 1, dimensions),
                 'y': expand(sigmay, len(dimensions) - 1, dimensions),
                 'z': expand(sigmaz, len(dimensions) - 1, dimensions)}

        U = np.eye(u.shape[1])
        for pulse in pulse_sequence:
            angle = pulse[1]
            ax = pulse[0]
            rotation = scipy.linalg.expm(-1j * sigma[ax] * angle / 2)
            U = np.matmul(u, U)
            U = np.matmul(rotation, U)
            U = np.matmul(u, U)

        return U


def compute_dm(dm0, dimensions, H, alpha, beta, timespace,
               pulse_sequence=None, as_delay=False, states=None):
    """
    Function to compute density matrix of the central spin spin_matrix, given Hamiltonian H
    @param dm0: ndarray
        initial density matrix of nuclear spin
    @param dimensions: list
        list of nuclear spin dimensions. Last entry - electron spin dimensions
    @param H: ndarray
        Cluster Hamiltonian
    @param S: QSpinMatrix
        QSpinMatrix of the central spin
    @param timespace: ndarray
        Time delay values at which to compute propagators
    @param pulse_sequence: list
        pulse_sequence should have format of list with tuples,
        each tuple contains two entries: first: axis the rotation is about; second: angle of rotation.
        E.g. for Hahn-Echo [('x', np.pi/2)]. For now only pulses with same delay are supported
    @param as_delay: bool
        True if time points are delay between pulses,
        False if time points are total time
    @param states: array_like
        List of nuclear spin states. if len(shape) == 1, contains Sz projections of nuclear spins.
        Otherwise, contains array of initial dms of nuclear spins
    @return: dm
        density matrix of the electron
    """
    if not as_delay and pulse_sequence:
        N = len(pulse_sequence)
        timespace = timespace / (2 * N)

    if states is None:
        dm0 = expand(dm0, len(dimensions) - 1, dimensions) / np.prod(dimensions[:-1])
    else:
        dm0 = generate_dm0(dm0, dimensions, states)
    dm = full_dm(dm0, dimensions, H, alpha, beta, timespace, pulse_sequence=pulse_sequence)

    initial_shape = dm.shape
    dm.shape = (initial_shape[0], *dimensions, *dimensions)
    for d in range(len(dimensions) + 1, 2, -1):  # The last one is el spin
        dm = np.trace(dm, axis1=1, axis2=d)
    return dm


def full_dm(dm0, dimensions, H, alpha, beta, timespace, pulse_sequence=None):
    """
     A function to compute dm using hamiltonian H from expanded dm0
    @param dm0: ndarray
        initial density matrix of nuclear spin
    @param dimensions: list
        list of nuclear spin dimensions. Last entry - electron spin dimensions
    @param H: ndarray
        Cluster Hamiltonian
    @param S: QSpinMatrix
        QSpinMatrix of the central spin
    @param timespace: ndarray
        Time delay values at which to compute propagators
    @param pulse_sequence: list
        pulse_sequence should have format of list with tuples,
        each tuple contains two entries: first: axis the rotation is about; second: angle of rotation.
        E.g. for Hahn-Echo [('x', np.pi/2)]. For now only even pulses are supported
    @return: dm
        density matrix of the cluster
    """

    U = propagator_dm(timespace, H, pulse_sequence, alpha, beta, dimensions)

    dmUdagger = np.matmul(dm0, np.transpose(U.conj(), axes=(0, 2, 1)))
    dm = np.matmul(U, dmUdagger)
    # einsum does the same as the following
    # dm = np.einsum('zli,ij,zkj->zlk', U, dm0, U.conj())
    return dm


def generate_dm0(dm0, dimensions, states):
    """
    A function to generate initial density matrix of the cluster
    @param dm0: ndarray
        initial dm of the central spin
    @param dimensions: list
        list of nuclear spin dimensions. Last entry - electron spin dimensions
    @param states: array_like
        List of nuclear spin states. if len(shape) == 1, contains Sz projections of nuclear spins.
        Otherwise, contains array of initial dms of nuclear spins
    @return: dm
        initial density matrix of the cluster
    """
    dmtotal0 = gen_density_matrix(states, dimensions[:-1])
    dmtotal0 = np.kron(dmtotal0, dm0)
    return dmtotal0


def gen_density_matrix(states, dimensions=None):
    dmtotal0 = np.eye(1, dtype=np.complex128)
    for i, s in enumerate(states):
        if not hasattr(s, "__len__"):
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


@cluster_expansion_decorator
def decorated_density_matrix(cluster, allspin, dm0, alpha, beta, B, D, E,
                             timespace, pulse_sequence,
                             gyro_e=-17608.597050,
                             as_delay=False, zeroth_cluster=None,
                             bath_state=None, eta=None):
    """
    Decorated function to compute electron density matrix with gCCE (without mean field)
    @param subclusters: dict
        dict of subclusters included in different CCE order
        of structure {int order: np.array([[i,j],[i,j]])}
    @param allnspin: ndarray
        array of all bath
    @param ntype: dict
        dict with NSpinType objects inside, each key - name of the isotope
    @param dm0: ndarray
        initial density matrix of the central spin
    @param alpha: ndarray
        dict with SpinMatrix objects inside, each key - spin
    @param beta: ndarray
        QSpinMatrix of the central spin
    @param B: ndarray
        Magnetic field of B = np.array([Bx, By, Bz])
    @param D: float
        D parameter in central spin ZFS
    @param E: float
        E parameter in central spin ZFS
    @param timespace: ndarray
        Time points at which to compute density matrix
    @param pulse_sequence: list
        pulse_sequence should have format of list with tuples,
        each tuple contains two entries: first: axis the rotation is about; second: angle of rotation.
        E.g. for Hahn-Echo [('x', np.pi/2)]. For now only pulses with same delay are supported
    @param gyro_e: float
        gyromagnetic ratio (in rad/(msec*Gauss)) of the central spin
    @param as_delay: bool
        True if time points are delay between pulses,
        False if time points are total time
    @param zeroth_cluster: ndarray
        density matrix of isolated central spin at all time poins.
        Shape (n x m x m) where n = len(time_space) and m is central spin dimensions
    @param bath_state: list or None
        List of nuclear spin states. if len(shape) == 1, contains Sz projections of nuclear spins.
        Otherwise, contains array of initial dms of nuclear spins
    @param allspins: ndarray
        array of all bath. Passed twice because one is passed to decorator, another - directly to function
    @param eta: float
        value of eta (see eta_hamiltonian)
    @return: dms
        array of central spin dm for each time point
    """
    nspin = allspin[cluster]
    central_spin = (alpha.size - 1) / 2
    if bath_state is not None:
        others_mask = np.ones(allspin.shape, dtype=bool)
        others_mask[cluster] = False
        states = bath_state[others_mask]
    else:
        states = None

    if zeroth_cluster is None:
        H, dimensions = total_hamiltonian(BathArray(0), central_spin, B, D, E=E, central_gyro=gyro_e)
        zeroth_cluster = compute_dm(dm0, dimensions, H, alpha, beta, timespace, pulse_sequence,
                                    as_delay=as_delay)
        zeroth_cluster = ma.masked_array(zeroth_cluster, mask=(zeroth_cluster == 0))

    H, dimensions = total_hamiltonian(nspin, central_spin, B, D, E=E, central_gyro=gyro_e)
    if eta is not None:
        H += eta_hamiltonian(nspin, alpha, beta, eta)
    dms = compute_dm(dm0, dimensions, H, alpha, beta, timespace,
                     pulse_sequence, as_delay=as_delay, states=states) / zeroth_cluster

    return dms
