import numpy as np
import numpy.ma as ma
import scipy.linalg

from .bath.array import BathArray
from .cluster_expansion import cluster_expansion_decorator
from .hamiltonian import total_hamiltonian, expand, eta_hamiltonian
hbar = 1.05457172  # When everything else in rad, kHz, ms, G, A

def propagator_dm(timespace, H, pulse_sequence, S, dimensions):
    """
    Function to compute bath propagator U in gCCE

    @param timespace: ndarray
        Time delay values at which to compute propagators
    @param H: ndarray
        Cluster Hamiltonian
    @param pulse_sequence: list
    pulse_sequence should have format of list with tuples,
       each tuple contains two entries:
       first: axis the rotation is about;
       second: angle of rotation. E.g. for Hahn-Echo [('x', np.pi/2)]
    @param S: QSpinMatrix
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
        alpha_x_alpha = np.tensordot(S.alpha, S.alpha, axes=0)
        beta_x_beta = np.tensordot(S.beta, S.beta, axes=0)
        alpha_x_beta = np.tensordot(S.alpha, S.beta, axes=0)
        beta_x_alpha = np.tensordot(S.beta, S.alpha, axes=0)

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


def compute_dm(dm0, dimensions, H, S, timespace,
               pulse_sequence=None, as_delay=False, states=None):
    """
    Function to compute density matrix of the central spin S, given Hamiltonian H
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
    dm = full_dm(dm0, dimensions, H, S, timespace, pulse_sequence=pulse_sequence)

    initial_shape = dm.shape
    dm.shape = (initial_shape[0], *dimensions, *dimensions)
    for d in range(len(dimensions) + 1, 2, -1):  # The last one is el spin
        dm = np.trace(dm, axis1=1, axis2=d)
    return dm


def full_dm(dm0, dimensions, H, S, timespace, pulse_sequence=None):
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
    # if timespace was given not as delay between pulses,
    # divide to obtain the delay

    U = propagator_dm(timespace, H, pulse_sequence, S, dimensions)
    dmUdagger = np.matmul(dm0, np.transpose(U.conj(), axes=(0, 2, 1)))
    dm = np.matmul(U, dmUdagger)

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

    dmtotal0 = np.eye(1, dtype=np.complex128)
    for s, d in zip(states, dimensions[:-1]):
        s = np.asarray(s)
        dm_nucleus = np.zeros((d, d), dtype=np.complex128)
        if len(s.shape) == 0:
            state_number = int(round((d - 1) / 2 - s))
            dm_nucleus[state_number, state_number] = 1
        else:
            np.fill_diagonal(dm_nucleus, s)

        dmtotal0 = np.kron(dmtotal0, dm_nucleus)

    dmtotal0 = np.kron(dmtotal0, dm0)
    return dmtotal0


def cluster_dm_direct_approach(subclusters, nspin,
                               dm0, I, S, B, gyro_e, D, E,
                               timespace, pulse_sequence, as_delay=False):
    """
    Direct function to compute electron density matrix with gCCE (without mean field)
    @param subclusters: dict
        dict of subclusters included in different CCE order
        of structure {int order: np.array([[i,j],[i,j]])}
    @param nspin: ndarray
        array of all atoms
    @param ntype: dict
        dict with NSpinType objects inside, each key - name of the isotope
    @param dm0: ndarray
        initial density matrix of the central spin
    @param I: dict
        dict with SpinMatrix objects inside, each key - spin
    @param S: QSpinMatrix
        QSpinMatrix of the central spin
    @param B: ndarray
        Magnetic field of B = np.array([Bx, By, Bz])
    @param gyro_e: float
        gyromagnetic ratio (in rad/(msec*Gauss)) of the central spin
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
    @param as_delay: bool
        True if time points are delay between pulses,
        False if time points are total time
    @return: dms
        array of central spin dm for each time point
    """
    orders = sorted(subclusters)
    norders = len(orders)

    # Data for zero cluster
    H, dimensions = total_hamiltonian(BathArray(0), I, S, B, D, E=E, gyro_e=gyro_e)
    dms_zero = compute_dm(dm0, dimensions, H, S, timespace, pulse_sequence,
                          as_delay=as_delay)
    dms_zero = ma.masked_array(dms_zero, mask=(dms_zero == 0))
    # print(dms_zero.mask)
    # If there is only one set of indexes for only one order,
    # Then for this subcluster nelements < maximum CCE order
    if norders == 1 and subclusters[orders[0]].shape[0] == 1:
        verticles = subclusters[orders[0]][0]

        H, dimensions = total_hamiltonian(nspin[verticles], I, S, B, D, E=E, gyro_e=gyro_e)
        dms = compute_dm(dm0, dimensions, H, S, timespace,
                         pulse_sequence, as_delay=as_delay) / dms_zero

        return dms

        # print(zero_power)
    # The Highest possible L will have all powers of 1
    dm_tilda = {}
    visited = 0
    dms = np.ones([*timespace.shape, *dm0.shape], dtype=np.complex128)
    dms = ma.masked_array(dms, mask=(dms_zero == 0))

    for order in orders:
        dm_tilda[order] = []
        # indexes of the cluster of size order are stored in v

        for index in range(subclusters[order].shape[0]):

            v = subclusters[order][index]

            H, dimensions = total_hamiltonian(nspin[v], I, S, B, D, E=E, gyro_e=gyro_e)
            dms_v = (compute_dm(dm0, dimensions, H, S, timespace, pulse_sequence,
                                as_delay=as_delay) / dms_zero)

            for lowerorder in orders[:visited]:
                contained_in_v = np.all(np.isin(subclusters[lowerorder], v), axis=1)
                lower_dmtilda = np.prod(dm_tilda[lowerorder][contained_in_v], axis=0)
                dms_v /= lower_dmtilda

            dms *= dms_v
            dm_tilda[order].append(dms_v)

        dm_tilda[order] = np.array(dm_tilda[order], copy=False)

        visited += 1

        print('Computed density matrices of order {} for {} clusters'.format(
            order, subclusters[order].shape[0]))

    return dms


@cluster_expansion_decorator
def decorated_density_matrix(nspin, dm0, I, S, B, D, E,
                             timespace, pulse_sequence,
                             gyro_e=-17608.597050,
                             as_delay=False, zeroth_cluster=None,
                             bath_state=None, allspins=None,
                             eta=None):
    """
    Decorated function to compute electron density matrix with gCCE (without mean field)
    @param subclusters: dict
        dict of subclusters included in different CCE order
        of structure {int order: np.array([[i,j],[i,j]])}
    @param allnspin: ndarray
        array of all atoms
    @param ntype: dict
        dict with NSpinType objects inside, each key - name of the isotope
    @param dm0: ndarray
        initial density matrix of the central spin
    @param I: dict
        dict with SpinMatrix objects inside, each key - spin
    @param S: QSpinMatrix
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
        array of all atoms. Passed twice because one is passed to decorator, another - directly to function
    @param eta: float
        value of eta (see eta_hamiltonian)
    @return: dms
        array of central spin dm for each time point
    """
    if allspins is not None and bath_state is not None:
        others_mask = np.isin(allspins, nspin)
        states = bath_state[others_mask]
    else:
        states = None

    if zeroth_cluster is None:
        H, dimensions = total_hamiltonian(BathArray(0), I, S, B, D, E=E, gyro_e=gyro_e)
        zeroth_cluster = compute_dm(dm0, dimensions, H, S, timespace, pulse_sequence,
                                    as_delay=as_delay)
        zeroth_cluster = ma.masked_array(zeroth_cluster, mask=(zeroth_cluster == 0))

    H, dimensions = total_hamiltonian(nspin, I, S, B, D, E=E, gyro_e=gyro_e)
    if eta is not None:
        H += eta_hamiltonian(nspin, I, S, eta)
    dms = compute_dm(dm0, dimensions, H, S, timespace,
                     pulse_sequence, as_delay=as_delay, states=states) / zeroth_cluster

    return dms
