import numpy as np
import numpy.ma as ma
import scipy.linalg

from .bath.array import BathArray
from .cluster_expansion import cluster_expansion_decorator
from .density_matrix import compute_dm, full_dm, generate_dm0
from .hamiltonian import mf_hamiltonian, expand

hbar = 1.05457172  # When everything else in rad, kHz, ms, G, A


def compute_mf_dm(dm0, dimensions, states, H, S, timespace,
                  pulse_sequence=None, as_delay=False):
    """
    @param dm0: ndarray
        Initial density matrix of the central spin
    @param dimensions: array_like
        A list of spins dimensions. Last entry - electron spin dimensions
    @param states: array_like
    @param H: ndarray
        Cluster Hamiltonian
    @param S: QSpinMatrix
        QSpinMatrix of the central spin
    @param timespace: ndarray
        Time delay values at which to compute density matrix
    @param pulse_sequence: list
    pulse_sequence should have format of list with tuples,
       each tuple contains two entries:
       first: axis the rotation is about;
       second: angle of rotation. E.g. for Hahn-Echo [('x', np.pi/2)]
    @param as_delay: bool
    @return: ndarray
        array of the density matrices of the central spin for the given cluster
    """
    if not as_delay and pulse_sequence:
        N = len(pulse_sequence)
        timespace = timespace / (2 * N)

    dmtotal0 = generate_dm0(dm0, dimensions, states)
    dm = full_dm(dmtotal0, dimensions, H, S, timespace, pulse_sequence=pulse_sequence)

    initial_shape = dm.shape
    dm.shape = (initial_shape[0], *dimensions, *dimensions)
    for d in range(len(dimensions) + 1, 2, -1):  # The last one is el spin
        dm = np.trace(dm, axis1=1, axis2=d)
    return dm


@cluster_expansion_decorator
def mean_field_density_matrix(nspin, dm0, I, S, B, D, E,
                              timespace, pulse_sequence, allspins, bath_state,
                              gyro_e=-17608.597050,
                              as_delay=False, zeroth_cluster=None):
    """
    Decorated function to compute electron density matrix with gCCE with Monte-Carlo sampling of bath states
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
    @param allspins: ndarray
        array of all atoms. Passed twice because one is passed to decorator, another - directly to function
    @param bath_state: list
        List of nuclear spin states, contains Sz projections of nuclear spins.
    @param gyro_e: float
        gyromagnetic ratio (in rad/(msec*Gauss)) of the central spin
    @param as_delay: bool
        True if time points are delay between pulses,
        False if time points are total time
    @param zeroth_cluster: ndarray
        density matrix of isolated central spin at all time poins.
        Shape (n x m x m) where n = len(time_space) and m is central spin dimensions
    @return: dms
        array of central spin dm for each time point
    """
    others_mask = np.isin(allspins, nspin)

    others = allspins[~others_mask]
    others_state = bath_state[~others_mask]

    states = bath_state[others_mask]

    if zeroth_cluster is None:
        H, dimensions = mf_hamiltonian(BathArray(0), I, S, B, others, others_state, D, E, gyro_e)
        zeroth_cluster = compute_dm(dm0, dimensions, H, S, timespace, pulse_sequence,
                                    as_delay=as_delay)
        zeroth_cluster = ma.masked_array(zeroth_cluster, mask=(zeroth_cluster == 0))

    H, dimensions = mf_hamiltonian(nspin, I, S, B, others, others_state, D, E, gyro_e)
    dms = compute_mf_dm(dm0, dimensions, states, H, S, timespace,
                        pulse_sequence, as_delay=as_delay) / zeroth_cluster
    return dms
