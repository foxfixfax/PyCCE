import numpy as np
import numpy.ma as ma
from pycce.bath.array import BathArray
from pycce.cluster_expansion import cluster_expansion_decorator
from pycce.hamiltonian import mf_hamiltonian

from .density_matrix import compute_dm

@cluster_expansion_decorator
def mean_field_density_matrix(cluster, allspin, dm0, alpha, beta, B, D, E,
                              timespace, pulse_sequence, bath_state,
                              gyro_e=-17608.597050,
                              as_delay=False, zeroth_cluster=None,
                              imap=None, map_error=None):
    """
    Decorated function to compute electron density matrix with gCCE with Monte-Carlo sampling of bath states
    @param subclusters: dict
        dict of subclusters included in different CCE order
        of structure {int order: np.array([[i,j],[i,j]])}
    @param allspin: BathArray
        array of all bath spins
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
    nspin = allspin[cluster]
    central_spin = (alpha.size - 1) / 2

    if imap is not None:
        imap = imap.subspace(cluster)

    others_mask = np.ones(allspin.shape, dtype=bool)
    others_mask[cluster] = False
    others = allspin[others_mask]
    others_state = bath_state[others_mask]

    states = bath_state[~others_mask]

    if zeroth_cluster is None:
        H, dimensions = mf_hamiltonian(BathArray(0), B, central_spin, others, others_state, D, E, gyro_e)
        zeroth_cluster = compute_dm(dm0, dimensions, H, alpha, beta, timespace, pulse_sequence,
                                    as_delay=as_delay)
        zeroth_cluster = ma.masked_array(zeroth_cluster, mask=(zeroth_cluster == 0))

    H, dimensions = mf_hamiltonian(nspin, B, central_spin, others, others_state, D, E, gyro_e,
                                   imap=imap, map_error=map_error)

    dms = compute_dm(dm0, dimensions, H, alpha, beta, timespace,
                     pulse_sequence, states=states, as_delay=as_delay) / zeroth_cluster
    return dms
