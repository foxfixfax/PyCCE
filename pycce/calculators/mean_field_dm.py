import numpy as np
import numpy.ma as ma

from pycce.bath.array import BathArray
from pycce.cluster_expansion import cluster_expansion_decorator
from pycce.hamiltonian import mean_field_hamiltonian
from pycce.units import ELECTRON_GYRO
from .density_matrix import compute_dm
from pycce.sm import _smc


@cluster_expansion_decorator
def mean_field_density_matrix(cluster, allspin, dm0, alpha, beta, magnetic_field, zfs, timespace, pulse_sequence,
                              bath_state, gyro_e=ELECTRON_GYRO, as_delay=False, zeroth_cluster=None, imap=None,
                              map_error=None, projected_bath_state=None):
    """
    Decorated function to compute electron density matrix with gCCE with Monte-Carlo sampling of bath states
    :param subclusters: dict
        dict of subclusters included in different CCE order
        of structure {int order: np.array([[i,j],[i,j]])}
    :param allspin: BathArray
        array of all bath spins
    :param ntype: dict
        dict with NSpinType objects inside, each key - name of the isotope
    :param dm0: ndarray
        initial density matrix of the central spin
    :param I: dict
        dict with SpinMatrix objects inside, each key - spin
    :param S: QSpinMatrix
        QSpinMatrix of the central spin
    :param magnetic_field: ndarray
        Magnetic field of mfield = np.array([Bx, By, Bz])
    :param zfs: float
        D parameter in central spin ZFS
    :param timespace: ndarray
        Time points at which to compute density matrix
    :param pulse_sequence: list
        pulse_sequence should have format of list with tuples,
        each tuple contains two entries: first: axis the rotation is about; second: angle of rotation.
        E.g. for Hahn-Echo [('x', np.pi/2)]. For now only pulses with same delay are supported
    :param bath_state: list
        List of nuclear spin states, contains Sz projections of nuclear spins.
    :param gyro_e: float
        gyromagnetic ratio (in rad/(msec*Gauss)) of the central spin
    :param as_delay: bool
        True if time points are delay between pulses,
        False if time points are total time
    :param zeroth_cluster: ndarray
        density matrix of isolated central spin at all time poins.
        Shape (n x m x m) where n = len(time_space) and m is central spin dimensions
    :return: dms
        array of central spin dm for each time point
    """
    nspin = allspin[cluster]
    central_spin = (alpha.size - 1) / 2

    if imap is not None:
        imap = imap.subspace(cluster)

    others_mask = np.ones(allspin.shape, dtype=bool)
    others_mask[cluster] = False
    others = allspin[others_mask]

    states = bath_state[~others_mask]

    if projected_bath_state is None:
        projected_bath_state = bath_state

    others_state = projected_bath_state[others_mask]

    if zeroth_cluster is None:
        H = mean_field_hamiltonian(BathArray(0), magnetic_field, allspin, projected_bath_state, zfs,
                                   central_spin=central_spin, central_gyro=gyro_e)
        zeroth_cluster = compute_dm(dm0, H, alpha, beta, timespace, pulse_sequence, as_delay=as_delay)
        zeroth_cluster = ma.masked_array(zeroth_cluster, mask=(zeroth_cluster == 0))

    H = mean_field_hamiltonian(nspin, magnetic_field, others, others_state, zfs,
                               central_spin=central_spin, central_gyro=gyro_e,
                               imap=imap, map_error=map_error)

    dms = compute_dm(dm0, H, alpha, beta, timespace, pulse_sequence, as_delay=as_delay, states=states) / zeroth_cluster
    return dms


def generate_bath_state(bath, nbstates, seed=None):
    rgen = np.random.default_rng(seed)

    for _ in range(nbstates):
        bath_state = np.empty(bath.shape, dtype=np.float64)
        for n in bath.types:
            s = bath.types[n].s
            snumber = int(round(2 * s + 1))
            mask = bath['N'] == n
            bath_state[mask] = rgen.integers(snumber, size=np.count_nonzero(mask)) - s

        yield bath_state


def monte_carlo_sampling(clusters, bath, dm0, alpha, beta, magnetic_field, zfs, timespace, pulse_sequence,
                         central_gyro=ELECTRON_GYRO, as_delay=False, imap=None,
                         nbstates=100, seed=None, masked=True,
                         normalized=None, parallel_states=False,
                         fixstates=None, direct=False, parallel=False):
    """
    Compute density matrix of the central spin using generalized CCE with Monte-Carlo bath state sampling
    :param timespace: 1D-ndarray
        time points at which compute density matrix
    :param magnetic_field: ndarray
        magnetic field as (Bx, By, Bz)
    :param D: float or ndarray with shape (3,3)
        D (longitudinal splitting) parameter of central spin in ZFS tensor of central spin in rad * kHz
        OR total ZFS tensor
    :param E: float
        E (transverse splitting) parameter of central spin in ZFS tensor of central spin in rad * kHz
    :param N: int
        number of pulses in CPMG sequence. Overrides pulse_sequence if provided
    :param pulse_sequence: list
        pulse_sequence should have format of list with tuples,
        each tuple contains two entries: first: axis the rotation is about; second: angle of rotation.
        E.g. for Hahn-Echo [('x', np.pi/2)]. For now only pulses with same delay are supported
    :param as_delay: bool
        True if time points are delay between pulses,
        False if time points are total time
    :param state: ndarray
        Initial state of the central spin. Defaults to sqrt(1 / 2) * (state + beta) if not set
    :param nbstates: int
        Number of random bath states to sample
    :param seed: int
        Seed for the RNG
    :param masked: bool
        True if mask numerically unstable points (with density matrix elements > 1)
        in the averaging over bath states
        False if not. Default True
    :param normalized: ndarray of bools
        which diagonal elements to renormalize, so the total sum of the diagonal elements is 1
    :param parallel_states: bool
        whether to use MPI to parallelize the calculations of density matrix
        for each random bath state
    :param fixstates: dict
        dict of which bath states to fix. Each key is the index of bath spin,
        value - fixed Sz projection of the mixed state of nuclear spin
    :return: dms
    """
    central_spin = (alpha.size - 1) / 2

    if parallel_states:
        try:
            from mpi4py import MPI
        except ImportError:
            print('Parallel failed: mpi4py is not found. Running serial')
            parallel_states = False

    if masked:
        divider = np.zeros(timespace.shape, dtype=np.int32)
    else:
        root_divider = nbstates

    if parallel_states:
        comm = MPI.COMM_WORLD

        size = comm.Get_size()
        rank = comm.Get_rank()

        remainder = nbstates % size
        add = int(rank < remainder)
        nbstates = nbstates // size + add

        if seed:
            seed = seed + rank
    else:
        rank = 0

    averaged_dms = ma.zeros((timespace.size, *dm0.shape), dtype=np.complex128)

    for bath_state in generate_bath_state(bath, nbstates, seed=seed):

        if fixstates is not None:
            for fs in fixstates:
                bath_state[fs] = fixstates[fs]

        H0 = mean_field_hamiltonian(BathArray(0), magnetic_field, bath, bath_state, zfs,
                                    central_spin=central_spin, central_gyro=central_gyro)

        dmzero = compute_dm(dm0, H0, alpha, beta, timespace, pulse_sequence, as_delay=as_delay)
        dmzero = ma.array(dmzero, mask=(dmzero == 0), fill_value=0j, dtype=np.complex128)

        dms = mean_field_density_matrix(clusters, bath, dm0, alpha, beta, magnetic_field,
                                        zfs, timespace, pulse_sequence, bath_state, as_delay=as_delay,
                                        zeroth_cluster=dmzero, imap=imap,
                                        direct=direct, parallel=parallel) * dmzero
        if masked:
            dms = dms.filled()
            proper = np.all(np.abs(dms) <= 1, axis=(1, 2))
            divider += proper.astype(np.int32)
            dms[~proper] = 0.

        if normalized is not None:
            norm = np.asarray(normalized)
            ind = np.arange(dms.shape[1])
            diagonals = dms[:, ind, ind]

            sums = np.sum(diagonals[:, norm], keepdims=True, axis=-1)
            sums[sums == 0.] = 1

            expsum = 1 - np.sum(diagonals[:, ~norm], keepdims=True, axis=-1)

            diagonals[:, norm] = diagonals[:, norm] / sums * expsum
            dms[:, ind, ind] = diagonals

        averaged_dms += dms

    if parallel_states:
        root_dms = ma.array(np.zeros(averaged_dms.shape), dtype=np.complex128)
        comm.Reduce(averaged_dms, root_dms, MPI.SUM, root=0)
        if masked:
            root_divider = np.zeros(divider.shape, dtype=np.int32)
            comm.Reduce(divider, root_divider, MPI.SUM, root=0)

    else:
        root_dms = averaged_dms
        if masked:
            root_divider = divider

    if rank == 0:
        root_dms = ma.array(root_dms, fill_value=0j, dtype=np.complex128)

        if masked:
            root_dms[root_divider == 0] = ma.masked
            root_divider = root_divider[:, np.newaxis, np.newaxis]
        root_dms /= root_divider

        return root_dms
    else:
        return
