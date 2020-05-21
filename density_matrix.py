import numpy as np
import numpy.ma as ma
import scipy.linalg
from .hamiltonian import total_elhamiltonian, expand
from .cluster_expansion import cluster_expansion_decorator

hbar = 1.05457172  # When everything else in rad, kHz, ms, G, A


def propagator_dm(timespace, H, pulse_sequence, S, dimensions):
    """pulse_sequence should have format of list with tuples,
       containing two entries:
       first: axis the rotation is about; 
       second: angle of rotation"""
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
        U = u
        for pulse in pulse_sequence:
            angle = pulse[1]
            ax = pulse[0]
            rotation = scipy.linalg.expm(-1j * sigma[ax] * angle / 2)
            U = np.matmul(U, rotation)
            U = np.matmul(U, u)

        return U


def compute_dm(dm0, dimensions, H, S, timespace,
               pulse_sequence=None, as_delay=False):
    if not as_delay and pulse_sequence:
        N = len(pulse_sequence)
        timespace = timespace / (2 * N)

    dm0 = expand(dm0, len(dimensions) - 1, dimensions) / np.prod(dimensions[:-1])

    dm = full_dm(dm0, dimensions, H, S, timespace, pulse_sequence=pulse_sequence)

    initial_shape = dm.shape
    dm.shape = (initial_shape[0], *dimensions, *dimensions)
    for d in range(len(dimensions) + 1, 2, -1):  # The last one is el spin
        dm = np.trace(dm, axis1=1, axis2=d)
    return dm


def full_dm(dm0, dimensions, H, S, timespace, pulse_sequence=None):
    # if timespace was given not as delay between pulses,
    # divide to obtain the delay

    U = propagator_dm(timespace, H, pulse_sequence, S, dimensions)
    dmUdagger = np.matmul(dm0, np.transpose(U.conj(), axes=(0, 2, 1)))
    dm = np.matmul(U, dmUdagger)

    return dm


def cluster_dm(subclusters, nspin, ntype,
               dm0, I, S, B, gyro_e, D, E,
               timespace, pulse_sequence, as_delay=False):
    # List of orders from highest to lowest
    revorders = sorted(subclusters)[::-1]
    norders = len(revorders)

    # Data for zero cluster
    H, dimensions = total_elhamiltonian(np.array([]), ntype,
                                        I, B, S, gyro_e, D, E)
    dms_zero = compute_dm(dm0, dimensions, H, S, timespace, pulse_sequence,
                          as_delay=as_delay)
    dms_zero = ma.masked_array(dms_zero, mask=(dms_zero == 0))
    # print(dms_zero.mask)
    # If there is only one set of indexes for only one order,
    # Then for this subcluster nelements < maximum CCE order
    if norders == 1 and subclusters[revorders[0]].shape[0] == 1:
        verticles = subclusters[revorders[0]][0]

        H, dimensions = total_elhamiltonian(nspin[verticles], ntype,
                                            I, B, S, gyro_e, D, E)
        dms = compute_dm(dm0, dimensions, H, S, timespace,
                         pulse_sequence, as_delay=as_delay) / dms_zero

        return dms

    # print(zero_power)
    # The Highest possible L will have all powers of 1
    power = {}
    zero_power = 0
    # Number of visited orders from highest to lowest
    visited = 0
    dms = np.ones([*timespace.shape, *dm0.shape], dtype=np.complex128)
    dms = ma.masked_array(dms, mask=(dms_zero == 0))
    for order in revorders:
        power[order] = np.ones(subclusters[order].shape[0], dtype=np.int32)
        # indexes of the cluster of size order are stored in v

        for index in range(subclusters[order].shape[0]):

            v = subclusters[order][index]
            # First, find the correct power. Iterate over all higher orders
            for higherorder in revorders[:visited]:
                # np.isin gives bool array of shape subclusters[higherorder],
                # which is np.array of
                # indexes of subclusters with order = higherorder.
                # Entries are True if value is
                # present in v and False if values are not present in v.
                # Sum bool entries in inside cluster,
                # if the sum equal to size of v,
                # then v is inside the given subcluster.
                # containv is 1D bool array with values of i-element True
                # if i-subcluster of
                # subclusters[higherorder] contains v
                containv = np.count_nonzero(
                    np.isin(subclusters[higherorder], v), axis=1) == v.size

                # Power of cluster v is decreased by sum of powers of all the higher orders,
                # As all of them have to be divided by v
                power[order][index] -= np.sum(power[higherorder]
                                              [containv], dtype=np.int32)

            H, dimensions = total_elhamiltonian(nspin[v], ntype,
                                                I, B, S, gyro_e, D, E)
            dms_v = (compute_dm(dm0, dimensions, H, S, timespace, pulse_sequence,
                                as_delay=as_delay) / dms_zero) ** power[order][index]
            dms *= dms_v

            zero_power -= power[order][index]
        # print(np.abs(power[order]).max())
        # print(zero_power)
        visited += 1
        print('Computed density matrices of order {} for {} clusters in subcluster of size {}'.format(
            order, subclusters[order].shape[0], subclusters[1].size))

    return dms


def cluster_dm_direct_approach(subclusters, nspin, ntype,
                               dm0, I, S, B, gyro_e, D, E,
                               timespace, pulse_sequence, as_delay=False):
    orders = sorted(subclusters)
    norders = len(orders)

    # Data for zero cluster
    H, dimensions = total_elhamiltonian(np.array([]), ntype,
                                        I, B, S, gyro_e, D, E)
    dms_zero = compute_dm(dm0, dimensions, H, S, timespace, pulse_sequence,
                          as_delay=as_delay)
    dms_zero = ma.masked_array(dms_zero, mask=(dms_zero == 0))
    # print(dms_zero.mask)
    # If there is only one set of indexes for only one order,
    # Then for this subcluster nelements < maximum CCE order
    if norders == 1 and subclusters[orders[0]].shape[0] == 1:
        verticles = subclusters[orders[0]][0]

        H, dimensions = total_elhamiltonian(nspin[verticles], ntype,
                                            I, B, S, gyro_e, D, E)
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

            H, dimensions = total_elhamiltonian(nspin[v], ntype,
                                                I, B, S, gyro_e, D, E)
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
def decorated_density_matrix(nspin, ntype,
                             dm0, I, S, B, D, E,
                             timespace, pulse_sequence,
                             gyro_e=-17608.597050,
                             as_delay=False, zeroth_cluster=None):

    if zeroth_cluster is None:
        H, dimensions = total_elhamiltonian(np.array([]), ntype,
                                            I, B, S, gyro_e, D, E)
        zeroth_cluster = compute_dm(dm0, dimensions, H, S, timespace, pulse_sequence,
                                    as_delay=as_delay)
        zeroth_cluster = ma.masked_array(zeroth_cluster, mask=(zeroth_cluster == 0))

    H, dimensions = total_elhamiltonian(nspin, ntype,
                                        I, B, S, gyro_e, D, E)
    dms = compute_dm(dm0, dimensions, H, S, timespace,
                     pulse_sequence, as_delay=as_delay) / zeroth_cluster

    return dms


@cluster_expansion_decorator
def decorated_density_matrix(nspin, ntype,
                             dm0, I, S, B, D, E,
                             timespace, pulse_sequence,
                             gyro_e=-17608.597050,
                             as_delay=False, zeroth_cluster=None):

    if zeroth_cluster is None:
        H, dimensions = total_elhamiltonian(np.array([]), ntype,
                                            I, B, S, gyro_e, D, E)
        zeroth_cluster = compute_dm(dm0, dimensions, H, S, timespace, pulse_sequence,
                                    as_delay=as_delay)
        zeroth_cluster = ma.masked_array(zeroth_cluster, mask=(zeroth_cluster == 0))

    H, dimensions = total_elhamiltonian(nspin, ntype,
                                        I, B, S, gyro_e, D, E)
    dms = compute_dm(dm0, dimensions, H, S, timespace,
                     pulse_sequence, as_delay=as_delay) / zeroth_cluster

    return dms
