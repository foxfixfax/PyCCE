import numpy as np
import numpy.ma as ma
from .hamiltonian import total_elhamiltonian, expand
from .elham import full_dm


def compute_correlation(dm0, dimensions, H, S, timespace):
    # Initializing all of the nuclear spins in the completely random state - their

    dm0 = expand(dm0, len(dimensions) - 1, dimensions) / \
          np.prod(dimensions[:-1])

    dm = full_dm(dm0, dimensions, H, S, timespace, pulse_sequence=None, as_delay=False)

    initial_shape = dm.shape
    dm.shape = (initial_shape[0], *dimensions, *dimensions)
    for d in range(len(dimensions) + 1, 2, -1):  # The last one is el spin
        dm = np.trace(dm, axis1=1, axis2=d)
    return dm


def cluster_correlation(subclusters, nspin, ntype,
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
        print(np.abs(power[order]).max())
        print(zero_power)
        visited += 1
        print('Computed density matrices of order {} for {} clusters in subcluster of size {}'.format(
            order, subclusters[order].shape[0], subclusters[1].size))

    return dms


def cluster2_dm(subclusters, nspin, ntype,
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

        # zero_power -= np.sum(power[order])
    # print(dms_zero)
    # print(dms)
    # dms *= dms_v ** zero_power
    # print(dms)
    return dms
