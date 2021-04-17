import functools
import operator

import numpy as np

from .sm import _smc

try:
    from mpi4py import MPI

    mpiop = {'imul': MPI.PROD,
             'mul': MPI.PROD,
             'prod': MPI.PROD,
             'iadd': MPI.SUM,
             'add': MPI.SUM,
             'sum': MPI.SUM
             }

except ImportError:
    mpiop = None


def cluster_expansion_decorator(_func=None, *,
                                result_operator=operator.imul,
                                contribution_operator=operator.ipow,
                                removal_operator=operator.itruediv,
                                addition_operator=np.prod):
    """
    Decorator for creating cluster correlation expansion. Each expanded function will have two first arguments:
    subclusters and allnspin
    :param _func: function to expand
    :param result_operator: function
        operator which will combine the result of expansion (default: operator.imul)
    :param contribution_operator: function
        operator which will combine multiple contributions
        of the same cluster (default: operator.ipow)
    :return: function
    """

    def inner_cluster_expansion_decorator(function):

        @functools.wraps(function)
        def cluster_expansion(subclusters, allspin, *arg, parallel=False, direct=False, **kwarg):
            """
            Inner part of cluster expansion.
            :param direct: bool
                True if use direct approach (requires way more memory but might be more numerically stable).
                False if use memory efficient approach. Default False
            :param parallel: bool
                True if parallelize calculation of cluster contributions over different mpi threads.
                Default False
            :param subclusters: dict
                dict of subclusters included in different CCE order
                of structure {int order: np.array([[i,j],[i,j]])}
            :param allspin: ndarray
                array of bath

            :param arg:
                all additional arguments
            :param kwarg:
                all additional keyword arguments

            :return:
            """
            if direct:
                return direct_approach(function, subclusters, allspin, *arg, parallel=parallel,
                                       result_operator=result_operator,
                                       removal_operator=removal_operator,
                                       addition_operator=addition_operator,
                                       **kwarg)
            else:
                return optimized_approach(function, subclusters, allspin, *arg, parallel=parallel,
                                          result_operator=result_operator,
                                          contribution_operator=contribution_operator,
                                          **kwarg)

        return cluster_expansion

    if _func is None:
        return inner_cluster_expansion_decorator
    else:
        return inner_cluster_expansion_decorator(_func)


def optimized_approach(function, subclusters, allspin, *arg, parallel=False,
                       result_operator=operator.imul,
                       contribution_operator=operator.ipow,
                       **kwarg):
    """
    Inner part of cluster expansion.
    :param subclusters: dict
        dict of subclusters included in different CCE order
        of structure {int order: np.array([[i,j],[i,j]])}
    :param allspin: ndarray
        array of bath

    :param arg:
        all additional arguments
    :param kwarg:
        all additional keyword arguments
    :return:
    """
    revorders = sorted(subclusters)[::-1]
    norders = len(revorders)

    if parallel:
        try:
            from mpi4py import MPI
        except ImportError:
            print('Parallel failed: mpi4py is not found. Running serial')
            parallel = False

    if parallel:
        comm = MPI.COMM_WORLD

        size = comm.Get_size()
        rank = comm.Get_rank()
    else:
        rank = 0

    # If there is only one set of indexes for only one order,
    # Then for this subcluster nelements < maximum CCE order
    if norders == 1 and subclusters[revorders[0]].shape[0] == 1:
        verticles = subclusters[revorders[0]][0]
        return function(verticles, allspin, *arg, **kwarg)

    result = 1
    result = contribution_operator(result, 0)
    # The Highest possible L will have all powers of 1
    power = {}
    # Number of visited orders from highest to lowest
    visited = 0
    for order in revorders:
        nclusters = subclusters[order].shape[0]
        current_power = np.ones(nclusters, dtype=np.int32)
        # indexes of the cluster of size order are stored in v
        if parallel:
            remainder = nclusters % size
            add = int(rank < remainder)
            each = nclusters // size
            block = each + add
            start = rank * each + rank if rank < remainder else rank * each + remainder
        else:
            start = 0
            block = nclusters

        for index in range(start, start + block):

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
                current_power[index] -= np.sum(power[higherorder][containv], dtype=np.int32)

            vcalc = function(v, allspin, *arg, **kwarg)
            vcalc = contribution_operator(vcalc, current_power[index])

            result = result_operator(result, vcalc)
        if parallel:
            comm.Barrier()
            buffer = np.empty(current_power.shape, dtype=np.int32)
            comm.Allreduce(current_power, buffer, MPI.SUM)
            current_power = buffer - size + 1

        power[order] = current_power

        visited += 1
        # print('Computed {} of order {} for {} clusters'.format(
        #     function.__name__, order, subclusters[order].shape[0]))
    _smc.clear()

    if parallel:
        if rank == 0:
            result_shape = result.shape
        else:
            result_shape = None
        result_shape = comm.bcast(result_shape, root=0)
        if np.asarray(result).shape != result_shape:
            result = np.ones(result_shape, dtype=np.complex128)
            result = contribution_operator(result, 0)

        root_result = np.zeros(result_shape, dtype=np.complex128)
        comm.Allreduce(result, root_result, mpiop[result_operator.__name__])

    else:
        root_result = result

    return root_result


def direct_approach(function, subclusters, allspin, *arg, parallel=False,
                    result_operator=operator.imul,
                    removal_operator=operator.itruediv,
                    addition_operator=np.prod,
                    **kwarg):
    """
    Inner part of cluster expansion.
    :param subclusters: dict
        dict of subclusters included in different CCE order
        of structure {int order: np.array([[i,j],[i,j]])}
    :param allspin: ndarray
        array of bath

    :param arg:
        all additional arguments
    :param kwarg:
        all additional keyward
    :return:
    """
    if parallel:
        try:
            from mpi4py import MPI
        except ImportError:
            print('Parallel failed: mpi4py is not found. Running serial')
            parallel = False
    orders = sorted(subclusters)
    norders = len(orders)

    if parallel:
        comm = MPI.COMM_WORLD

        size = comm.Get_size()
        rank = comm.Get_rank()
    else:
        rank = 0

    # print(dms_zero.mask)
    # If there is only one set of indexes for only one order,
    # Then for this subcluster nelements < maximum CCE order
    if norders == 1 and subclusters[orders[0]].shape[0] == 1:
        verticles = subclusters[orders[0]][0]

        return function(verticles, allspin, *arg, **kwarg)

        # print(zero_power)
    # The Highest possible L will have all powers of 1
    result_tilda = {}
    visited = 0
    result = 1 - result_operator(1, 0)

    for order in orders:
        current_order = []
        # indexes of the cluster of size order are stored in v
        nclusters = subclusters[order].shape[0]

        if parallel:
            remainder = nclusters % size
            add = int(rank < remainder)
            each = nclusters // size
            block = each + add
            start = rank * each + rank if rank < remainder else rank * each + remainder
        else:
            start = 0
            block = nclusters

        for index in range(start, start + block):

            v = subclusters[order][index]
            vcalc = function(v, allspin, *arg, **kwarg)

            for lowerorder in orders[:visited]:
                contained_in_v = np.all(np.isin(subclusters[lowerorder], v), axis=1)
                lower_vcalc = addition_operator(result_tilda[lowerorder][contained_in_v], axis=0)
                vcalc = removal_operator(vcalc, lower_vcalc)

            current_order.append(vcalc)
        current_order = np.array(current_order, copy=False)

        if parallel:
            comm.Barrier()
            result_shape = vcalc.shape if rank == 0 else None
            result_shape = comm.bcast(result_shape, root=0)

            chunk = np.zeros((nclusters, *result_shape), dtype=np.complex128)
            chunk[start:start + block] = current_order.reshape(block, *result_shape)

            currrent_buffer = np.zeros((nclusters, *result_shape), dtype=np.complex128)
            comm.Allreduce(chunk, currrent_buffer, MPI.SUM)
            current_order = currrent_buffer

        result_tilda[order] = current_order
        visited += 1

    if rank == 0:
        for o in orders:
            result = result_operator(result, np.prod(result_tilda[o], axis=0))
    else:
        result = None
    if parallel: comm.Barrier()
    return result
