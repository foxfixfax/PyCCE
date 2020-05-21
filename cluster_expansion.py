import numpy as np
import functools
import operator


def cluster_expansion_decorator(_func=None, *, result_operator=operator.imul, contribution_operator=operator.ipow):
    def inner_cluster_expansion_decorator(function):

        @functools.wraps(function)
        def cluster_expansion(subclusters, nspin, *arg, **kwarg):
            revorders = sorted(subclusters)[::-1]
            norders = len(revorders)

            # If there is only one set of indexes for only one order,
            # Then for this subcluster nelements < maximum CCE order
            if norders == 1 and subclusters[revorders[0]].shape[0] == 1:
                verticles = subclusters[revorders[0]][0]
                result = function(nspin[verticles], *arg, **kwarg)

                return result

                # print(zero_power)

            result = 1
            result = contribution_operator(result, 0)
            # The Highest possible L will have all powers of 1
            power = {}
            # Number of visited orders from highest to lowest
            visited = 0
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

                    vcalc = function(nspin[v], *arg, **kwarg)

                    vcalc = contribution_operator(vcalc, power[order][index])

                    result = result_operator(result, vcalc)

                visited += 1
                print('Computed {} of order {} for {} clusters'.format(
                    function.__name__, order, subclusters[order].shape[0]))

            return result
        return cluster_expansion

    if _func is None:
        return inner_cluster_expansion_decorator
    else:
        return inner_cluster_expansion_decorator(_func)
