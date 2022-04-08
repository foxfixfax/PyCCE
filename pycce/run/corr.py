import operator

import numpy as np
import pycce.center
from pycce import dimensions_spinvectors
from pycce.h import total_hamiltonian, conditional_hyperfine, \
    Hamiltonian, bath_hamiltonian
from pycce.run.base import RunObject, simple_propagator, generate_initial_state

_rows = None
_cols = None
_newcols = None


# def _gen_indexes(size):
#     r = np.arange(size)
#     global _rows
#     global _cols
#     global _newcols
#
#     _rows = np.concatenate([r[:i] for i in range(r.size, 0, -1)])
#     _cols = np.concatenate([r[i:] for i in range(r.size, )])
#     _newcols = np.concatenate([np.ones(i, dtype=int) * (r.size - i) for i in range(r.size, 0, -1)])
#
#     if equispaced:
#         if _rows is None:
#             _gen_indexes(corr.size)
#
#         density_matrix = np.zeros((corr.size, corr.size), dtype=np.complex128)
#         density_matrix[_rows, _newcols] = np.triu(corr[np.newaxis, :] - corr[:, np.newaxis])[_rows, _cols]
#         top = density_matrix.sum(axis=0)
#         bottom = np.count_nonzero(density_matrix, axis=0)
#         bottom[bottom == 0] = 1
#         corr = top / bottom

def correlation_it_j0(operator_i, operator_j, dm0_expanded, U):
    """
    Function to compute correlation function of the operator i at time t and operator j at time 0

    Args:
        operator_i (ndarray with shape (n, n)):
            Matrix representation of operator i.
        operator_j (ndarray with shape (n, n)):
            Matrix representation of operator j.
        dm0_expanded (ndarray with shape (n, n)):
            Initial density matrix of the cluster.
        U (ndarray with shape (t, n, n)):
            Time evolution propagator, evaluated over t time points.

    Returns:
        ndarray with shape (t,): Autocorrelation of the z-noise at each time point.

    """

    operator_i_t = np.matmul(U.conj().transpose(0, 2, 1), np.matmul(operator_i, U))
    # operator_j_t = np.matmul(np.transpose(U.conj(), axes=(0, 2, 1)), np.matmul(operator_j, U))
    it_j0 = (np.matmul(operator_i_t, operator_j) + np.matmul(operator_j, operator_i_t)) / 2
    matmul = np.matmul(it_j0, dm0_expanded)
    corr = matmul.trace(axis1=1, axis2=2, dtype=np.complex128)

    return corr


def compute_correlations(nspin, dm0_expanded, U, central_spin=None):
    """
    Function to compute correlations for the given cluster, given time propagator U.

    Args:
        nspin (BathArray):
            BathArray of the given cluster of bath spins.
        dm0_expanded (ndarray with shape (n, n)):
            Initial density matrix of the cluster.
        U (ndarray with shape (t, n, n)):
            Time evolution propagator, evaluated over t time points.
        central_spin (CenterArray): Array of central spins.

    Returns:
        ndarray with shape (t,):
            correlation of the Overhauser field, induced by the given cluster at each time point.

    """
    a_is = np.zeros((3, *U.shape[1:]), dtype=np.complex128)

    dimensions, vectors = dimensions_spinvectors(nspin, central_spin=central_spin)
    if central_spin is not None and central_spin.size > 1:
        raise ValueError('Correlation calculations are supported only for single central spin')

    if len(nspin.A.shape) == len(nspin.shape) + 2:
        for j, n in enumerate(nspin):
            ivec = vectors[j]
            hyperfine_tensor = n['A']
            aivec = np.array([hyperfine_tensor[0, 0] * ivec[0],
                              hyperfine_tensor[1, 1] * ivec[1],
                              hyperfine_tensor[2, 2] * ivec[2]])
            # aivec = np.einsum('ij,jkl->ikl', hyperfine_tensor, ivec)
            # Still don't understand why this doesn't work
            a_is += aivec

        # AI_x = correlation_it_j0(AIs[0], AIs[0], dm0_expanded, U)
        # AI_y = correlation_it_j0(AIs[1], AIs[1], dm0_expanded, U)
        AI_z = correlation_it_j0(a_is[2], a_is[2], dm0_expanded, U)

    else:

        AI_z = []

        for i in range(nspin.A.shape[len(nspin.shape)]):
            for j, n in enumerate(nspin):
                ivec = vectors[j]
                hyperfine_tensor = n['A'][i]
                aivec = np.array([hyperfine_tensor[0, 0] * ivec[0],
                                  hyperfine_tensor[1, 1] * ivec[1],
                                  hyperfine_tensor[2, 2] * ivec[2]])
                a_is += aivec

            AI_z.append(correlation_it_j0(a_is[2], a_is[2], dm0_expanded, U))

    return AI_z  # np.array([AI_x, AI_y, AI_z])


class gCCENoise(RunObject):
    """
    Class for running generalized CCE simulations of the noise autocorrelation function.

    .. note::

        Subclass of the ``RunObject`` abstract class.

    Args:
        *args: Positional arguments of the ``RunObject``.
        **kwargs: Keyword arguments of the ``RunObject``.

    """
    result_operator = operator.iadd
    """Overridden operator which will combine the result of expansion: ``operator.iadd``."""
    contribution_operator = operator.imul
    """Overridden operator which will combine multiple contributions of the same cluster 
    in the optimized approach: ``operator.imul``."""

    removal_operator = operator.isub
    """Overridden operator which remove subcluster contribution
    from the given cluster contribution: ``operator.isub``."""

    addition_operator = np.sum
    """Overridden group operation which will combine contributions from the different clusters into
    one contribution in the direct approach: ``numpy.sum``."""

    def __init__(self, *args, **kwargs):
        self.dm0 = None
        super().__init__(*args, **kwargs)

    def preprocess(self):
        super().preprocess()

        self.dm0 = np.tensordot(self.center.state, self.center.state, axes=0)

    def postprocess(self):
        pass

    def generate_hamiltonian(self):
        """
        Using the attributes of the ``self`` object,
        compute the cluster hamiltonian including the central spin.

        Returns:
            Hamiltonian: Cluster hamiltonian.

        """
        ham = total_hamiltonian(self.cluster, self.center, self.magnetic_field)
        # ham.data += zero_order_addition(ham.vectors, self.cluster, self.others, self.others.proj)

        return ham

    def compute_result(self):
        """
        Using the attributes of the ``self`` object,
        compute autocorrelation function of the noise from bath spins in the given cluster.

        Returns:

            ndarray: Computed autocorrelation function.
        """
        time_propagator = simple_propagator(self.timespace, self.hamiltonian)

        dmtotal0 = generate_initial_state(self.base_hamiltonian.dimensions, central_state=self.dm0, states=self.states)

        return compute_correlations(self.cluster, dmtotal0, time_propagator, central_spin=self.center)


class CCENoise(RunObject):
    """
    Class for running conventional CCE simulations of the noise autocorrelation function.

    .. note::

        Subclass of the ``RunObject`` abstract class.


    .. warning::

        In general, for calculations of the autocorrelation function, better results are achieved with
        generalized CCE, which accounts for the evolution of the entangled state of the central spin.

        Second order couplings between nuclear spins are not implemented.

    Args:
        *args: Positional arguments of the ``RunObject``.
        **kwargs: Keyword arguments of the ``RunObject``.

    """
    result_operator = operator.iadd
    """Overridden operator which will combine the result of expansion: ``operator.iadd``."""
    contribution_operator = operator.imul
    """Overridden operator which will combine multiple contributions of the same cluster 
    in the optimized approach: ``operator.imul``."""

    removal_operator = operator.isub
    """Overridden operator which remove subcluster contribution
    from the given cluster contribution: ``operator.isub``."""

    addition_operator = np.sum
    """Overridden group operation which will combine contributions from the different clusters into
    one contribution in the direct approach: ``numpy.sum``."""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.projections = None

    def preprocess(self):
        super().preprocess()
        pycce.center.generate_projections()
        if self.center.size > 1:
            raise ValueError('Correlation calculations are supported only for single central spin')

        self.projections = (np.abs(self.center.projections_alpha) + np.abs(self.center.projections_beta)) / 2
        # self.projections = generate_projections(self.state)

    def postprocess(self):
        global _cols
        global _rows
        global _newcols

        _cols = None
        _rows = None
        _newcols = None

    def generate_hamiltonian(self):
        """
        Using the attributes of the ``self`` object,
        compute the projected cluster hamiltonian, averaged for two qubit states.

        Returns:
            Hamiltonian: Cluster hamiltonian.

        """
        totalh = bath_hamiltonian(self.cluster, self.magnetic_field)

        for ivec, n in zip(totalh.vectors, self.cluster):
            hsingle = 0
            for i, proj in enumerate(self.projections):
                if self.center.size > 1:
                    hf = n['A'][i]
                else:
                    hf = n['A']

                hsingle += conditional_hyperfine(hf, ivec, proj)

            totalh.data += hsingle

        # totalh.data += zero_order_addition(totalh.vectors, self.cluster, self.others, self.others.proj)

        return totalh
        # ham = total_hamiltonian(self.cluster, self.magnetic_field, self.zfs, others=self.others,
        #                         other_states=self.other_states, central_gyro=self.gyro, central_spin=self.spin)
        # return ham

    def compute_result(self):
        """
        Using the attributes of the ``self`` object,
        compute autocorrelation function of the noise from bath spins in the given cluster.

        Returns:

            ndarray: Computed autocorrelation function.
        """
        dm0_expanded = generate_initial_state(self.base_hamiltonian.dimensions, states=self.states, central_state=None)

        time_propagator = simple_propagator(self.timespace, self.hamiltonian)

        return compute_correlations(self.cluster, dm0_expanded, time_propagator)
