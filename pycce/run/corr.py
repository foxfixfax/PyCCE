import operator

import numpy as np
from pycce.h import total_hamiltonian, bath_interactions, expanded_single, conditional_hyperfine, \
    dimensions_spinvectors, overhauser_bath, overhauser_central, Hamiltonian
from pycce.run.base import RunObject, _check_projected_states, generate_supercluser_states
from pycce.utilities import generate_projections

from .gcce import generate_dm0, gen_density_matrix, propagator


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

    operator_i_t = np.matmul(np.transpose(U.conj(), axes=(0, 2, 1)), np.matmul(operator_i, U))
    # operator_j_t = np.matmul(np.transpose(U.conj(), axes=(0, 2, 1)), np.matmul(operator_j, U))
    it_j0 = np.matmul(operator_i_t, operator_j)  # + np.matmul(operator_j, operator_i_t)) / 2
    matmul = np.matmul(dm0_expanded, it_j0)
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
        central_spin (float):
            Value of the central spin.

    Returns:
        ndarray with shape (t,):
            correlation of the Overhauser field, induced by the given cluster at each time point.

    """
    a_is = np.zeros((3, *U.shape[1:]), dtype=np.complex128)

    dimensions, vectors = dimensions_spinvectors(nspin, central_spin=central_spin)
    for j, n in enumerate(nspin):
        ivec = vectors[j]
        hyperfine_tensor = n['A']
        aivec = np.array([hyperfine_tensor[0, 0] * ivec[0],
                          hyperfine_tensor[1, 1] * ivec[1],
                          hyperfine_tensor[2, 2] * ivec[2]])
        # aivec = np.einsum('ij,jkl->ikl', hyperfine_tensor, ivec)
        a_is += aivec

    # AI_x = correlation_it_j0(AIs[0], AIs[0], dm0_expanded, U)
    # AI_y = correlation_it_j0(AIs[1], AIs[1], dm0_expanded, U)
    AI_z = correlation_it_j0(a_is[2], a_is[2], dm0_expanded, U)

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

        self.dm0 = np.tensordot(self.state, self.state, axes=0)

    def postprocess(self):
        pass

    def generate_hamiltonian(self):
        """
        Using the attributes of the ``self`` object,
        compute the cluster hamiltonian including the central spin.

        Returns:
            Hamiltonian: Cluster hamiltonian.

        """
        ham = total_hamiltonian(self.cluster, self.magnetic_field, self.zfs, others=self.others,
                                other_states=self.other_states, central_gyro=self.gyro, central_spin=self.spin)
        return ham

    def compute_result(self):
        """
        Using the attributes of the ``self`` object,
        compute autocorrelation function of the noise from bath spins in the given cluster.

        Returns:

            ndarray: Computed autocorrelation function.
        """
        time_propagator = propagator(self.timespace, self.cluster_hamiltonian.data)

        dmtotal0 = generate_dm0(self.dm0, self.cluster_hamiltonian.dimensions, self.states)

        return compute_correlations(self.cluster, dmtotal0, time_propagator, central_spin=self.spin)


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
        self.projections = (np.abs(generate_projections(self.alpha)) + np.abs(generate_projections(self.beta))) / 2
        # self.projections = generate_projections(self.state)

    def postprocess(self):
        pass

    def generate_hamiltonian(self):
        """
        Using the attributes of the ``self`` object,
        compute the projected cluster hamiltonian, averaged for two qubit states.

        Returns:
            Hamiltonian: Cluster hamiltonian.

        """
        dimensions, ivectors = dimensions_spinvectors(self.cluster, central_spin=None)

        totalh = Hamiltonian(dimensions, vectors=ivectors)

        for ivec, n in zip(ivectors, self.cluster):
            hsingle = expanded_single(ivec, n.gyro, self.magnetic_field, n['Q'], n.detuning)

            if self.others is not None and self.other_states is not None:
                hsingle += overhauser_bath(ivec, n['xyz'], n.gyro, self.others.gyro,
                                           self.others['xyz'], self.other_states)

            hf = conditional_hyperfine(n['A'], ivec, self.projections)

            totalh.data += hsingle + hf

        totalh.data += bath_interactions(self.cluster, ivectors)

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
        dm0_expanded = gen_density_matrix(self.states, dimensions=self.cluster_hamiltonian.dimensions)

        time_propagator = propagator(self.timespace, self.cluster_hamiltonian.data)

        return compute_correlations(self.cluster, dm0_expanded, time_propagator)

    #
    # def kernel(self, cluster, **kwargs):
    #     """
    #     Inner decorated function to compute coherence function in conventional CCE.
    #
    #     Args:
    #         cluster (dict):
    #             clusters included in different CCE orders of structure ``{int order: ndarray([[i,j],[i,j]])}``.
    #
    #         **kwargs (any): Additional arguments for projected_hamiltonian.
    #
    #     Returns:
    #         ndarray: Coherence function of the central spin.
    #     """
    #     nspin = self.bath[cluster]
    #     states, others, other_states = _check_projected_states(cluster, self.bath, self.bath_state,
    #                                                            self.projected_bath_state)
    #
    #     dimensions, ivectors = dimensions_spinvectors(nspin, central_spin=None)
    #
    #     totalh = 0
    #
    #     for ivec, n in zip(ivectors, nspin):
    #         hsingle = expanded_single(ivec, n.gyro, self.magnetic_field, n['Q'], n.detuning)
    #
    #         if others is not None and other_states is not None:
    #             hsingle += overhauser_bath(ivec, n['xyz'], n.gyro, others.gyro,
    #                                        others['xyz'], other_states)
    #
    #         hf = conditional_hyperfine(n['A'], ivec, self.projections)
    #
    #         totalh += hsingle + hf
    #
    #     totalh += bath_interactions(nspin, ivectors)
    #
    #     dm0_expanded = gen_density_matrix(states, dimensions=dimensions)
    #
    #     time_propagator = propagator(self.timespace, totalh)
    #
    #     return compute_correlations(nspin, dm0_expanded, time_propagator)
    #
    # def interlaced_kernel(self, cluster, supercluster, *args, **kwargs):
    #     """
    #     Inner kernel function to compute coherence function in generalized CCE with interlaced averaging.
    #
    #     Args:
    #         cluster (ndarray): Indexes of the bath spins in the given cluster.
    #         supercluster (ndarray): Indexes of the bath spins in the supercluster of the given cluster.
    #
    #     Returns:
    #         ndarray: Coherence function of the central spin.
    #     """
    #     nspin = self.bath[cluster]
    #
    #     _, others, other_states = _check_projected_states(supercluster, self.bath, self.bath_state,
    #                                                       self.projected_bath_state)
    #
    #     dimensions, ivectors = dimensions_spinvectors(nspin, central_spin=None)
    #
    #     totalh = 0
    #
    #     for ivec, n in zip(ivectors, nspin):
    #         hsingle = expanded_single(ivec, n.gyro, self.magnetic_field, n['Q'], n.detuning)
    #
    #         if others is not None and other_states is not None:
    #             hsingle += overhauser_bath(ivec, n['xyz'], n.gyro, others.gyro,
    #                                        others['xyz'], other_states)
    #
    #         hf = conditional_hyperfine(n['A'], ivec, self.projections)
    #
    #         totalh += hsingle + hf
    #
    #     totalh += bath_interactions(nspin, ivectors)
    #
    #     sc_mask = ~np.isin(supercluster, cluster)
    #
    #     outer_indexes = supercluster[sc_mask]
    #     outer_spin = self.bath[outer_indexes]
    #
    #     initial_h = totalh.data
    #
    #     result = 0
    #     i = 0
    #
    #     for i, state in enumerate(generate_supercluser_states(self, supercluster)):
    #
    #         cluster_states = state[~sc_mask]
    #         outer_states = state[sc_mask]
    #
    #         if outer_spin.size > 0:
    #             addition = 0
    #
    #             for ivec, n in zip(ivectors, nspin):
    #                 addition += overhauser_bath(ivec, n['xyz'], n.gyro, outer_spin.gyro,
    #                                             outer_spin['xyz'], outer_states)
    #
    #             totalh.data = initial_h + addition
    #
    #         dm0_expanded = gen_density_matrix(cluster_states, dimensions=dimensions)
    #
    #         time_propagator = propagator(self.timespace, totalh)
    #
    #         result += compute_correlations(nspin, dm0_expanded, time_propagator)
    #
    #     result /= i + 1
    #
    #     return result
