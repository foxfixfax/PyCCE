import operator

import numpy as np
from pycce.bath.array import BathArray
from pycce.h.total import total_hamiltonian
from pycce.h.functions import overhauser_central, overhauser_bath
from pycce.run.mc import monte_carlo_method_decorator

from .clusters import cluster_expansion_decorator, interlaced_decorator


class RunObject:
    r"""
    Abstract class of the CCE simulation runner.

    Implements cluster correlation expansion, interlaced averaging, and sampling over random bath states.
    Requires definition of the following methods, from which the kernel will be automatically created:

        - ``.generate_hamiltonian(self)`` method which, using the attributes of the ``self`` object,
          computes cluster hamiltonian stored in ``self.cluster_hamiltonian``.

        - ``.compute_result(self)`` method which, using the attributes of the ``self``, computes the resulting
          quantity for the given cluster.

    Alternatively, user can define the kernel manually. Then the following methods have to be overridden:

        - ``.kernel(self, cluster, *args, **kwargs)`` method which takes indexes of the bath spins in the given
          cluster as a first positional argument. This method is required for usual CCE runs.

        - ``.interlaced_kernel(self, cluster, supercluster, *args, **kwargs)`` method which takes
          indexes of the bath spins in the given cluster as a first positional argument, indexes of the supercluster
          as a second positional argument. This method is required for interlaced CCE runs.

    Args:

        timespace (ndarray with shape (t, )): Time delay values at which to compute propagators.

        clusters (dict):
            Clusters included in different CCE orders of structure ``{int order: ndarray([[i,j],[i,j]])}``.

        bath (BathArray with shape (n,)): Array of *n* bath spins.

        magnetic_field (ndarray): Magnetic field of type ``magnetic_field = np.array([Bx, By, Bz])``.

        alpha (int or ndarray with shape (2s+1, )): :math:`\ket{0}` state of the qubit in :math:`S_z`
            basis or the index of eigenstate to be used as one.

        beta (int or ndarray with shape (2s+1, )): :math:`\ket{1}` state of the qubit in :math:`S_z` basis
            or the index of the eigenstate to be used as one.

        state (ndarray with shape (2s+1, )):
            Initial state of the central spin, used in gCCE and noise autocorrelation calculations.
            Defaults to :math:`\frac{1}{N}(\ket{0} + \ket{1})` if not set **OR** if alpha and beta are provided as
            indexes.


        spin (float): Value of the central spin.

        zfs (ndarray with shape (3,3)): Zero Field Splitting tensor of the central spin.

        gyro (float or ndarray with shape (3, 3)):
            Gyromagnetic ratio of the central spin

            **OR**

            tensor corresponding to interaction between magnetic field and
            central spin.

        nbstates (int): Number of random bath states to sample over in bath state sampling runs.

        bath_state (ndarray): Array of bath states in any accepted format.

        seed (int): Seed for the random number generator in bath states sampling.

        masked (bool):
            True if mask numerically unstable points (with result > result[0]) in the sampling over bath states
            False if not. Default True.

        fixstates (dict):
            If provided, contains indexes of bath spins with fixed pure state for sampling runs and interlaced runs.

            Each key is the index of bath spin,
            value - fixed :math:`\hat{I}_z` projection of the **pure** :math:`\hat{I}_z` eigenstate of bath spin.

        projected_bath_state (ndarray with shape (n,)):
            Array with z-projections of the bath spins states.
            Overridden in runs with random bath state sampling.

        parallel (bool):
            True if parallelize calculation of cluster contributions over different mpi processes.
            Default False.

        direct (bool):
            True if use direct approach in run (requires way more memory but might be more numerically stable).
            False if use memory efficient approach. Default False.

        parallel_states (bool):
            True if use MPI to parallelize the calculations of density matrix
            for each random bath state.

        **kwargs: Additional keyword arguments to be set as the attributes of the given object.

    """

    result_operator = operator.imul
    """Operator which will combine the result of expansion,.
    
    Default: ``operator.imul``."""
    contribution_operator = operator.ipow
    """Operator which will combine multiple contributions of the same cluster 
    in the optimized approach.
    
    Default: ``operator.ipow``."""
    removal_operator = operator.itruediv
    """Operator which will remove subcluster contribution from the given cluster contribution.
    First argument cluster contribution, second - subcluster contribution.
    
    Defalut: ``operator.itruediv``."""
    addition_operator = np.prod
    """Group operation which will combine contributions from the different clusters into
    one contribution in the direct approach.
    
    Default: ``numpy.prod``."""

    def __init__(self, timespace,
                 clusters, bath,
                 magnetic_field,
                 alpha, beta, state,
                 spin, zfs, gyro,
                 nbstates=None,
                 bath_state=None,
                 seed=None,
                 masked=True,
                 fixstates=None,
                 projected_bath_state=None,
                 parallel=False,
                 direct=False,
                 parallel_states=False,
                 **kwargs):

        self.nbstates = nbstates
        """int: Number of random bath states to sample over in bath state sampling runs."""
        self.timespace = timespace
        """ndarray with shape (t, ): Time points at which result will be computed."""
        self.clusters = clusters
        """dict: Clusters included in different CCE orders of structure ``{int order: ndarray([[i,j],[i,j]])}``."""
        self.bath = bath
        """BathArray with shape (n,): Array of *n* bath spins."""
        self.spin = spin
        """float: Value of the central spin"""
        self.zfs = zfs
        """ndarray with shape (3, 3): Zero Field Splitting tensor of the central spin."""
        self.gyro = gyro
        """float or ndarray with shape (3, 3):
        Gyromagnetic ratio of the central spin

        **OR**

        tensor corresponding to interaction between magnetic field and
        central spin."""
        self.bath_state = bath_state
        """ndarray: Array of bath states in any accepted format."""
        self.projected_bath_state = projected_bath_state
        """ndarray with shape (n,): Array with z-projections of the bath spins states.
        Overridden in runs with random bath state sampling."""
        self.magnetic_field = magnetic_field
        """ndarray: Magnetic field of type ``magnetic_field = np.array([Bx, By, Bz])``."""
        alpha = np.asarray(alpha)

        beta = np.asarray(beta)

        self.initial_alpha = alpha
        r"""ndarray: :math:`\ket{0}` state of the qubit in :math:`S_z`
        basis or the index of eigenstate to be used as one."""
        self.initial_beta = beta
        r"""ndarray: :math:`\ket{1}` state of the qubit in :math:`S_z`
        basis or the index of eigenstate to be used as one."""
        self.alpha = alpha
        r"""ndarray: :math:`\ket{0}` state of the qubit in :math:`S_z`
        basis. If initially provided as index, generated as a state during the run."""
        self.beta = beta
        r"""ndarray: :math:`\ket{1}` state of the qubit in :math:`S_z`
        basis. If initially provided as index, generated as a state during the run."""
        self.state = state
        r"""ndarray: Initial state of the central spin, used in gCCE and noise autocorrelation calculations.
        
        Defaults to :math:`\frac{1}{N}(\ket{0} + \ket{1})` if not set **OR** if alpha and beta are provided as
        indexes."""
        self.parallel = parallel
        """bool: True if parallelize calculation of cluster contributions over different mpi processes.
        Default False."""
        self.parallel_states = parallel_states
        """bool: True if use MPI to parallelize the calculations of density matrix
        for each random bath state."""
        self.direct = direct
        """bool: True if use direct approach in run (requires way more memory but might be more numerically stable).
        False if use memory efficient approach. Default False."""
        # MC Bath state sampling parameters
        self.seed = seed
        """int:  Seed for the random number generator in bath states sampling."""
        self.masked = masked
        """bool: True if mask numerically unstable points (with result > result[0]) in the sampling over bath states
        False if not. Default True."""
        self.fixstates = fixstates
        r"""dict: If provided, contains indexes of bath spins with fixed pure state for sampling runs and interlaced runs.

        Each key is the index of bath spin,
        value - fixed :math:`\hat{I}_z` projection of the **pure** :math:`\hat{I}_z` eigenstate of bath spin."""

        for k in kwargs:
            setattr(self, k, kwargs[k])

        # self._cluster_hamiltonian = None

        # Central spin hamiltonian
        self.hamiltonian = None
        """ndarray: central spin hamiltonian."""
        self.energies = None
        """ndarray: Eigen energies of the central spin hamiltonian."""
        self.eigenvectors = None
        """ndarray: Eigen states of the central spin hamiltonian."""
        self.cluster = None
        """BathArray: Array of the bath spins inside the given cluster."""

        self.states = None
        """ndarray: Array of the states of bath spins inside the given cluster."""
        self.others = None
        """BathArray: Array of the bath spins outside the given cluster."""
        self.other_states = None
        """ndarray: Array of the z-projections of the states of bath spins outside the given cluster."""
        self.cluster_hamiltonian = None
        """Hamiltonian or tuple: Full hamiltonian of the given cluster. In conventional CCE, tuple with two 
        projected hamiltonians."""
        self.result = None
        """ndarray: Result of the calculation."""

    def preprocess(self):
        """
        Method which will be called before cluster-expanded run.
        """
        self.hamiltonian = total_hamiltonian(BathArray((0,)), self.magnetic_field, zfs=self.zfs, others=self.bath,
                                             other_states=self.projected_bath_state,
                                             central_gyro=self.gyro, central_spin=self.spin)

        self.energies, self.eigenvectors = np.linalg.eigh(self.hamiltonian)

        alpha = self.initial_alpha
        beta = self.initial_beta

        if (not alpha.shape) or (not beta.shape):

            alpha = self.eigenvectors[:, alpha]
            beta = self.eigenvectors[:, beta]

            state = (alpha + beta) / np.linalg.norm(alpha + beta)

            self.alpha = alpha
            self.beta = beta

        else:
            state = self.state

        self.state = state

    def postprocess(self):
        """
            Method which will be called after cluster-expanded run.
        """
        pass

    def generate_hamiltonian(self):
        raise NotImplementedError

    def compute_result(self):
        raise NotImplementedError

    def kernel(self, cluster, *args, **kwargs):
        """
        Central kernel that will be called in the cluster-expanded calculations

        Args:
            cluster (ndarray): Indexes of the bath spins in the given cluster.
            *args: Positional arguments of the kernel.
            **kwargs: Keyword arguments of the kernel.

        Returns:

            ndarray: Results of the calculations.

        """
        self.cluster = self.bath[cluster]

        self.states, self.others, self.other_states = _check_projected_states(cluster, self.bath, self.bath_state,
                                                                              self.projected_bath_state)

        self.cluster_hamiltonian = self.generate_hamiltonian()

        result = self.compute_result()

        return result

    @property
    def __inner_kernel(self):
        return cluster_expansion_decorator(self.kernel,
                                           result_operator=self.result_operator,
                                           contribution_operator=self.contribution_operator,
                                           removal_operator=self.removal_operator,
                                           addition_operator=self.addition_operator)

    def run(self, *args, **kwargs):
        """
        Method that runs cluster-expanded single calculation.

        Args:
            *args: Positional arguments of the kernel.
            **kwargs: Keyword arguments of the kernel.

        Returns:

            ndarray: Results of the calculations.

        """
        self.preprocess()
        self.result = self.__inner_kernel(self, *args, **kwargs)
        self.postprocess()
        return self.result

    @monte_carlo_method_decorator
    def __inner_sampled_run(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def sampling_run(self, *args, **kwargs):
        """
        Method that runs bath sampling calculations.

        Args:
            *args: Positional arguments of the kernel.
            **kwargs: Keyword arguments of the kernel.

        Returns:

            ndarray: Results of the calculations.

        """
        self.result = self.__inner_sampled_run(*args,
                                               **kwargs)
        return self.result

    def interlaced_kernel(self, cluster, supercluster, *args, **kwargs):
        """
        Central kernel that will be called in the cluster-expanded calculations
        with interlaced averaging of bath spin states.

        Args:
            cluster (ndarray): Indexes of the bath spins in the given cluster.

            supercluster (ndarray): Indexes of the bath spins in the supercluster of the given cluster.
                Supercluster is the union of all clusters in ``.clusters`` attribute, for which given cluster
                is a subset.

            *args: Positional arguments of the kernel.

            **kwargs: Keyword arguments of the kernel.

        Returns:

            ndarray: Results of the calculations.

        """

        self.cluster = self.bath[cluster]

        self.states, self.others, self.other_states = _check_projected_states(supercluster, self.bath, self.bath_state,
                                                                              self.projected_bath_state)

        self.cluster_hamiltonian = self.generate_hamiltonian()

        projected = isinstance(self.cluster_hamiltonian, tuple)

        sc_mask = ~np.isin(supercluster, cluster)

        outer_indexes = supercluster[sc_mask]
        outer_spin = self.bath[outer_indexes]

        if projected:
            initial_h0, initial_h1 = (c.data for c in self.cluster_hamiltonian)
            vectors = self.cluster_hamiltonian[0].vectors

        else:
            initial_h0 = self.cluster_hamiltonian.data
            initial_h1 = None
            vectors = self.cluster_hamiltonian.vectors

        result = 0
        i = 0

        for i, state in enumerate(generate_supercluser_states(self, supercluster)):

            self.states = state[~sc_mask]
            outer_states = state[sc_mask]

            if outer_spin.size > 0:
                addition = 0 if projected else overhauser_central(vectors[-1], outer_spin['A'], outer_states)

                for ivec, n in zip(vectors, self.cluster):
                    addition += overhauser_bath(ivec, n['xyz'], n.gyro, outer_spin.gyro,
                                                outer_spin['xyz'], outer_states)

                if projected:
                    self.cluster_hamiltonian[0].data = initial_h0 + addition
                    self.cluster_hamiltonian[1].data = initial_h1 + addition

                else:
                    self.cluster_hamiltonian.data = initial_h0 + addition

            result += self.compute_result()

        result /= i + 1
        return result

    @property
    def __inner_interlaced_kernel(self):
        return interlaced_decorator(self.interlaced_kernel,
                                    result_operator=self.result_operator,
                                    contribution_operator=self.contribution_operator)

    def interlaced_run(self, *args, **kwargs):
        """
        Method that runs cluster-expanded single calculation with interlaced averaging of bath spin states.

        Args:
            *args: Positional arguments of the interlaced kernel.
            **kwargs: Keyword arguments of the interlaced kernel.

        Returns:

            ndarray: Results of the calculations.

        """
        self.preprocess()
        self.result = self.__inner_interlaced_kernel(self, *args, **kwargs)
        self.postprocess()
        return self.result

    @monte_carlo_method_decorator
    def __inner_sampled_interlaced_run(self, *args, **kwargs):
        return self.interlaced_run(*args, **kwargs)

    def sampling_interlaced_run(self, *args, **kwargs):
        """
        Method that runs bath sampling calculations with interlaced averaging of bath spin states.

        Args:
            *args: Positional arguments of the interlaced kernel.
            **kwargs: Keyword arguments of the interlaced kernel.

        Returns:

            ndarray: Results of the calculations.

        """
        self.result = self.__inner_sampled_interlaced_run(*args,
                                                          **kwargs)
        return self.result

    @classmethod
    def from_simulator(cls, sim, **kwargs):
        r"""
        Class method to generate ``RunObject`` from the properties of ``Simulator`` object.

        Args:
            sim (Simulator): Object, whose properties will be used to initialize ``RunObject`` instance.
            **kwargs: Additional keyword arguments that will replace ones, recovered from the ``Simulator`` object.

        Returns:
            RunObject: New instance of ``RunObject`` class.

        """
        parameters = vars(sim).copy()

        for k in list(parameters.keys()):
            if k[0] == '_':
                parameters[k[1:]] = parameters.pop(k)

        parameters.update(**kwargs)
        run = cls(**parameters)

        return run


def generate_supercluser_states(self, supercluster):
    """
    Helper function to generate all possible pure states of the given supercluster.

    Args:
        self (RunObject): Instance of the ``RunObject`` class, used in the calculation.
        supercluster (ndarray with shape (n, )): Indexes of the bath spins in the supercluster.

    Yields:
        ndarray with shape (n, ): Pure state of the given supercluster.

    """
    scspins = self.bath[supercluster]
    states = np.asarray(np.meshgrid(*(np.linspace(-s.s, s.s, s.dim) for s in scspins))).T.reshape(-1, scspins.size)

    if self.fixstates is not None:

        indexes = np.fromiter((ind for ind in self.fixstates.keys()), dtype=np.int32)

        which = np.isin(supercluster, indexes)

        if any(which):

            newindexes = np.arange(supercluster.size)

            for k, nk in zip(supercluster[which], newindexes[which]):
                states = states[states[:, nk] == self.fixstates[which]]

    for single in states:
        yield single


def _check_projected_states(cluster, allspin, states=None, projected_states=None):
    others = None
    other_states = None

    if states is not None:
        states = states[cluster]

    if projected_states is not None:
        others_mask = np.ones(allspin.shape, dtype=bool)
        others_mask[cluster] = False
        others = allspin[others_mask]
        other_states = projected_states[others_mask]

    return states, others, other_states
