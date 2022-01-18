import operator
from collections import defaultdict

import numpy as np
import scipy.linalg
from pycce.h import zero_order_addition, zero_order_imap
from pycce.run.clusters import cluster_expansion_decorator, interlaced_decorator
from pycce.run.mc import monte_carlo_method_decorator
from pycce.run.pulses import Sequence
from pycce.sm import _smc
from pycce.utilities import expand, shorten_dimensions, gen_state_list


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

    #        fixstates (dict):
    #             If provided, contains indexes of bath spins with fixed pure state for sampling runs and interlaced runs.
    #
    #             Each key is the index of bath spin,
    #             value - fixed :math:`\hat{I}_z` projection of the **pure** :math:`\hat{I}_z` eigenstate of bath spin.

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
                 center=None,
                 pulses=None,
                 nbstates=None,
                 seed=None,
                 masked=True,
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
        self.center = center
        """CenterArray: Properties of the central spin."""
        # self.projected_bath_state = projected_bath_state
        # """ndarray with shape (n,): Array with z-projections of the bath spins states.
        # Overridden in runs with random bath state sampling."""
        self.magnetic_field = magnetic_field
        """ndarray: Magnetic field of type ``magnetic_field = np.array([Bx, By, Bz])``."""

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

        for k in kwargs:
            setattr(self, k, kwargs[k])

        # self._cluster_hamiltonian = None

        self.hamiltonian = None
        """ndarray: Cluster Hamiltonian."""

        self.cluster = None
        """BathArray: Array of the bath spins inside the given cluster."""
        self.cluster_indexes = None
        self.others = None
        """BathArray: Array of the bath spins outside the given cluster."""
        self.others_mask = None
        """ndarray: Bool array of of size self.bath with True entries for each of the spin outside of the cluster."""
        self.states = None

        self.has_states = False
        """bool: Whether there are states provided in the bath during the run."""
        self.initial_states_mask = bath.has_state
        """ndarray: Bool array of the states, initially present in the bath."""
        self.pulses = pulses
        """ Sequence: Sequence object, containing series of pulses, applied to the system."""

        self.projected_states = None

        self.base_hamiltonian = None
        """Hamiltonian or tuple: Full hamiltonian of the given cluster. In conventional CCE, tuple with two 
        projected hamiltonians."""
        self.result = None
        """ndarray: Result of the calculation."""
        self.delays = None
        """list or None: List with delays before each pulse or None if equispaced.
        Generated by ``.generate_pulses`` method."""
        self.rotations = None
        """list: List with matrix representations of the rotation from each pulse.
        Generated by ``.generate_pulses`` method."""

    def preprocess(self):
        """
        Method which will be called before cluster-expanded run.
        """

        self.base_hamiltonian = None
        self.hamiltonian = None

        self.cluster = None
        self.cluster_indexes = None
        self.others = None
        self.others_mask = None

        self.states = None
        self.projected_states = None

        self.result = None
        self.delays = None
        self.rotations = None

        self.has_states = self.bath.state.any()

        if self.has_states:
            bath = self.bath
            # proj = self.bath.proj
            if isinstance(self.pulses, Sequence) and any(p.bath_names is not None for p in self.pulses):
                self.projected_states = generate_rotated_projected_states(bath, self.pulses)

        else:
            bath = None
            # proj = None

        self.center.generate_states(self.magnetic_field, bath=bath)

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
        self.cluster_indexes = cluster
        self.cluster = self.bath[cluster]

        if self.has_states:
            others_mask = np.ones(self.bath.shape, dtype=bool)
            others_mask[cluster] = False
            self.others_mask = others_mask
            self.others = self.bath[others_mask]
            self.states = self.cluster.state  # list(  [...])
        else:
            self.others = None
            self.states = None

        self.base_hamiltonian = self.generate_hamiltonian()
        self.check_hamiltonian()
        if isinstance(self.pulses, Sequence):
            self.generate_pulses()

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

        others_mask = np.ones(self.bath.shape, dtype=bool)
        others_mask[supercluster] = False
        self.others_mask = others_mask
        self.base_hamiltonian = self.generate_hamiltonian()

        sc_mask = ~np.isin(supercluster, cluster)

        outer_indexes = supercluster[sc_mask]
        outer_spin = self.bath[outer_indexes]

        # initial_h0 = self.cluster_hamiltonian.data
        # vectors = self.cluster_hamiltonian.vectors

        index_state = 0
        initial_outer_spin_state = outer_spin.state

        result = 0

        for index_state, state in enumerate(self.generate_supercluser_states(outer_indexes)):

            self.bath.state[outer_indexes] = gen_state_list(state, outer_spin.dim)
            # Note that if put others in different line the updated bath state might not copy properly
            self.others = self.bath[others_mask]

            if self.projected_states is not None:
                self.projected_states[outer_indexes] = generate_rotated_projected_states(outer_spin, self.pulses)
            self.check_hamiltonian()
            result += self.compute_result()

        self.bath.state[outer_indexes] = initial_outer_spin_state

        result /= index_state + 1

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
            supercluster (ndarray with shape (n, )): Indexes of the bath spins in the supercluster.

        Yields:
            ndarray with shape (n, ): Pure state of the given supercluster.

        """
        sc = self.bath[supercluster]
        his = self.initial_states_mask[supercluster]  # have initial states

        if not his.any():
            states = np.asarray(np.meshgrid(*[np.linspace(-s.s, s.s, s.dim) for s in sc])).T.reshape(-1, sc.size)
        else:
            # one liner I'm hecking proud of
            # 1) generate list of linspaces spanning the possible Sz projections if no initial state provided
            #    othervise give an zero-sized array of the projection
            # 2) Create meshgrid out of this list with the dimensions (sc.size, dim_1, dim_2, 1, dim_4, ..., dim_n)
            #    where dim is dimensions of the nuclear spin and 1 in case only one projection is given
            # 3) Transpose to change to (dim_n, dim_n-1, ..., dim_4, 1, dim_2, dim_1, sc.size)
            states = np.asarray(np.meshgrid(
                *[np.linspace(-sc[i].s, sc[i].s, sc[i].dim) if not his[i] else sc[i].proj for i in
                  range(sc.size)])).T.reshape(-1, sc.size)

        for single in states:
            yield single

    def generate_pulses(self):
        """
        Generate list of matrix representations of the rotations, induced by the sequence of the pulses.

        The rotations are stored in the ``.rotation`` attribute of the each ``Pulse`` object
        and in ``Sequence.rotations``.

        Args:
            dimensions (ndarray with shape (N,)): Array of spin dimensions in the system.
            bath (BathArray with shape (n,)): Array of bath spins in the system.
            vectors (ndarray  with shape (N, 3, prod(dimensions), prod(dimensions))):
                 Array with vectors of spin matrices for each spin in the system.

            central_spin (CenterArray): Optional. Array of central spins.

        Returns:
            tuple: *tuple* containing:

            * **list** or **None**: List with delays before each pulse or None if equispaced.

            * **list**: List with matrix representations of the rotation from each pulse.
        """
        dimensions = self.base_hamiltonian.dimensions
        vectors = self.base_hamiltonian.vectors

        cs = self.center
        bath = self.cluster
        ndims = len(dimensions)
        if cs is not None:
            if bath is not None and ndims == bath.size:
                cs = None  # Ignore central spin
                nc = None
                shortdims = None
            else:
                nc = len(cs)  # number of central spins
                shortdims = shorten_dimensions(dimensions, nc)

        self.delays = None
        self.rotations = None

        equispaced = not any(p._has_delay for p in self.pulses)

        if equispaced:
            delays = None

        else:
            delays = [p.delay if p.delay is not None else 0 for p in self.pulses]

        rots = []
        # Sigma as if central spin array is total spin. Sigma - pauli matrix
        total_sigma = {}
        separate_sigma = defaultdict(dict)
        for p in self.pulses:

            initial_rotation = rotation = 1

            if p.angle and cs is not None:
                if p.which is None:
                    if p.axis not in total_sigma:
                        total_sigma[p.axis] = expand(cs.sigma[p.axis], ndims - nc, shortdims)
                    rotation = spin_matrix_rotation(total_sigma[p.axis] / 2, p.angle, spin_half=True)

                else:
                    for i in p.which:
                        c = cs[i]
                        if p.axis not in separate_sigma[i]:
                            separate_sigma[i][p.axis] = expand(c.sigma[p.axis], ndims - nc, shortdims)
                        rotation = np.dot(spin_matrix_rotation(separate_sigma[i][p.axis] / 2,
                                                               p.angle, spin_half=True),
                                          rotation)

            if p.bath_names is not None and bath is not None:
                if vectors.shape != bath.shape:
                    vectors = vectors[:bath.shape[0]]
                    # print(vectors.shape)
                properties = np.broadcast(p.bath_names, p.bath_axes, p.bath_angles)

                for name, axis, angle in properties:
                    if angle:
                        which = (bath.N == name)

                        if any(which):
                            vecs = vectors[which]
                            rotation = np.dot(bath_rotation(vecs, axis, angle), rotation)

            if initial_rotation is rotation:
                rotation = None

            p.rotation = rotation

            rots.append(rotation)

        self.delays = delays
        self.rotations = rots

        return delays, rots

    def check_hamiltonian(self):
        self.hamiltonian = None
        # if no projected states hamiltonian is simple
        if self.projected_states is None:

            if self.others is not None:
                addition = zero_order_imap(self.base_hamiltonian.vectors, self.cluster_indexes, self.bath,
                                           self.bath.proj)
                # addition = zero_order_addition(self.base_hamiltonian.vectors, self.cluster, self.others,
                #                                self.others.proj)
                self.hamiltonian = self.base_hamiltonian.data + addition

            else:
                self.hamiltonian = self.base_hamiltonian.data

    def ham_generator(self, index=0):
        if self.projected_states is None:
            return self.hamiltonian
        if self.others is not None:
            # addition = zero_order_addition(self.base_hamiltonian.vectors, self.cluster, self.others,
            #                                self.projected_states[self.others_mask, index])
            addition = zero_order_imap(self.base_hamiltonian.vectors, self.cluster_indexes, self.bath,
                                       self.projected_states[:, index])
        else:
            addition = 0
        return self.base_hamiltonian.data + addition


def generate_rotated_projected_states(bath, pulses):
    state = bath.state
    projected = [state.proj]
    rotations = {}
    for p in pulses:
        if p.bath_names is not None:
            proj = state.proj.copy()

            properties = np.broadcast(p.bath_names, p.bath_axes, p.bath_angles)
            for name, axis, angle in properties:
                which = (bath.N == name)

                if any(which) and angle:
                    spin = bath.types[name].s

                    rotation = spin_matrix_rotation(getattr(_smc[spin], axis), angle, spin < 1)

                    if name in rotations:
                        rotation = rotation @ rotations[name]

                    rotations[name] = rotation
                    proj[which] = bath[which].state.project(rotation)

            projected.append(proj)

    return np.swapaxes(projected, 0, -1)


_rot = {'x': 0, 'y': 1, 'z': 2}


def bath_rotation(vectors, axis, angle):
    """
    Generate rotation of the bath spins with given spin vectors.

    Args:
        vectors (ndarray with shape (n, 3, x, x)): Array of *n* bath spin vectors.
        axis (str): Axis of rotation.
        angle (float): Angle of rotation.

    Returns:
        ndarray with shape (x, x): Matrix representation of the spin rotation.

    """
    ax = _rot[axis]  # name -> index
    rotation = spin_matrix_rotation(vectors[0][ax], angle, spin_half=vectors[0, 0, 0, 0] < 1)

    for v in vectors[1:]:
        np.matmul(rotation, spin_matrix_rotation(v[ax], angle, spin_half=v[0, 0, 0] < 1), out=rotation)

    return rotation


def spin_matrix_rotation(sm, angle, spin_half=False):
    if angle == np.pi and spin_half:
        rotation = -1j * 2 * sm
    else:
        rotation = scipy.linalg.expm(-1j * sm * angle)
    return rotation


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
