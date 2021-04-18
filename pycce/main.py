"""
    Default Units
    - Distance: Angstrom, A
    - Time: Millisecond, ms
    - Magnetic Field: Gaussian, G = 1e-4 Tesla
    - Gyromagnetic Ratio: rad/(msec*Gauss)
    - Quadrupole moment: millibarn

"""
import warnings

import numpy as np
import numpy.ma as ma
from pycce.io.xyz import read_xyz

from .bath.array import BathArray, SpinDict
from .bath.read_cube import Cube
from .calculators.coherence_function import decorated_coherence_function
from .calculators.correlation_function import decorated_noise_correlation, \
    projected_noise_correlation, noise_sampling
from .calculators.density_matrix import decorated_density_matrix, compute_dm
from .calculators.mean_field_dm import mean_field_density_matrix, monte_carlo_sampling
from .find_clusters import generate_clusters
from .hamiltonian import total_hamiltonian, mean_field_hamiltonian, generate_projections
from .units import ELECTRON_GYRO
from .utilities import zfs_tensor, project_bath_states


# TODO unit conversion
class Simulator:
    """
    The main class for CCE calculations

    Parameters
    ------------
    :param spin: Total spin of the central spin (default 1.)
    :type spin: float

    :param position: xyz coordinates in Angstrom of the central spin (default (0., 0., 0.))
    :type position: ndarray

    :param alpha:
        0-state of the qubit in Sz basis. Default state with
        ms = +s where ms is the z-projection of the spin and s is the total spin)
    :type alpha: ndarray

    :param beta: 1-state of the qubit in Sz basis (default state with ms = +s - 1)
    :type beta: ndarray

    :param gyro: Gyromagnetic ratio of central spin in rad / (ms * G) **OR** tensor describing central spin
        interactions with the magnetic field (default -17608.597050 rad / (ms * G) - gyromagnetic ratio of the
        free electron spin)
    :type gyro: float or ndarray

    :param D: D (longitudinal splitting) parameter of central spin in ZFS tensor of central spin in rad * kHz
        **OR** total ZFS tensor
    :type D: float or ndarray with shape (3,3)

    :param E: E (transverse splitting) parameter of central spin in ZFS tensor of central spin in rad * kHz
    :type E: float

    :param order:
        Argument for the ``Simulator.generate_clusters`` method. Denotes maximum size of the cluster
    :type order: int

    :param r_dipole:
        Argument for the ``Simulator.generate_clusters`` method. Denotes maximum connectivity distance
    :type r_dipole: float

    :param bath:
        First positional argument of the ``Simulator.read_bath`` method.

        Either:
        - Instance of BathArray class;
        - ndarray with ``dtype([('N', np.unicode_, 16), ('xyz', np.float64, (3,))])`` containing names
        of bath spins (same ones as stored in self.ntype) and positions of the spins in angstroms;
        - the name of the xyz text file containing 4 cols: name of the bath spin and xyz coordinates in A
    :type bath: ndarray or str

    :param bath_kw:
        Additional keyword arguments for the ``Simulator.read_bath`` method.

    """

    def __init__(self, spin=1., position=None, alpha=None, beta=None, gyro=ELECTRON_GYRO,
                 D=0, E=0, r_dipole=None, order=None, bath=None, **bath_kw):

        if position is None:
            position = np.zeros(3)

        self.position = np.asarray(position, dtype=np.float64)
        self.spin = spin

        self.zfs = zfs_tensor(D, E)

        self.bath_types = SpinDict()

        if alpha is None:
            alpha = np.zeros(int(round(2 * spin + 1)), dtype=np.complex128)
            alpha[0] = 1

        if beta is None:
            beta = np.zeros(int(round(2 * spin + 1)), dtype=np.complex128)
            beta[1] = 1

        self._alpha = np.asarray(alpha)
        self._beta = np.asarray(beta)

        self.gyro = gyro

        self.r_bath = None
        self._bath = BathArray(0)
        self.imap = None

        if bath is not None:
            self.read_bath(bath, **bath_kw)

        self.r_dipole = r_dipole

        self.order = order
        self.clusters = None
        if r_dipole is not None and order is not None:
            assert self.bath is not None, "Bath spins were not provided to compute clusters"
            self.generate_clusters(order, r_dipole)

    def __repr__(self):
        return f"""Simulator for spin-{self.spin}.
alpha: {self.alpha}
beta: {self.beta}
gyromagnetic ratio: {self.gyro} kHz * rad / G

Parameters of cluster expansion:
r_bath: {self.r_bath}
r_dipole: {self.r_dipole}
order: {self.order}

Bath consists of {self.bath.size} spins."""

    @property
    def alpha(self):
        """
        0 qubit state of the central spin in Sz basis
        """
        return self._alpha

    @alpha.setter
    def alpha(self, alpha_state):
        self._alpha = np.asarray(alpha_state, dtype=np.complex128)

    @property
    def beta(self):
        """
        1-qubit state of the central spin in Sz basis
        """
        return self._beta

    @beta.setter
    def beta(self, beta_state):
        self._beta = np.asarray(beta_state, dtype=np.complex128)

    @property
    def bath(self):
        """
        BathArray instance containing bath spins
        """
        return self._bath

    @bath.setter
    def bath(self, bath_array):
        try:
            self._bath = bath_array.view(BathArray)
        except AttributeError as e:
            print('Bath array should be ndarray or a subclass')
            raise e
        self.imap = None

    def eigenstates(self, magnetic_field, D=None, E=0, alpha=0, beta=1):
        """
        Compute central spin states as eigenstates of the central spin Hamiltonian

        :param magnetic_field: ndarray containing external magnetic field as (Sx, By, Bz)
        :type magnetic_field: ndarray with shape (3,)

        :param D: D (longitudinal splitting) parameter of central spin in ZFS tensor of central spin in rad * kHz
            **OR** total ZFS tensor
        :type D: float or ndarray with shape (3,3)

        :param E: E (transverse splitting) parameter of central spin in ZFS tensor of central spin in rad * kHz
        :type E: float

        :param alpha: index of the state to be considered as 0-qubit state in order of increasing energy
            (0 - lowest energy)
        :type alpha: int

        :param beta: int
            index of the state to be considered as 1-qubit state
        :type beta: int
        """
        if D is not None:
            self.zfs = zfs_tensor(D, E)

        hamilton = total_hamiltonian(BathArray(0), magnetic_field, self.zfs,
                                     central_spin=self.spin,
                                     central_gyro=self.gyro)
        en, eiv = np.linalg.eigh(hamilton)
        self.alpha = eiv[:, alpha]
        self.beta = eiv[:, beta]

    def read_bath(self, bath, r_bath=None,
                  skiprows=1,
                  external_bath=None,
                  hyperfine=None,
                  types=None,
                  error_range=0.2,
                  ext_r_bath=None,
                  imap=None):
        """
        Read spin bath into self.bath

        :param bath:

            Either:
            - Instance of BathArray class;
            - ndarray with ``dtype([('N', np.unicode_, 16), ('xyz', np.float64, (3,))])`` containing names
            of bath spins (same ones as stored in self.ntype) and positions of the spins in angstroms;
            - the name of the xyz text file containing 4 cols: name of the bath spin and xyz coordinates in A
        :type bath: ndarray or str

        :param r_bath: cutoff size of the spin bath
        :type r_bath: float

        :param skiprows: if ``bath`` is name of the file, this argument
            gives number of rows to skip in reading the file (default 1)
        :type skiprows: int, optional

        :param external_bath: BathArray containing spins read from DFT output (see ``pycce.io``)
        :type external_bath: BathArray, optional

        :param hyperfine:
            This argument tells the code how to generate hyperfine couplings.
            If (``hyperfine = None`` and all A in provided bath are 0) or (``hyperfine = 'pd'``),
            use point dipole approximation. Otherwise can be an instance of ``Cube`` object,
            or callable with signature:
            ``func(coord, gyro, central_gyro)``, where coord is array of the bath spin coordinate,
            gyro is the gyromagnetic ratio of bath spin,
            central_gyro is the gyromagnetic ratio of the central bath spin.
        :type hyperfine: str, func, or Cube instance, optional

        :param types: SpinDict or input to create one
            Contains either SpinTypes of the bath spins or tuples which will initialize those.
            See ``pycce.bath.SpinDict`` documentation for details
        :type types: SpinDict or input to create one

        :param error_range: maximum distance between positions in nspin and external
            bath to consider two positions the same (default 0.2)
        :type error_range: float, optional

        :param ext_r_bath: maximum distance from the central spins of the bath spins
            for which to use the DFT positions
        :type ext_r_bath: float, optional

        :param imap: InteractionMap
            Instance of InteractionMap class, containing interaction tensors for bath spins.
        :type imap: InteractionMap

        :return: Returns the view of ``Simulator.bath`` attribute, generated by the method
        """
        self._bath = None

        bath = read_xyz(bath, skiprows=skiprows, spin_types=types)

        self.r_bath = r_bath if r_bath is not None else self.r_bath

        if self.r_bath is not None:
            mask = np.linalg.norm(bath['xyz'] - np.asarray(self.position), axis=-1) < self.r_bath
            bath = bath[mask]
            if imap is not None:
                imap = imap.subspace(mask)

        if external_bath is not None and ext_r_bath is not None:
            where = np.linalg.norm(external_bath['xyz'] - self.position, axis=1) <= ext_r_bath
            external_bath = external_bath[where]

        if hyperfine == 'pd' or (hyperfine is None and np.all(bath['A'] == 0)):
            bath.from_point_dipole(self.position, gyro_e=self.gyro)

        elif isinstance(hyperfine, Cube):
            bath.from_cube(hyperfine, gyro_e=self.gyro)
        elif hyperfine:
            bath.from_function(hyperfine, gyro_e=self.gyro)

        if external_bath is not None:
            bath.update(external_bath, error_range=error_range, ignore_isotopes=True,
                        inplace=True)

        self.bath = bath
        self.imap = imap

        return self.bath

    def generate_clusters(self, order=None, r_dipole=None, r_inner=0, strong=False, ignore=None):
        """
        Generate set of clusters used in CCE calculations.

        :param order: maximum size of the cluster
        :type order: int

        :param r_dipole: maximum connectivity distance
        :type r_dipole: float

        :param r_inner: minimum connectivity distance
        :type r_inner: float

        :param strong: True -  generate only clusters with "strong" connectivity (all nodes should be interconnected).
            Default False
        :type strong: bool

        :param ignore: optional. If not None,
            includes the names of bath spins which are ignored in the cluster generation
        :type ignore: list or str

        :return: view of self.clusters. self.clusters is a dictionary
            with keys corresponding to size of the cluster.
            i.e. self.clusters[n] contains ndarray of shape (m, n),
            where m is the number of clusters of given size,
            n is the size of the cluster.
            Each row contains indexes of the bath spins included in the given cluster
        :rtype: dict
        """

        self.order = order if order is not None else self.order
        self.r_dipole = r_dipole if r_dipole is not None else self.r_dipole
        self.clusters = None

        clusters = generate_clusters(self.bath, self.r_dipole, self.order,
                                     r_inner=r_inner, strong=strong, ignore=ignore)

        self.clusters = clusters

        return self.clusters

    def cce_coherence(self, timespace, magnetic_field, N, as_delay=False, direct=False, states=None,
                      parallel=False, mean_field=True):
        """
        Compute coherence function L with conventional CCE.

        :param timespace: Time points at which compute density matrix
        :type timespace: ndarray with shape (n,)

        :param magnetic_field: magnetic field vector of form (Bx, By, Bz)
        :type magnetic_field: ndarray with shape (3,)

        :param N: number of pulses in CPMG sequence. Overrides pulse_sequence if provided
        :type N: int

        :param as_delay: True if time points are delay between pulses (for equispaced pulses),
            False if time points are total time. Ignored if pulse_sequence contains the time fractions
        :type as_delay: bool


        :param states: List of bath spin states. if len(shape) == 1, contains Iz projections of Iz eigenstates.
            Otherwise, contains array of initial dms of bath spins
        :type states: array_like

        :param direct: True if use direct approach (requires way more memory but might be more numerically stable).
            False if use memory efficient approach. Default False
        :type direct: bool

        :param parallel: True if parallelize calculation of cluster contributions over different mpi processes.
            Default False.
        :type  parallel: bool

        :param mean_field: Optional. Takes effect only if ``states`` keyword is provided
            True if include mean field effect of all
            Default False.
        :type  parallel: bool

        :return: Coherence function computed at the time points in listed in the timespace
        :rtype: ndarray with shape (n,)

        """

        projections_alpha = generate_projections(self.alpha)
        projections_beta = generate_projections(self.beta)

        if states is not None and mean_field:
            proj_states = project_bath_states(states)
        else:
            proj_states = None

        coherence = decorated_coherence_function(self.clusters, self.bath, projections_alpha, projections_beta,
                                                 magnetic_field, timespace, N, as_delay=as_delay, states=states,
                                                 projected_states=proj_states,
                                                 parallel=parallel, direct=direct, imap=self.imap)

        return coherence

    def gcce_dm(self, timespace, magnetic_field,
                N=None, D=None, E=0, pulse_sequence=None, as_delay=False, state=None,
                bath_states=None, mean_field=True, nbstates=100, seed=None, masked=True,
                normalized=None, parallel_states=False,
                fixstates=None, direct=False, parallel=False) -> np.ndarray:
        """
        Compute density matrix of the central spin using generalized CCE

        :param timespace: Time points at which compute density matrix
        :type timespace: ndarray with shape (n,)

        :param magnetic_field: magnetic field vector of form (Bx, By, Bz)
        :type magnetic_field: ndarray with shape (3,)

        :param N: number of pulses in CPMG sequence. Overrides pulse_sequence if provided
        :type N: int

        :param D: D (longitudinal splitting) parameter of central spin in ZFS tensor of central spin in rad * kHz
            **OR** total ZFS tensor
        :type D: float or ndarray with shape (3,3)

        :param E: E (transverse splitting) parameter of central spin in ZFS tensor of central spin in rad * kHz
        :type E: float

        :param pulse_sequence: sequence of the instantaneous ideal control pulses.
            ``pulse_sequence`` should have format of list with tuples,
            each tuple contains can contain two or three entries:
            1. axis the rotation is about;
            2. angle of rotation;
            3. (optional) fraction of the total time before this pulse is applied.
            If not provided, assumes even delay of CPMG sequence. Then total experiment is assumed to be:

            tau -- pulse -- 2tau -- pulse -- ... -- 2tau -- pulse -- tau

            Where tau is the delay between pulses.

            E.g. for Hahn-Echo the pulse_sequence can be defined as ``[('x', np.pi)]`` or ``[('x', np.pi, 0.5)]``.
            Note, that if fraction is provided the computation becomes less effective than without it.

        :type pulse_sequence: list

        :param as_delay: True if time points are delay between pulses (for equispaced pulses),
            False if time points are total time. Ignored if pulse_sequence contains the time fractions
        :type as_delay: bool

        :param state: Initial state of the central spin. Defaults to sqrt(1 / 2) * (state + beta) if not set
        :type state: ndarray with shape (2s+1,)

        :param bath_states: List of bath spin states. if len(shape) == 1, contains Iz projections of Iz eigenstates.
            Otherwise, contains array of initial dms of bath spins.
        :type bath_states: array_like

        :param mean_field: If True, add mean field corrections and sample over random bath states.
            If ``bath_states`` keyword is provided, then compute only for the given state with mean field corrections.
            Default True.
        :type mean_field: bool

        :param nbstates: Number or random bath states to sample over
        :type nbstates: int

        :param seed: Seed for random number generator, used in random bath states sampling.
        :type seed: int

        :param masked:  True if mask numerically unstable points (with density matrix elements > 1)
            in the averaging over bath states. Default True
        :type masked: bool

        :param normalized: which diagonal elements to renormalize, so the total sum of the diagonal elements is 1
        :type normalized: ndarray of bool

        :param parallel_states: True if to use MPI to parallelize the calculations of density matrix equally over
            present mpi processes. Compared to ``parallel`` keyword, when this argument is True each process is given
            a fraction of random bath states. This makes the implementation faster. Works best when the
            number of bath states is divisible by the number of processes
            ``nbstates % size == 0``
        :type parallel_states: bool

        :param fixstates: shows which bath states to fix. Each key is the index of bath spin,
            value - fixed Sz projection of the mixed state of nuclear spin
        :type fixstates: dict

        :param direct: True if use direct approach (requires way more memory but might be more numerically stable).
            False if use memory efficient approach. Default False
        :type direct: bool

        :param parallel: True if parallelize calculation of cluster contributions over different mpi processes.
            Default False.
        :type  parallel: bool

        :return: array of density matrix, where first dimension corresponds to the time space size and last two -
            density matrix dimensions
        :rtype: ndarray with shape(n, 2s + 1, 2s + 1)
        """

        if D is not None:
            self.zfs = zfs_tensor(D, E)

        if state is None:
            state = np.sqrt(1 / 2) * (self.alpha + self.beta)

        if N is not None:
            pulse_sequence = [('x', np.pi)] * N

        dm0 = np.tensordot(state, state, axes=0)

        if not mean_field:
            if bath_states is not None:
                bath_states = np.asarray(bath_states)

            H0 = total_hamiltonian(BathArray(0), magnetic_field, self.zfs,
                                   central_spin=self.spin, central_gyro=self.gyro)

            dms = compute_dm(dm0, H0, self.alpha, self.beta, timespace, pulse_sequence, as_delay=as_delay)
            dms = ma.masked_array(dms, mask=(dms == 0), fill_value=0j, dtype=np.complex128)

            dms *= decorated_density_matrix(self.clusters, self.bath, dm0, self.alpha, self.beta, magnetic_field,
                                            self.zfs, timespace, pulse_sequence, gyro_e=self.gyro, as_delay=as_delay,
                                            zeroth_cluster=dms, bath_state=bath_states, imap=self.imap,
                                            direct=direct, parallel=parallel)

        else:
            if bath_states is not None:
                proj_bath_states = project_bath_states(bath_states)
                H0 = mean_field_hamiltonian(BathArray(0), magnetic_field, self.bath, bath_states, D=self.zfs,
                                            central_gyro=self.gyro)

                dms = compute_dm(dm0, H0, self.alpha, self.beta, timespace, pulse_sequence, as_delay=as_delay)
                dms = ma.masked_array(dms, mask=(dms == 0), fill_value=0j, dtype=np.complex128)

                dms *= mean_field_density_matrix(self.clusters, self.bath, dm0, self.alpha, self.beta,
                                                 magnetic_field, self.zfs, timespace,
                                                 pulse_sequence, bath_states, projected_bath_state=proj_bath_states,
                                                 gyro_e=self.gyro, as_delay=as_delay, zeroth_cluster=dms,
                                                 imap=self.imap)

            else:
                dms = monte_carlo_sampling(self.clusters, self.bath, dm0, self.alpha, self.beta, magnetic_field,
                                           self.zfs, timespace, pulse_sequence, central_gyro=self.gyro,
                                           as_delay=as_delay, imap=self.imap, nbstates=nbstates, seed=seed,
                                           masked=masked, normalized=normalized, parallel_states=parallel_states,
                                           fixstates=fixstates, direct=direct, parallel=parallel)
        return dms

    def gcce_noise(self, timespace, magnetic_field, D=None, E=0, state=None,
                   nbstates=100, seed=None, parallel_states=False, mean_field=False,
                   direct=False, parallel=False):
        """
        EXPERIMENTAL Compute noise auto correlation function
        using generalized CCE with Monte-Carlo bath state sampling
        :param timespace: Time points at which compute density matrix
        :type timespace: ndarray with shape (n,)

        :param magnetic_field: magnetic field vector of form (Bx, By, Bz)
        :type magnetic_field: ndarray with shape (3,)

        :param N: number of pulses in CPMG sequence. Overrides pulse_sequence if provided
        :type N: int

        :param D: D (longitudinal splitting) parameter of central spin in ZFS tensor of central spin in rad * kHz
            **OR** total ZFS tensor
        :type D: float or ndarray with shape (3,3)

        :param E: E (transverse splitting) parameter of central spin in ZFS tensor of central spin in rad * kHz
        :type E: float

        :param state: Initial state of the central spin. Defaults to sqrt(1 / 2) * (state + beta) if not set
        :type state: ndarray with shape (2s+1,)

        :param mean_field: If True, add mean field corrections and sample over random bath states.
            Default False.
        :type mean_field: bool

        :param nbstates: Number or random bath states to sample over
        :type nbstates: int

        :param seed: Seed for random number generator, used in random bath states sampling.
        :type seed: int

        :param masked:  True if mask numerically unstable points (with density matrix elements > 1)
            in the averaging over bath states. Default True
        :type masked: bool

        :param normalized: which diagonal elements to renormalize, so the total sum of the diagonal elements is 1
        :type normalized: ndarray of bool

        :param parallel_states: True if to use MPI to parallelize the calculations of density matrix equally over
            present mpi processes. Compared to ``parallel`` keyword, when this argument is True each process is given
            a fraction of random bath states. This makes the implementation faster, however works best when
            number of bath states is divisible by the number of processes
            ``nbstates % size == 0``
        :type parallel_states: bool

        :param direct: True if use direct approach (requires way more memory but might be more numerically stable).
            False if use memory efficient approach. Default False
        :type direct: bool

        :param parallel: True if parallelize calculation of cluster contributions over different mpi processes.
            Default False.
        :type  parallel: bool

        :return: Autocorrelation function of the noise, in (kHz*rad)^2
        :rtype: ndarray with shape (n,)
        """
        if D is not None:
            self.zfs = zfs_tensor(D, E)

        if state is None:
            state = np.sqrt(1 / 2) * (self.alpha + self.beta)

        dm0 = np.tensordot(state, state, axes=0)
        if mean_field:
            corr = noise_sampling(self.clusters, self.bath, dm0, timespace, magnetic_field, self.zfs,
                                  gyro_e=self.gyro, imap=self.imap,
                                  nbstates=nbstates, seed=seed, parallel_states=parallel_states,
                                  direct=direct, parallel=parallel)
        else:
            corr = decorated_noise_correlation(self.clusters, self.bath, dm0, magnetic_field, self.zfs, timespace,
                                               gyro_e=self.gyro,
                                               parallel=parallel, direct=direct)
        return corr

    def cce_noise(self, timespace, magnetic_field, state=None, parallel=False, direct=False):
        """
        Compute noise autocorrelation function of the noise with conventional CCE

        :param timespace: Time points at which compute density matrix
        :type timespace: ndarray with shape (n,)

        :param magnetic_field: magnetic field vector of form (Bx, By, Bz)
        :type magnetic_field: ndarray with shape (3,)

        :param state: Initial state of the central spin. Defaults to sqrt(1 / 2) * (state + beta) if not set
        :type state: ndarray with shape (2s+1,)

        :param direct: True if use direct approach (requires way more memory but might be more numerically stable).
            False if use memory efficient approach. Default False
        :type direct: bool

        :param parallel: True if parallelize calculation of cluster contributions over different mpi processes.
            Default False.
        :type  parallel: bool

        :return: Autocorrelation function of the noise, in (kHz*rad)^2
        :rtype: ndarray with shape (n,)s
        """
        magnetic_field = np.asarray(magnetic_field)

        if state is None:
            state = np.sqrt(1 / 2) * (self.alpha + self.beta)
        else:
            state = np.asarray(state)

        projections_state = generate_projections(state)
        corr = projected_noise_correlation(self.clusters, self.bath, projections_state, magnetic_field, timespace,
                                           parallel=parallel, direct=direct)

        return corr

    def compute(self, *arg, quantity='coherence', method='cce', **kwarg):
        """
        General interface for computing properties with CCE

        :param arg:
            Position arguments for the function to be used

        :param quantity: Which quantity to compute. Possible values:
                'coherence' - compute coherence function
                'dm' - compute full density matrix
                'noise' - compute noise autocorrelation function
        :type quantity: str

        :param method: Which implementation of CCE to use. Possible values:
                'CCE' - conventional CCE, where interactions are mapped on 2 level pseudospin
                'gCCE' - generalized CCE with or without random bath state sampling (given by ``mean_field`` keyword)
        :type method: str

        :param kwarg: keyword arguments for the function to be used

        :return: computed property
        :rtype: np.ndarray
        """
        func = getattr(self, self._compute_func[method.lower()][quantity.lower()])
        result = func(*arg, **kwarg)

        if (quantity == 'coherence') and (len(result.shape) != 1):
            try:
                result = result.filled(0j)
            except AttributeError:
                pass

            dm0 = result[0]
            state = np.sqrt(1 / 2) * (self.alpha + self.beta)
            dm0expected = np.tensordot(state, state, axes=0).astype(np.complex128)

            if not np.isclose(dm0, dm0expected).all():
                warnings.warn('Initial qubit state is not superposition of alpha and beta states. '
                              'The coherence might yield unexpected results')
            result = self.alpha @ result @ self.beta
            result = result / result[0]

        return result

    _compute_func = {
        'cce': {
            'coherence': 'cce_coherence',
            'noise': 'cce_noise'
        },
        'gcce': {
            'coherence': 'gcce_dm',
            'dm': 'gcce_dm',
            'noise': 'gcce_noise',
        }
    }
