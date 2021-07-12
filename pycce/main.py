"""
Default Units:

- Distance: Angstrom, A
- Time: Millisecond, ms
- Magnetic Field: Gaussian, G = 1e-4 Tesla
- Gyromagnetic Ratio: rad / ms / Gauss
- Quadrupole moment: barn
- Couplings: kHz

"""
import warnings

import numpy as np
from pycce.io.xyz import read_xyz

from .bath.array import BathArray, SpinDict
from .bath.cube import Cube
from .constants import ELECTRON_GYRO
from .find_clusters import generate_clusters
from .h import total_hamiltonian
from .run.cce import CCE
from .run.corr import CCENoise, gCCENoise
from .run.gcce import gCCE
from .run.pulses import Sequence
from .utilities import zfs_tensor, project_bath_states


def _add_args(after):
    def prepare_with_args(func):
        func.__doc__ = func.__doc__ + after
        return func

    return prepare_with_args


_returns = r"""

            Returns:
    
                ndarray: Computed property."""

_args = r"""
            magnetic_field (ndarray with shape (3,)): Magnetic field vector of form (Bx, By, Bz). 
    
                Default is **None**. Overrides  ``Simulator.magnetic_field`` if provided.

            D (float or ndarray with shape (3,3)):
                D (longitudinal splitting) parameter of central spin in ZFS tensor of central spin in kHz.
    
                **OR**
    
                Total ZFS tensor.
                
                Default is **None**. Overrides``Simulator.zfs`` if provided.
    
            E (float): E (transverse splitting) parameter of central spin in ZFS tensor of central spin in kHz.
                Ignored if ``D`` is None or tensor.
                
                Default is 0.

            pulses (list or int or Sequence): Number of pulses in CPMG sequence.

                **OR**

                Sequence of the instantaneous ideal control pulses.
                It can be provided as an instance of ``Sequence`` class
                (See documentation for pycce.Sequence).
                
                ``pulses`` can be provided as a list with tuples or dictionaries,
                each tuple or dictionary is used to initialize ``Pulse`` class instance.

                For example, for only central spin pulses the ``pulses`` argument
                can be provided as a list of tuples, containing:

                1. axis the rotation is about;
                2. angle of rotation;
                3. (optional) Time before the pulse. Can be as fixed, as well as varied.
                   If varied, it should be provided as an array with the same
                   length as ``timespace``.

                E.g. for Hahn-Echo the ``pulses`` can be defined as ``[('x', np.pi)]`` or 
                ``[('x', np.pi, timespace / 2)]``.

                .. note::
                
                    If delay is not provided in **all** pulses, assumes even delay of CPMG sequence. 
                    If only **some** delays are provided, assumes ``delay = 0``  in the pulses without delay.

                    Then total experiment is assumed to be:

                        tau -- pulse -- 2tau -- pulse -- ... -- 2tau -- pulse -- tau

                    Where tau is the delay between pulses.

                    The sum of delays at each time point should be less or equal to the total time of the experiment
                    at the same time point, provided in ``timespace`` argument. 

                .. warning::
    
                    In conventional CCE calculations, only :math:`pi` pulses
                    on the central spin are allowed. 

                In the calculations of noise autocorrelation this parameter is ignored.

                Default is **None**. Overrides``Simulator.pulses`` if provided.

            alpha (int or ndarray with shape (2s+1, )): :math:`\ket{0}` state of the qubit in :math:`S_z`
                basis or the index of eigenstate to be used as one.

                Default is **None**. Overrides``Simulator.alpha`` if provided.

            beta (int or ndarray with shape (2s+1, )): :math:`\ket{1}` state of the qubit in :math:`S_z` basis
                or the index of the eigenstate to be used as one.

                Default is **None**. Overrides``Simulator.beta`` if provided.
    
            as_delay (bool): True if time points are delay between pulses (for equispaced pulses),
                False if time points are total time. Ignored in gCCE if ``pulses`` contains the time fractions.
                Conventional CCE calculations do not support custom time fractions.

                Default is **False**.

            interlaced (bool): True if use hybrid CCE approach - for each cluster 
                sample over states of the supercluster.

                Default is **False**.

            state (ndarray with shape (2s+1,)):
                Initial state of the central spin, used in gCCE and noise autocorrelation calculations.
                
                Defaults to :math:`\frac{1}{N}(\ket{0} + \ket{1})` if not set.
    
            bath_state (array-like):
                List of bath spin states. If ``len(shape) == 1``, contains
                :math:`I_z` projections of :math:`I_z` eigenstates.
                Otherwise, contains array of initial density matrices of bath spins.
    
                Default is **None**. If not set, the code assumes completely random spin bath
                (density matrix of each nuclear spin is proportional to identity, :math:`\hat {\mathbb{1}}/N`).

            nbstates (int): Number or random bath states to sample over.

                If provided, sampling of random states is carried and ``bath_states`` values are
                ignored.
                    
                Default is 0.

            seed (int): Seed for random number generator, used in random bath states sampling.
                
                Default is **None**.

            masked (bool):
                True if mask numerically unstable points (with coherence > 1)
                in the averaging over bath states. 
                
                .. note::
                
                    It is up to user to check whether the possible instability is due to numerical error
                    or unphysical assumptions of the calculations.

                Default is **True** for coherence calculations, **False** for noise calculations.

            parallel_states (bool):
                True if to use MPI to parallelize the calculations of density matrix equally over
                present mpi processes for random bath state sampling calculations.
                
                Compared to ``parallel`` keyword,
                when this argument is True each process is given a fraction of random bath states.
                This makes the implementation faster. Works best when the
                number of bath states is divisible by the number of processes, ``nbstates % size == 0``.
    
                Default is **False**.

            fixstates (dict): If not None, shows which bath states to fix in random bath states.
                Each key is the index of bath spin,
                value - fixed :math:`\hat S_z` projection of the mixed state of nuclear spin.

                Default is **None**.

            second_order (bool):
                True if add second order perturbation theory correction
                to the cluster Hamiltonian in conventional CCE.
                Relevant only for conventional CCE calculations.
                
                If set to True sets the qubit states as eigenstates of central spin Hamiltonian
                from the following procedure. If qubit states are provided as vectors in :math:`S_z` basis,
                for each qubit state compute the fidelity of the qubit state and
                all eigenstates of the central spin and chose the one with fidelity higher than ``level_confidence``.
                If such state is not found, raises an error.
    
                .. warning::
    
                    Second order corrections are not implemented as mean field.
                    
                    I.e., setting ``second_order=True``
                    and ``nbstates != 0`` leads to the calculation, when mean field effect is accounted only from
                    dipolar interactions within the bath.
    
                Default is **False**.
    
            level_confidence (float): Maximum fidelity of the qubit state to be considered eigenstate of the
                central spin Hamiltonian. 
                
                Default is 0.95.
    
            direct (bool):
                True if use direct approach (requires way more memory but might be more numerically stable).
                False if use memory efficient approach.
                
                Default is **False**.
    
            parallel (bool):
                True if parallelize calculation of cluster contributions over different mpi processes.

                Default is **False**."""


class Environment:
    def __init__(self, position, gyro, *arg, **kwarg):
        # Properties of the central spin
        self.position = position
        self.gyro = gyro

        # Properties of the total bath
        self.total_bath = None
        # self.total_imap = None
        # The reduced bath
        self._r_bath = None
        self._bath = None
        # self.imap = None
        self._hyperfine = None
        # External bath properties
        self._external_bath = None
        self._ext_r_bath = None
        self._error_range = 0.2

        self.read_bath(*arg, **kwarg)

    @property
    def r_bath(self):
        """
        float: Cutoff size of the spin bath.
        """
        return self._r_bath

    @r_bath.setter
    def r_bath(self, r_bath):
        self.read_bath(r_bath=r_bath)

    @property
    def external_bath(self):
        """
        BathArray: Array with spins read from DFT output (see ``pycce.io``).
        """
        return self._external_bath

    @external_bath.setter
    def external_bath(self, external_bath):
        self.read_bath(external_bath=external_bath)

    @property
    def ext_r_bath(self):
        """
        float: Maximum distance from the central spins of the bath spins
            for which to use the data from ``external_bath``.
        """
        return self._ext_r_bath

    @ext_r_bath.setter
    def ext_r_bath(self, ext_r_bath):
        self.read_bath(ext_r_bath=ext_r_bath)

    @property
    def error_range(self):
        """
        float: Maximum distance between positions in bath and external
            bath to consider two positions the same (default 0.2).
        """
        return self._error_range

    @error_range.setter
    def error_range(self, error_range):
        self.read_bath(error_range=error_range)

    @property
    def hyperfine(self):
        """
        str, func, or Cube instance: This argument tells the code how to generate hyperfine couplings.
            If (``hyperfine = None`` and all A in provided bath are 0) or (``hyperfine = 'pd'``),
            use point dipole approximation. Otherwise can be an instance of ``Cube`` object,
            or callable with signature:

                ``func(coord, gyro, central_gyro)``

            where coord is array of the bath spin coordinate, gyro is the gyromagnetic ratio of bath spin,
            central_gyro is the gyromagnetic ratio of the central bath spin.
        """
        return self._hyperfine

    @hyperfine.setter
    def hyperfine(self, hyperfine):
        self.read_bath(hyperfine=hyperfine)

    @property
    def bath(self):
        """
        BathArray: Array of bath spins used in CCE simulations.
        """
        return self._bath

    @bath.setter
    def bath(self, bath_array):
        try:
            self._bath = bath_array.view(BathArray)
        except AttributeError as e:
            print('Bath array should be ndarray or a subclass')
            raise e

        self.total_bath = self._bath
        # self.imap = None
        # self.total_imap = None
        self._r_bath = None
        self.external_bath = None
        self._ext_r_bath = None
        self.hyperfine = None

    def read_bath(self, bath=None, r_bath=None,
                  skiprows=1,
                  external_bath=None,
                  hyperfine=None,
                  types=None,
                  error_range=None,
                  ext_r_bath=None,
                  imap=None):
        r"""
        Read spin bath from the file or from the ``BathArray``.

        Args:
            bath (ndarray, BathArray or str): Either:

                * Instance of BathArray class;
                * ndarray with ``dtype([('N', np.unicode_, 16), ('xyz', np.float64, (3,))])`` containing names
                  of bath spins (same ones as stored in self.ntype) and positions of the spins in angstroms;
                * the name of the xyz text file containing 4 cols: name of the bath spin and xyz coordinates in A.

            r_bath (float): Cutoff size of the spin bath.

            skiprows (int, optional): If ``bath`` is name of the file, this argument
                gives number of rows to skip while reading the .xyz file (default 1).

            external_bath (BathArray, optional):
                BathArray containing spins read from DFT output (see ``pycce.io``).

            hyperfine (str, func, or Cube instance, optional):
                This argument tells the code how to generate hyperfine couplings.

                If (``hyperfine = None`` and all A in provided bath are 0) or (``hyperfine = 'pd'``),
                use point dipole approximation.

                Otherwise can be an instance of ``Cube`` object,
                or callable with signature:
                ``func(coord, gyro, central_gyro)``, where coord is array of the bath spin coordinate,
                gyro is the gyromagnetic ratio of bath spin,
                central_gyro is the gyromagnetic ratio of the central bath spin.

            types (SpinDict): SpinDict or input to create one.
                Contains either SpinTypes of the bath spins or tuples which will initialize those.

                See ``pycce.bath.SpinDict`` documentation for details.

            error_range (float, optional): Maximum distance between positions in bath and external
                bath to consider two positions the same (default 0.2).

            ext_r_bath (float, optional): Maximum distance from the central spins of the bath spins
                for which to use the DFT positions.

            imap (InteractionMap): Instance of ``InteractionMap`` class, containing interaction tensors for bath spins.
                Each key of the ``InteractionMap`` is a tuple with indexes of two bath spins.
                The value is the 3x3 tensor describing the interaction between two spins in a format:

                .. math::

                    I^iJI^j = I^i_{x}J_{xx}I^j_{x} + I^i_{x}J_{xy}I^j_{y} ...

        .. note::

            For each bath spin pair without interaction tensor in ``imap``, coupling is approximated assuming
            magnetic point dipole–dipole interaction.  If ``imap = None`` all interactions between bath spins
            are approximated in this way. Then interaction tensor between spins `i` and `j` is computed as:

            .. math::

                \mathbf{J}_{ij} = -\gamma_{i} \gamma_{j} \frac{\hbar^2}{4\pi \mu_0}
                                   \left[ \frac{3 \vec{r}_{ij} \otimes \vec{r}_{ij} - |r_{ij}|^2 I}{|r_{ij}|^5} \right]

            Where :math:`\gamma_{i}` is gyromagnetic ratio of `i` spin, :math:`I` is 3x3 identity matrix, and
            :math:`\vec{r}_{ij}` is distance between two spins.

        Returns:
            BathArray: The view of ``Simulator.bath`` attribute, generated by the method.
        """

        self._bath = None

        if bath is not None:
            self.total_bath = read_xyz(bath, skiprows=skiprows, spin_types=types, imap=imap)

        bath = self.total_bath

        if bath is None:
            return

        self._r_bath = r_bath if r_bath is not None else self._r_bath
        # self.total_imap = imap if imap is not None else self.total_imap
        self._external_bath = external_bath if external_bath is not None else self._external_bath
        self._ext_r_bath = ext_r_bath if ext_r_bath is not None else self._ext_r_bath
        self._hyperfine = hyperfine if hyperfine is not None else self._hyperfine
        self._error_range = error_range if error_range is not None else self._error_range

        # imap = self.total_imap

        if self.r_bath is not None:
            mask = np.linalg.norm(bath['xyz'] - np.asarray(self.position), axis=-1) < self.r_bath
            bath = bath[mask]
            # if self.total_imap is not None:
            #     imap = self.total_imap.subspace(mask)

        if self.external_bath is not None and self.ext_r_bath is not None:
            where = np.linalg.norm(self.external_bath['xyz'] - self.position, axis=1) <= self.ext_r_bath
            external_bath = self.external_bath[where]

        if self._hyperfine == 'pd' or (self._hyperfine is None and not np.any(bath['A'])):
            bath.from_point_dipole(self.position, gyro_e=self.gyro)
            self._hyperfine = 'pd'

        elif isinstance(self._hyperfine, Cube):
            bath.from_cube(self._hyperfine, gyro_e=self.gyro)

        elif self._hyperfine:
            bath.from_function(self._hyperfine, gyro_e=self.gyro)

        if external_bath is not None:
            bath.update(external_bath, error_range=self._error_range, ignore_isotopes=True,
                        inplace=True)

        if not bath.size:
            warnings.warn('Empty bath is provided.', stacklevel=2)

        self._bath = bath
        # self.imap = imap

        return self.bath


# TODO unit conversion
class Simulator(Environment):
    r"""
    The main class for CCE calculations.

    The typical usage includes:

        1. Read array of the bath spins. This is done with ``Simulator.read_bath`` method which accepts either
           reading from .xyz file or from the ``BathArray`` instance with defined positions and names of the bath spins.
           In the process, the subset of the array within the distance of ``r_dipole`` from the central spin is taken
           and for this subset the Hyperfine couplings can be generated.

           If no ``hyperfine`` keyword is provided and there are some hyperfine couplings already,
           then no changes are done to the hyperfine tensors.
           If ``hyperfine='pd'``, the hyperfine couplings are computed assuming point dipole approximation. For all
           accepted arguments, see ``Simulator.read_bath``.

        2. Generate set of clusters with ``Simulator.generate_clusters``,
           determined by the maximum connectivity radius ``r_dipole``
           and the maximum size of the cluster ``order``.

        3. Compute the desired property with ``Simulator.compute`` method.

    .. note::

        Directly setting up the attribute values will rerun ``Simulator.read_bath``
        and/or ``Simulator.generate_clusters`` to reflect updated value of the given attribute.

        E.g. If ``Simulator.r_bath`` is set to some new value after initialization,
        then ``Simulator.read_bath`` and ``Simulator.generate_clusters`` are called with the increased bath.

    First two steps are usually done during the initialization of the ``Simulator`` object by providing the necessary
    arguments.

    Notes:

        Depending on the number of provided arguments, in the initialization process will call the following methods
        to setup the calculation engine.

        - If ``bath`` is provided, ``Simulator.read_bath`` is called with additional keywords in ``**bath_kw``.
        - If both ``r_dipole`` and ``order`` are provided and ``bath`` is not None,
          the ``Simulator.generate_clusters`` is called.

        See the corresponding method documentation for details.

    Examples::

        >>> atoms = random_bath('13C', 100, number=2000, seed=10)
        >>> calc = Simulator(1, bath=atoms, r_bath=40, r_dipole=6,
        >>>                  order=2, D=2.88 * 2 * np.pi * 1e6,
        >>>                  magnetic_field=500, pulses=1)
        >>> print(calc)
        Simulator for spin-1.
        alpha: [0.+0.j 1.+0.j 0.+0.j]
        beta: [0.+0.j 0.+0.j 1.+0.j]
        gyromagnetic ratio: -17608.59705 kHz * rad / G
        zero field splitting:
        array([[-6031857.895,        0.   ,        0.   ],
               [       0.   , -6031857.895,        0.   ],
               [       0.   ,        0.   , 12063715.79 ]])
        magnetic field:
        array([  0.,   0., 500.])

        Parameters of cluster expansion:
        r_bath: 40
        r_dipole: 6
        order: 2

        Bath consists of 549 spins.

        Clusters include:
        549  clusters of order 1.
        457  clusters of order 2.


    Args:

        spin (float): Total spin of the central spin.

        position (ndarray): Cartesian coordinates in Angstrom of the central spin. Default (0., 0., 0.).

        alpha (float or ndarray with shape (2s+1, )): :math:`\ket{0}` state of the qubit in :math:`S_z`
            basis or the index of eigenstate to be used as one.

            Default: state with :math:`m_s = +s` where :math:`m_s` is the z-projection of the spin
            and :math:`s` is the total spin if no information of central spin Hamiltonian is provided.
            Otherwise lowest energy eigenstate of the central spin Hamiltonian.

        beta (float or ndarray with shape (2s+1, )): :math:`\ket{1}` state of the qubit in :math:`S_z` basis
            or the index of the eigenstate to be used as one.

            Default: state with :math:`m_s = +s - 1` where :math:`m_s` is the z-projection of the spin
            and :math:`s` is the total spin if no information of central spin Hamiltonian is provided.
            Otherwise second lowest energy eigenstate of the central spin Hamiltonian.

        gyro (float or ndarray with shape (3,3)): Gyromagnetic ratio of central spin in rad / ms / G.

            *OR*

            Tensor describing central spin interactions with the magnetic field.

            Default -17608.597050 kHz * rad / G - gyromagnetic ratio of the free electron spin.

        D (float or ndarray with shape (3, 3)): D (longitudinal splitting) parameter of central spin
            in ZFS tensor of central spin in kHz.

            *OR*

            Total ZFS tensor. Default 0.

        E (float): E (transverse splitting) parameter of central spin in ZFS tensor of central spin in kHz.
            Default 0. Ignored if ``D`` is None or tensor.

        bath (ndarray or str): First positional argument of the ``Simulator.read_bath`` method.

            Either:

            - Instance of BathArray class;
            - ndarray with ``dtype([('N', np.unicode_, 16), ('xyz', np.float64, (3,))])`` containing names
              of bath spins (same ones as stored in self.ntype) and positions of the spins in angstroms;
            - the name of the xyz text file containing 4 cols: name of the bath spin and xyz coordinates in A.

        r_dipole (float): Maximum connectivity distance between two bath spins.

        order (int): Maximum size of the cluster to be considered in CCE expansion.

        pulses (list or int): Number of pulses in CPMG sequence or list with pulses.

        **bath_kw: Additional keyword arguments for the ``Simulator.read_bath`` method.

    """

    def __init__(self, spin, position=None, alpha=None, beta=None, gyro=ELECTRON_GYRO, magnetic_field=None,
                 D=0., E=0., r_dipole=None, order=None, bath=None, pulses=None, as_delay=False,
                 **bath_kw):

        if position is None:
            position = np.zeros(3)

        self.position = np.asarray(position, dtype=np.float64)
        """ndarray with shape (3, ): Position of the central spin in Cartesian coordinates."""
        self.spin = spin
        """float: Value of the central spin s."""
        self.gyro = gyro
        """gyro (float or ndarray with shape (3,3)): Gyromagnetic ratio of central spin in rad / ms / G.

        *OR*

        Tensor describing central spin interactions with the magnetic field.

        Default -17608.597050 kHz * rad / G - gyromagnetic ratio of the free electron spin."""
        self.zfs = zfs_tensor(D, E)
        """ndarray with shape (3,3): Zero field splitting tensor of the central spin"""
        self._magnetic_field = None
        self.magnetic_field = magnetic_field
        self.set_states(alpha, beta)

        self._r_dipole = r_dipole

        self._order = order
        self.clusters = None
        """dict: Dictionary containing information about cluster structure of the bath.
        
        Each keys n correspond to the size of the cluster.
        Each ``Simulator.clusters[n]`` contains ``ndarray`` of shape (m, n),
        where m is the number of clusters of given size, n is the size of the cluster.
        Each row  of this array contains indexes of the bath spins included in the given cluster.
        Generated during ``.generate_clusters`` call."""
        # Bath setting up
        super().__init__(self.position, self.gyro, bath, **bath_kw)

        # Parameters of the calculations
        self.pulses = pulses

        self.as_delay = as_delay
        """bool: True if time points are delay between pulses (for equispaced pulses),
        False if time points are total time. Ignored if ``pulses`` contains the time delays."""
        # Initial entangled state of the qubit
        self.state = None
        """
        ndarray: Innitial state of the qubit in gCCE simulations.
            Assumed to be :math:`1/\sqrt{2}(\ket{0} + \ket{1}` unless provided during ``Simulator.compute`` call."""

        # hybrid CCE
        self.interlaced = False
        """bool: True if use hybrid CCE approach - for each cluster sample over states of the supercluster."""
        # Parameters of MC states
        self.seed = None
        """int: Seed for random number generator, used in random bath states sampling."""
        self.nbstates = None
        """int: Number or random bath states to sample over."""
        self.fixstates = None
        """dict: If not None, shows which bath states to fix in random bath states.
        
        Each key is the index of bath spin,
        value - fixed :math:`\hat S_z` projection of the mixed state of nuclear spin."""
        self.masked = None
        """bool: True if mask numerically unstable points (with coherence > 1)
        in the averaging over bath states. 

        .. note::

            It is up to user to check whether the possible instability is due to numerical error
            or unphysical assumptions of the calculations."""
        # Parameters of conventional CCE
        self.second_order = None
        """bool: True if add second order perturbation theory correction to the cluster Hamiltonian in conventional CCE.
        Relevant only for conventional CCE calculations."""
        self.level_confidence = None
        """float: Maximum fidelity of the qubit state to be considered eigenstate of the
        central spin Hamiltonian when ``second_order`` set to True."""
        self.projected_bath_state = None
        """ndarray with shape (n,): Array with z-projections of the bath spins states."""
        self.bath_state = None
        """bath_state (ndarray): Array of bath states."""
        self.timespace = None
        """timespace (ndarray with shape (n,)): Time points at which compute the desired property."""

    def __repr__(self):
        bm = (f"Simulator for spin-{self.spin}.\n"
              f"alpha: {self.alpha}\n"
              f"beta: {self.beta}\n"
              f"gyromagnetic ratio: {self.gyro} kHz * rad / G\n"
              f"zero field splitting:\n{self.zfs.__repr__()}\n"
              f"magnetic field:\n{self.magnetic_field.__repr__()}\n\n"
              f"Parameters of cluster expansion:\n"
              f"r_bath: {self.r_bath}\n"
              f"r_dipole: {self.r_dipole}\n"
              f"order: {self.order}\n"
              )

        if self.bath is not None:
            bm += f"\nBath consists of {self.bath.size} spins.\n"

        if self.clusters is not None:
            m = "\nClusters include:\n"
            for k in sorted(self.clusters.keys()):
                m += f"{self.clusters[k].shape[0]}  clusters of order {k}.\n"
            bm += m

        return bm

    @property
    def alpha(self):
        r"""
        ndarray or int: :math:`\ket{0}` qubit state of the central spin in Sz basis
            **OR** index of the energy state to be considered as one.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, alpha_state):
        try:
            if len(alpha_state) > 1:
                self._alpha = np.asarray(alpha_state, dtype=np.complex128)
            else:
                self._alpha = np.asarray(alpha_state, dtype=np.int32)
        except TypeError:
            self._alpha = np.asarray(alpha_state, dtype=np.int32)

    @property
    def beta(self):
        r"""
        ndarray or int: :math:`\ket{1}` qubit state of the central spin in Sz basis
            **OR** index of the energy state to be considered as one.
        """
        return self._beta

    @beta.setter
    def beta(self, beta_state):
        try:
            if len(beta_state) > 1:
                self._beta = np.asarray(beta_state, dtype=np.complex128)
            else:
                self._beta = np.int(beta_state)
        except TypeError:
            self._beta = np.int(beta_state)

    @property
    def magnetic_field(self):
        """
        ndarray: Array containing external magnetic field as (Bx, By, Bz). Default (0, 0, 0).
        """
        return self._magnetic_field

    @magnetic_field.setter
    def magnetic_field(self, magnetic_field):
        self.set_magnetic_field(magnetic_field)

    @property
    def order(self):
        """
        int: Maximum size of the cluster.
        """
        return self._order

    @order.setter
    def order(self, order):
        self.generate_clusters(order=order)

    @property
    def r_dipole(self):
        """
        float: Maximum connectivity distance.
        """
        return self._r_dipole

    @r_dipole.setter
    def r_dipole(self, r_dipole):
        self.generate_clusters(r_dipole=r_dipole)

    @property
    def pulses(self):
        """
        Sequence: List-like object, containing the sequence of the instantaneous ideal control pulses.

        Each item is ``Pulse`` object, containing the following attributes:

        * **axis** (*str*): Axis of rotation of the central spin. Can be 'x', 'y', or 'z'.

        * **angle** (*float or str*): Angle of rotation of central spin.
          Can be provided in rad, or as a string, containing
          fraction of pi: ``'pi'``, ``'pi/2'``, ``'2*pi'`` etc. Default is None.

        * **delay** (*float or ndarray*): Delay before the pulse or array of delays
          with the same shape as time points.

        * **bath_names** (*str or array-like of str*): Name or array of names of bath spin types,
          impacted by the bath pulse.

        * **bath_axes** (*str or array-like of str*): Axis of rotation or array of axes of the bath spins.
          If ``bath_names`` is provided, but ``bath_axes`` and ``bath_angles`` are not,
          assumes the same axis and angle as the one of the central spin.

        * **bath_angles** (*float or str or array-like*): Angle of rotation or array of axes
          of rotations of the bath spins.

        If delay is not provided in **all** pulses, assumes even delay of CPMG sequence.

        If only **some** delays are provided, assumes 0 delay in the pulses without delay provided.
        """
        return self._pulses

    @pulses.setter
    def pulses(self, pulses):

        try:
            pulses = ([('x', np.pi), ('y', np.pi)] * ((pulses + 1) // 2))[:pulses]
        except TypeError:
            pulses = pulses

        self._pulses = Sequence(pulses)

    def set_zfs(self, D=None, E=0):
        """
        Set Zero Field Splitting of the central spin from longitudinal ZFS *D* and transverse ZFS *E*.

        Args:
            D (float or ndarray with shape (3, 3)): D (longitudinal splitting) parameter of central spin
                in ZFS tensor of central spin in kHz.

                **OR**

                Total ZFS tensor. Default 0.

            E (float): E (transverse splitting) parameter of central spin in ZFS tensor of central spin in kHz.
                 Default 0. Ignored if ``D`` is None or tensor.
        """

        self.zfs = zfs_tensor(D, E)

    def set_magnetic_field(self, magnetic_field=None):
        """
        Set magnetic field from either value of the magnetic field along z-direction or full magnetic field vector.

        Args:
            magnetic_field (float or array-like): Magnetic field along z-axis.

                **OR**

                Array containing external magnetic field as (Bx, By, Bz). Default (0, 0, 0).

        """
        if magnetic_field is None:
            magnetic_field = 0

        magnetic_field = np.asarray(magnetic_field, dtype=np.float64)

        if magnetic_field.size == 1:
            magnetic_field = np.array([0, 0, magnetic_field.flatten()[0]])

        assert magnetic_field.size == 3, "Improper magnetic field format."

        self._magnetic_field = magnetic_field

    def set_states(self, alpha=None, beta=None):
        """
        Set :math:`\ket{0}` and :math:`\ket{1}` Qubit states of the ``Simulator`` object.

        Args:

            alpha (int or ndarray with shape (2s+1, )): :math:`\ket{0}` state of the qubit in :math:`S_z`
                basis or the index of eigenstate to be used as one.

                Default: Lowest energy eigenstate of the central spin Hamiltonian.
                Otherwise state with :math:`m_s = +s` where :math:`m_s` is the z-projection of the spin
                and :math:`s` is the total spin if no information of central spin Hamiltonian is provided.

            beta (int or ndarray with shape (2s+1, )): :math:`\ket{1}` state of the qubit in :math:`S_z` basis
                or the index of the eigenstate to be used as one.

                Default: Second lowest energy eigenstate of the central spin Hamiltonian.
                Otherwise state with :math:`m_s = +s - 1` where :math:`m_s` is the z-projection of the spin
                and :math:`s` is the total spin if no information of central spin Hamiltonian is provided.
        """
        if alpha is None:
            alpha = 0

        if beta is None:
            beta = 1

        if np.asarray(alpha).size == 1 or np.asarray(beta).size == 1:
            if self.zfs.any() or self.magnetic_field.any():
                self.alpha = alpha
                self.beta = beta

            else:
                self.alpha = np.zeros(round(self.spin * 2 + 1))
                self.alpha[alpha] = 1

                self.beta = np.zeros(round(self.spin * 2 + 1))
                self.beta[beta] = 1
        else:
            self.alpha = alpha
            self.beta = beta

    def eigenstates(self, alpha=None, beta=None,
                    magnetic_field=None, D=None, E=0,
                    return_eigen=True):
        """
        Compute eigenstates of the central spin Hamiltonian.

        If ``alpha`` is provided, set alpha state as eigenstate.
        Similarly, if ``beta`` is provided, set beta state as eigenstate

        Args:
            alpha (int):
                Index of the state to be considered as 0 (alpha) qubit state in order of increasing energy
                (0 - lowest energy).
            beta (int):
                Index of the state to be considered as 1 (beta) qubit state.
            magnetic_field (ndarray with shape (3,)): Array containing external magnetic field as (Sx, By, Bz).
            D (float or ndarray with shape (3, 3)): D (longitudinal splitting) parameter of central spin
                in kHz *OR* total ZFS tensor.
            E (float): E (transverse splitting) parameter of central spin in kHz.
                Ignored if ``D`` is None or tensor.
            return_eigen (bool): If true, returns eigenvalues and eigenvectors of the central spin Hamiltonian.

        Returns:
            tuple: *tuple* containing:

                * **ndarray with shape (2s+1,)**: Array with eigenvalues of the central spin Hamiltonian.
                * **ndarray with shape (2s+1, 2s+1)**: Array with eigenvectors of the central spin Hamiltonian.
                  Each column of the array is eigenvector.
        """

        if D is not None:
            self.zfs = zfs_tensor(D, E)

        if magnetic_field is not None:
            self.magnetic_field = magnetic_field

        hamilton = total_hamiltonian(BathArray((0,)), self.magnetic_field, self.zfs,
                                     central_spin=self.spin,
                                     central_gyro=self.gyro)

        en, eiv = np.linalg.eigh(hamilton)

        if alpha is not None:
            self.alpha = eiv[:, alpha]
        if beta is not None:
            self.beta = eiv[:, beta]

        if return_eigen:
            return en, eiv

    def generate_clusters(self, order=None, r_dipole=None, r_inner=0, strong=False, ignore=None):
        r"""
        Generate set of clusters used in CCE calculations.

        The clusters are generated from the following procedure:

        * Each bath spin :math:`i` forms a cluster of one.

        * Bath spins :math:`i` and :math:`j` form cluster of two if there is an edge between them
          (distance :math:`d_{ij} \le` ``r_dipole``).

        * Bath spins :math:`i`, :math:`j`, and :math:`j` form a cluster of three if enough edges connect them
          (e.g., there are two edges :math:`ij` and :math:`jk`)

        * And so on.

        In general, we assume that spins :math:`\{i..n\}` form clusters if they form a connected graph.
        Only clusters up to the size imposed by the ``order`` parameter (equal to CCE order) are included.

        Args:
            order (int): Maximum size of the cluster.

            r_dipole (float): Maximum connectivity distance.

            r_inner (float): Minimum connectivity distance.

            strong (bool):
                True -  generate only clusters with "strong" connectivity (all nodes should be interconnected).
                Default False.

            ignore (list or str, optional): If not None, includes the names of bath spins
                which are ignored in the cluster generation.

        Returns:

            dict: View of ``Simulator.clusters``. ``Simulator.clusters`` is a dictionary
                with keys corresponding to size of the cluster.

                I.e. ``Simulator.clusters[n]`` contains ``ndarray`` of shape (m, n),
                where m is the number of clusters of given size,
                n is the size of the cluster.
                Each row contains indexes of the bath spins included in the given cluster."""

        self._order = order if order is not None else self._order
        self._r_dipole = r_dipole if r_dipole is not None else self._r_dipole
        self.clusters = None

        if self.r_dipole is None:
            raise ValueError('r_dipole was not set')
        elif self.order is None:
            raise ValueError('order was not set')

        clusters = generate_clusters(self.bath, self.r_dipole, self.order,
                                     r_inner=r_inner, strong=strong, ignore=ignore)

        self.clusters = clusters

        return self.clusters

    def read_bath(self, bath=None, r_bath=None,
                  skiprows=1,
                  external_bath=None,
                  hyperfine=None,
                  types=None,
                  error_range=None,
                  ext_r_bath=None,
                  imap=None):

        out = super().read_bath(bath=bath, r_bath=r_bath, skiprows=skiprows,
                                external_bath=external_bath, hyperfine=hyperfine,
                                types=types, error_range=error_range, ext_r_bath=ext_r_bath, imap=imap)

        if self.r_dipole and self.order:
            assert self.bath is not None, "Bath spins were not provided to compute clusters"
            self.generate_clusters()

        return out

    read_bath.__doc__ = Environment.read_bath.__doc__

    @_add_args(_args + _returns)
    def compute(self, timespace, quantity='coherence', method='cce', **kwargs):
        r"""
        General function for computing properties with CCE.

        The dynamics are simulated using the Hamiltonian:

        .. math::

            &\hat H_S = \mathbf{SDS} + \mathbf{B\gamma}_{S}\mathbf{S} \\
            &\hat H_{SB} = \sum_i \mathbf{S}\mathbf{A}_i\mathbf{I}_i \\
            &\hat H_{B} = \sum_i{\mathbf{I}_i\mathbf{P}_i \mathbf{I}_i +
                           \mathbf{B}\mathbf{\gamma}_i\mathbf{I}_i} +
                           \sum_{i>j} \mathbf{I}_i\mathbf{J}_{ij}\mathbf{I}_j

        Here :math:`\hat H_S` is the central spin Hamiltonian with Zero Field splitting tensor :math:`\mathbf{D}` and
        gyromagnetic ratio tensor :math:`\mathbf{\gamma_S} = \mu_S \mathbf{g}`
        are read from ``Simulator.zfs`` and ``Simulator.gyro`` respectively.

        The :math:`\hat H_{SB}` is the Hamiltonian describing interactions between central spin and the bath.
        The hyperfine coupling tensors :math:`\mathbf{A}_i` are read from the ``BathArray`` stored in
        ``Simulator.bath['A']``. They can be generated using point dipole approximation or provided
        by the user (see ``Simulator.read_bath`` for details).

        The :math:`\hat H_{B}` is the Hamiltonian describing interactions between the bath spins.
        The self interaction tensors :math:`\mathbf{P}_i` are read from the ``BathArray`` stored in
        ``Simulator.bath['Q']`` and have to be provided by the user.

        The gyromagnetic ratios :math:`\mathbf{\gamma}_i` are read from the ``BathArray.gyros`` attribuite,
        which is generated from the properties of the types of bath spins, stored in ``BathArray.types``. They can
        either be provided by user or read from the ``pycce.common_isotopes`` object.

        The interaction tensors :math:`\mathbf{J}_{ij}` are assumed from point dipole approximation
        or can be provided  in ``BathArray.imap`` attrubite.

        .. note ::

            The ``compute`` method takes two keyword arguments to determine which quantity to compute and how:

                * `method` can take 'cce' or 'gcce' values, and determines
                  which method to use - conventional or generalized CCE.
                * `quantity` can take 'coherence' or 'noise' values, and determines
                  which quantity to compute - coherence function
                  or autocorrelation function of the noise.

            Each of the methods can be performed with monte carlo bath state sampling
            (if ``nbstates`` keyword is non zero)
            and with interlaced averaging (If ``interlaced`` keyword is set to ``True``).

        Examples:

            First set up Simulator object using random bath of 1000 13C nuclear spins.

                >>> import pycce as pc
                >>> import numpy as np
                >>> atoms = pc.random_bath('13C', 100, number=2000, seed=10) # Random spin bath
                >>> calc = pc.Simulator(1, bath=atoms, r_bath=40, r_dipole=6,
                >>>                     order=2, D=2.88 * 1e6, # D of NV in GHz -> kHz
                >>>                     magnetic_field=500, pulses=1)
                >>> ts = np.linspace(0, 2, 101) # timesteps

            We set magnetic field to 500 G along z-axis and chose 1 decoupling pulse (Hahn-echo) in this example.
            The zero field splitting is set to the one of NV center in diamond.

            Run conventional CCE calculation at time points ``timespace`` to obtain coherence without second order
            effects:

                >>> calc.compute(ts)

            This will call ``Simulator.cce_coherence`` method with default keyword values.

            Compute the coherence conventional CCE coherence with second order interactions between bath spins:

                >>> calc.compute(ts, second_order=True)

            Compute the coherence with conventional CCE with bath state sampling (over 10 states):

                >>> calc.compute(ts, nbstates=10)

            Compute the coherence with generalized CCE:

                >>> calc.compute(ts, method='gcce')

            This will call ``Simulator.gcce_dm`` method with default keyword values and obtain off diagonal
            element as :math:`\bra{0} \hat \rho_{C} \ket{1}`,
            where :math:`\hat \rho_{C}` is the density matrix of the qubit.

            Compute the coherence with generalized CCE with bath state sampling (over 10 states):

                >>> calc.compute(ts, method='gcce', nbstates=10)

        Args:

            timespace (ndarray with shape (n,)): Time points at which compute the desired property.

            quantity (str): Which quantity to compute. Case insensitive.

                Possible values:

                    - 'coherence': compute coherence function.
                    - 'noise': compute noise autocorrelation function.

            method (str): Which implementation of CCE to use. Case insensitive.

                Possible values:

                    - 'cce': conventional CCE, where interactions are mapped on 2 level pseudospin.
                    - 'gcce': Generalized CCE where central spin is included in each cluster.
        """
        if quantity.lower() == 'noise':
            kwargs.setdefault('masked', False)

        else:
            kwargs.setdefault('masked', True)

        self.timespace = timespace
        self._prepare(**kwargs)

        runner = self._compute_func[method.lower()][quantity.lower()].from_simulator(self)

        if not self.interlaced:

            if self.nbstates:
                result = runner.sampling_run()

            else:
                result = runner.run()

        else:

            if self.nbstates:
                result = runner.sampling_interlaced_run()

            else:
                result = runner.interlaced_run()

        return result

    def _broadcast(self):
        """
        Update attributes for the ``Simulator`` object from the root process.
        """
        calc = _broadcast_simulator(self)
        dict_with_attr = vars(calc)

        for key in dict_with_attr:
            setattr(self, key, dict_with_attr[key])

    _compute_func = {
        'cce': {
            'coherence': CCE,
            'noise': CCENoise
        },
        'gcce': {
            'coherence': gCCE,
            'noise': gCCENoise,
        }
    }

    def _gen_state(self, state=None):

        if state is None:
            if not (self.alpha.shape and self.beta.shape):
                self.state = None
            else:
                self.state = (self.alpha + self.beta) / np.linalg.norm((self.alpha + self.beta))
        else:
            self.state = state

    @_add_args(_args)
    def _prepare(self, state=None,
                 pulses=None,
                 D=None, E=0,
                 magnetic_field=None,
                 as_delay=False,
                 alpha=None,
                 beta=None,
                 bath_state=None,
                 nbstates=None,
                 seed=None,
                 masked=True,
                 parallel_states=False,
                 fixstates=None,
                 second_order=False,
                 level_confidence=0.95,
                 direct=False,
                 parallel=False,
                 interlaced=False):
        """

        Args:

        """

        if D is not None:
            self.zfs = zfs_tensor(D, E)

        if alpha is not None:
            self.alpha = alpha

        if beta is not None:
            self.beta = beta

        self._gen_state(state)

        if pulses is not None:
            self.pulses = pulses
        if magnetic_field is not None:
            self.set_magnetic_field(magnetic_field)

        self.parallel = parallel
        self.parallel_states = parallel_states
        self.as_delay = as_delay
        self.direct = direct

        if bath_state is not None:
            self.bath_state = np.asarray(bath_state)
        else:
            self.bath_state = None

        self.second_order = second_order
        self.level_confidence = level_confidence
        self.fixstates = fixstates

        self.nbstates = nbstates
        self.seed = seed
        self.masked = masked
        self.interlaced = interlaced

        if parallel or parallel_states:
            self._broadcast()

        if bath_state is not None:
            self.projected_bath_state = project_bath_states(bath_state)

        else:
            self.projected_bath_state = None


def _broadcast_simulator(simulator=None, root=0):
    """
    Broadcast ``Simulator`` object from root to all mpi processes.

    Args:

        simulator (Simulator): Object to broadcast. Should be defined at root process.
        root (int): Index of the root process.

    Returns:

        Simulator: Broadcasted object.
    """
    try:
        import mpi4py
    except ImportError:
        raise RuntimeError('Could not find mpi4py. Cannot broadcast the Simulator')

    comm = mpi4py.MPI.COMM_WORLD

    # size = comm.Get_size()
    rank = comm.Get_rank()

    nsim = comm.bcast(simulator, root=root)
    if rank == root:
        bath = simulator.bath
        parameters = vars(bath)
    else:
        bath = None
        parameters = None

    nbath = comm.bcast(bath, root)
    nparam = comm.bcast(parameters, root)

    for k in nparam:
        setattr(nbath, k, nparam[k])
    nsim._bath = nbath

    return nsim
