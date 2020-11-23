import numpy as np
import numpy.ma as ma

from .bath.read_bath import read_pos, read_external, gen_hyperfine
from .coherence_function import decorated_coherence_function
from .correlation_function import mean_field_noise_correlation, decorated_noise_correlation
from .density_matrix import decorated_density_matrix, cluster_dm_direct_approach, compute_dm
from .find_clusters import make_graph, connected_components, find_subclusters
from .hamiltonian import generate_spinmatrices, QSpinMatrix, total_hamiltonian, mf_hamiltonian
from .mean_field_dm import mean_field_density_matrix


class Simulator:
    """
    The main class for CCE calculations

    Default Units
    Length: Angstrom, A
    Time: Millisecond, ms
    Magnetic Field: Gaussian, G = 1e-4 Tesla
    Gyromagnetic Ratio: rad/(msec*Gauss)
    Quadrupole moment: millibarn

    Parameters
    ------------
    @param spin: float
        total spin of the central spin (default 1.)
    @param position: ndarray
        xyz coordinates in angstrom of the central spin (default (0., 0., 0.))
    @param alpha: ndarray
        0 state of the qubit in Sz basis (default not set)
    @param beta: ndarray
        1 state of the qubit in Sz basis (default not set)
    @param gyro: ndarray
        gyromagnetic ratio of central spin in rad/(ms * G) (default -17608.597050)

    """
    _dtype_read = np.dtype([('N', np.unicode_, 16), ('xyz', np.float64, (3,))])

    _dtype_bath = np.dtype([('N', np.unicode_, 16),
                            ('xyz', np.float64, (3,)),
                            ('A', np.float64, (3, 3)),
                            ('V', np.float64, (3, 3))])

    def __init__(self, spin=1., position=None, alpha=None, beta=None, gyro=-17608.597050,
                 spin_types=None, bath_spins=None, r_bath=None, bath_kw={},
                 r_dipole=None, order=None):

        if position is None:
            position = np.zeros(3)

        self.position = np.asarray(position, dtype=np.float64)

        self.ntype = {}

        if spin_types is not None:
            self.add_spintype(*spin_types)
        else:
            self.I = None
            self.S = None

        self.spin = spin

        if alpha is None:
            alpha = np.zeros(int(round(2 * spin + 1)), dtype=np.complex128)
            alpha[1] = 1

        if beta is None:
            beta = np.zeros(int(round(2 * spin + 1)), dtype=np.complex128)
            beta[1] = 1

        self._alpha = np.asarray(alpha)
        self._beta = np.asarray(beta)
        self.S = QSpinMatrix(self.spin, self.alpha, self.beta)

        self.gyro = gyro

        self.r_bath = r_bath
        self.bath = None
        if bath_spins is not None and r_bath > 0:
            self.read_bath(bath_spins, r_bath, **bath_kw)

        self.r_dipole = r_dipole

        self.graph = None
        self.clusters = None
        if r_dipole is not None and order is not None:
            assert self.bath is not None, "Bath spins were not provided to compute clusters"
            self.generate_clusters(order, r_dipole)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha_state):
        self._alpha = np.asarray(alpha_state)
        self.S = QSpinMatrix(self.spin, self.alpha, self.beta)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta_state):
        self._beta = np.asarray(beta_state)
        self.S = QSpinMatrix(self.spin, self.alpha, self.beta)

    def add_spintype(self, *spin_types):
        """
        Add types of spin in the spin bath
        @param spin_types: tuple(s) or SpinType(s)
            arbitrary number of tuples, with each of them initializing SpinType object:
            SpinType(isotope, s=0, gyro=0, q=0)
            where isotope (str) is the name of the bath spin, s (float) is the total spin,
            gyro (float) is the gyromagnetic ratio in rad/(ms*G),
            q is spin quadrupole moment in millibarn (for s > 1/2)
            OR
            instances of SpinTypes
        @return: dict
            dict of the SpinType instances
        """
        for nuc in spin_types:
            if isinstance(nuc, SpinType):
                self.ntype[nuc.isotope] = nuc
            self.ntype[nuc[0]] = SpinType(*nuc)

        self.I = generate_spinmatrices(self.ntype)

        return self.ntype

    def read_bath(self, nspin, r_bath, *,
                  skiprows=1,
                  external_bath=None,
                  hf_positions=None,
                  hf_dipole=None,
                  hf_contact=None,
                  error_range=0.2,
                  ext_r_bath=None):
        """
        Read spin bath
        @param nspin: ndarray or str
            Either ndarray with dtype([('N', np.unicode_, 16), ('xyz', np.float64, (3,))]) containing names
            of bath spins (same ones as stored in self.ntype) and positions of the spins in A;
            or the name of the text file containing 4 rows: name of the bath spin and xyz coordinates in A
        @param r_bath: cutoff size of the nuclear bath
        @param skiprows: int, optional
            if nspin is name of the file, number of rows to skip in reading the file (default 1)
        @param external_bath: ndarray, optional
            ndarray of atoms read from GIPAW output (see bath.read_pw_gipaw)
        @param hf_positions: str, optional
            name of the file with positions of bath spins from GIPAW output (used for backwards capabilities)
        @param hf_dipole: str, optional
            name of the file with dipolar tensors of bath spins, similar to GIPAW output
        @param hf_contact: str, optional
            name of the file with contact terms from GIPAW output
        @param error_range: float, optional
            maximum distance between positions in nspin and external bath to consider two positions the same
            (default 0.2)
        @param ext_r_bath: float, optional
            maximum distance from the central spins of the bath spins for which to use the DFT positions
        @return: bath
            ndarray of atoms with dtype([('N', np.unicode_, 16), ('xyz', np.float64, (3,)), ('A', np.float64, (3, 3))])
            where N is the name of the isotope, xyz are coordinates (in A), A are HF tensors (in rad * KHz)
        """
        self.bath = None

        atoms = read_pos(nspin, r_bath=r_bath, center=self.position, skiprows=skiprows)

        if external_bath is not None and ext_r_bath is not None:
            where = np.linalg.norm(external_bath['xyz'] - self.position, axis=1) <= ext_r_bath
            external_bath = external_bath[where]

        if hf_positions and (hf_dipole or hf_contact) and external_bath is None:
            external_bath = read_external(hf_positions, hf_dipole, hf_contact,
                                          center=self.position, erbath=ext_r_bath)

        self.bath = gen_hyperfine(atoms, self.ntype,
                                  center=self.position,
                                  gyro_e=self.gyro,
                                  error_range=error_range,
                                  external_atoms=external_bath)

        return self.bath

    def generate_graph(self, r_dipole, r_inner=0.):
        """
        Generate the connectivity matrix for the given bath
        @param r_dipole: float
            maximum connectivity distance
        @param r_inner: float
            minimum connectivity distance
        @return: csr_matrix
            sparse connectivity matrix in csr format (see scipy.sparse.csr_matrix for details).
        """
        self.graph = None
        self.r_dipole = r_dipole
        self.graph = make_graph(self.bath, r_dipole, r_inner=r_inner)

        return self.graph

    def generate_clusters(self, order, r_dipole=None, r_inner=0, strong=False):
        """
        Generate clusters used in CCE calculations. First generates connectivity matrix
        if was not generated previously
        @param order: int
            maximum size of the cluster
        @param r_dipole: float
            maximum connectivity distance (used if graph was not generated before)
        @param r_inner: float
            minimum connectivity distance (used if graph was not generated before)
        @param strong: bool
            True -  generate only clusters with "strong" connectivity (all nodes should be interconnected)
            default False
        @return: dict
        dict with keys corresponding to size of the cluster, and value corresponds to ndarray of shape (M, N),
        M is the number of clusters of given size, N is the size of the cluster. Each row contains indexes of the bath
        spins included in the given cluster
        """
        if self.graph is None:
            if r_dipole is not None:
                self.r_dipole = r_dipole

            assert self.r_dipole is not None, "Graph generation failed: r_dipole is not provided"
            self.graph = self.generate_graph(r_dipole, r_inner=r_inner)

        self.clusters = None
        n_components, labels = connected_components(csgraph=self.graph, directed=False,
                                                    return_labels=True)

        clusters = find_subclusters(
            order, self.graph, labels, n_components, strong=strong)

        self.clusters = clusters

        return self.clusters

    def compute_coherence(self, timespace, B, N, as_delay=False):
        """
        Compute coherence function L with conventional CCE
        @param timespace: 1D-ndarray
            time points at which compute coherence function L
        @param B: ndarray
            magnetic field as (Bx, By, Bz)
        @param N: int
            number of pulses of CPMG sequence
        @param as_delay: bool
            True if time points correspond to delay between pulses
            False if time points correspond to the total time of the experiment
        @return: 1D-ndarray
            Coherence function computed at the time points in timespace
        """
        L = decorated_coherence_function(self.clusters, self.bath, self.ntype, self.I, self.S, B,
                                         timespace, N, as_delay=as_delay)

        return L

    def compute_dmatrix(self, timespace: np.ndarray, B: np.ndarray,
                        D: float, E: float = 0, pulse_sequence: list = None,
                        as_delay: bool = False, state: np.ndarray = None,
                        check: bool = True) -> np.ndarray:
        """
        Compute density matrix of the central spin using generalized CCE
        @param timespace: 1D-ndarray
            time points at which compute density matrix
        @param B: ndarray
            magnetic field as (Bx, By, Bz)
        @param D: float or ndarray with shape (3,3)
            D (longitudinal splitting) parameter of central spin in ZFS tensor of central spin in rad * kHz
            OR total ZFS tensor
        @param E: float
            E (transverse splitting) parameter of central spin in ZFS tensor of central spin in rad * kHz
        @param pulse_sequence: list
            pulse_sequence should have format of list with tuples,
            each tuple contains two entries: first: axis the rotation is about; second: angle of rotation.
            E.g. for Hahn-Echo [('x', np.pi)]. For now only pulses with same delay are supported
        @param as_delay: bool
            True if time points are delay between pulses,
            False if time points are total time
        @param state: ndarray
            Initial state of the central spin. Defaults to sqrt(1 / 2) * (alpha + beta) if not set
        @param check: bool
            True if use optimized algorithm of computing cluster contributions
            (faster but might fail if too many clusters)
            False if use direct approach (slower)
        @return: ndarray
            array of density matrix, where first dimension corresponds to the time space size and last two -
            density matrix dimensions
        """
        if state is None:
            state = np.sqrt(1 / 2) * (self.alpha + self.beta)

        dm0 = np.tensordot(state, state, axes=0)

        H0, dimensions0 = total_hamiltonian(np.array([]), self.ntype, self.I, self.S, B, self.gyro, D, E)

        dms = compute_dm(dm0, dimensions0, H0, self.S, timespace, pulse_sequence,
                         as_delay=as_delay)

        dms = ma.masked_array(dms, mask=(dms == 0), fill_value=0j, dtype=np.complex128)

        if check:
            dms_c = decorated_density_matrix(self.clusters, self.bath, self.ntype,
                                             dm0, self.I, self.S, B, D, E,
                                             timespace, pulse_sequence, gyro_e=self.gyro,
                                             as_delay=as_delay, zeroth_cluster=dms)

        else:
            dms_c = cluster_dm_direct_approach(self.clusters, self.bath, self.ntype,
                                               dm0, self.I, self.S, B, self.gyro, D, E,
                                               timespace, pulse_sequence, as_delay=as_delay)
        dms *= dms_c

        return dms

    def compute_mf_dm(self, timespace, B, D, E=0, pulse_sequence=None, as_delay=False, state=None,
                      nbstates=100, seed=None, masked=True, normalized=None, parallel=False,
                      fixstates=None):
        """
        Compute density matrix of the central spin using generalized CCE with Monte-Carlo bath state sampling
        @param timespace: 1D-ndarray
            time points at which compute density matrix
        @param B: ndarray
            magnetic field as (Bx, By, Bz)
        @param D: float or ndarray with shape (3,3)
            D (longitudinal splitting) parameter of central spin in ZFS tensor of central spin in rad * kHz
            OR total ZFS tensor
        @param E: float
            E (transverse splitting) parameter of central spin in ZFS tensor of central spin in rad * kHz
        @param pulse_sequence: list
            pulse_sequence should have format of list with tuples,
            each tuple contains two entries: first: axis the rotation is about; second: angle of rotation.
            E.g. for Hahn-Echo [('x', np.pi/2)]. For now only pulses with same delay are supported
        @param as_delay: bool
            True if time points are delay between pulses,
            False if time points are total time
        @param state: ndarray
            Initial state of the central spin. Defaults to sqrt(1 / 2) * (alpha + beta) if not set
        @param nbstates: int
            Number of random bath states to sample
        @param seed: int
            Seed for the RNG
        @param masked: bool
            True if mask numerically unstable points (with density matrix elements > 1)
            in the averaging over bath states
            False if not. Default True
        @param normalized: ndarray of bools
            which diagonal elements to renormalize, so the total sum of the diagonal elements is 1
        @param parallel: bool
            whether to use MPI to parallelize the calculations of density matrix
            for each random bath state
        @param fixstates: dict
            dict of which bath states to fix. Each key is the index of bath spin,
            value - fixed Sz projection of the mixed state of nuclear spin
        @return: dms
        """
        if parallel:
            try:
                from mpi4py import MPI
            except ImportError:
                print('Parallel failed: mpi4py is not found. Running serial')
                parallel = False

        if state is None:
            state = np.sqrt(1 / 2) * (self.alpha + self.beta)

        dm0 = np.tensordot(state, state, axes=0)

        if masked:
            divider = np.zeros(timespace.shape, dtype=np.int32)
        else:
            root_divider = nbstates

        if parallel:
            comm = MPI.COMM_WORLD

            size = comm.Get_size()
            rank = comm.Get_rank()

            remainder = nbstates % size
            add = int(rank < remainder)
            nbstates = nbstates // size + add

            if seed:
                seed = seed + rank
        else:
            rank = 0

        rgen = np.random.default_rng(seed)

        averaged_dms = ma.zeros((timespace.size, *dm0.shape), dtype=np.complex128)
        # avdm0 = 0

        for _ in range(nbstates):

            bath_state = np.empty(self.bath.shape, dtype=np.float64)
            for n in self.ntype:
                s = self.ntype[n].s
                snumber = int(round(2 * s + 1))
                mask = self.bath['N'] == n
                bath_state[mask] = rgen.integers(snumber, size=np.count_nonzero(mask)) - s

            if fixstates is not None:
                for fs in fixstates:
                    bath_state[fs] = fixstates[fs]

            H0, d0 = mf_hamiltonian(np.array([]), self.ntype, self.I, self.S, B,
                                    self.gyro, D, E, self.bath, bath_state)

            dmzero = compute_dm(dm0, d0, H0, self.S, timespace, pulse_sequence, as_delay=as_delay)
            dmzero = ma.array(dmzero, mask=(dmzero == 0), fill_value=0j, dtype=np.complex128)
            # avdm0 += dmzero
            dms = mean_field_density_matrix(self.clusters, self.bath, self.ntype,
                                            dm0, self.I, self.S, B, D, E,
                                            timespace, pulse_sequence, self.bath, bath_state,
                                            as_delay=as_delay, zeroth_cluster=dmzero) * dmzero
            if masked:
                dms = dms.filled()
                proper = np.all(np.abs(dms) <= 1, axis=(1, 2))
                divider += proper.astype(np.int32)
                dms[~proper] = 0.

            if normalized is not None:
                norm = np.asarray(normalized)
                ind = np.arange(dms.shape[1])
                diagonals = dms[:, ind, ind]

                sums = np.sum(diagonals[:, norm], keepdims=True, axis=-1)
                sums[sums == 0.] = 1

                expsum = 1 - np.sum(diagonals[:, ~norm], keepdims=True, axis=-1)

                diagonals[:, norm] = diagonals[:, norm] / sums * expsum

                # print(diagonals)
                dms[:, ind, ind] = diagonals

            averaged_dms += dms

        if parallel:
            root_dms = ma.array(np.zeros(averaged_dms.shape), dtype=np.complex128)
            comm.Reduce(averaged_dms, root_dms, MPI.SUM, root=0)
            if masked:
                root_divider = np.zeros(divider.shape, dtype=np.int32)
                comm.Reduce(divider, root_divider, MPI.SUM, root=0)

        else:
            root_dms = averaged_dms
            if masked:
                root_divider = divider

        if rank == 0:
            root_dms = ma.array(root_dms, fill_value=0j, dtype=np.complex128)

            if masked:
                root_dms[root_divider == 0] = ma.masked
                root_divider = root_divider[:, np.newaxis, np.newaxis]
            root_dms /= root_divider

            return root_dms
        else:
            return

    def mean_field_corr(self, timespace, B, D, E=0, state=None,
                        nbstates=100, seed=None, parallel=False):
        """
        EXPERIMENTAL Compute noise auto correlation function
        using generalized CCE with Monte-Carlo bath state sampling
        @param timespace: 1D-ndarray
            time points at which compute density matrix
        @param B: ndarray
            magnetic field as (Bx, By, Bz)
        @param D: float or ndarray with shape (3,3)
            D (longitudinal splitting) parameter of central spin in ZFS tensor of central spin in rad * kHz
            OR total ZFS tensor
        @param E: float
            E (transverse splitting) parameter of central spin in ZFS tensor of central spin in rad * kHz
        @param state: ndarray
            Initial state of the central spin. Defaults to sqrt(1 / 2) * (alpha + beta) if not set
        @param nbstates: int
            Number of random bath states to sample
        @param seed: int
            Seed for the RNG
        @param parallel: bool
            whether to use MPI to parallelize the calculations of density matrix
            for each random bath state
        @return: ndarray
            Autocorrelation function of the noise, in (kHz*rad)^2 of shape (N, 3)
            where N is the number of time points and at each point (Ax, Ay, Az) are noise autocorrelation functions
        """
        if parallel:
            try:
                from mpi4py import MPI
            except ImportError:
                print('Parallel failed: mpi4py is not found. Running serial')
                parallel = False

        if state is None:
            state = np.sqrt(1 / 2) * (self.alpha + self.beta)

        dm0 = np.tensordot(state, state, axes=0)
        root_divider = nbstates

        if parallel:
            comm = MPI.COMM_WORLD

            size = comm.Get_size()
            rank = comm.Get_rank()

            remainder = nbstates % size
            add = int(rank < remainder)
            nbstates = nbstates // size + add

            if seed:
                seed = seed + rank
        else:
            rank = 0

        rgen = np.random.default_rng(seed)

        averaged_corr = 0

        for _ in range(nbstates):

            bath_state = np.empty(self.bath.shape, dtype=np.float64)
            for n in self.ntype:
                s = self.ntype[n].s
                snumber = int(round(2 * s + 1))
                mask = self.bath['N'] == n
                bath_state[mask] = rgen.integers(snumber, size=np.count_nonzero(mask)) - s

            corr = mean_field_noise_correlation(self.clusters, self.bath, self.ntype,
                                                dm0, self.I, self.S, B, D, E, timespace,
                                                self.bath, bath_state, gyro_e=self.gyro)

            averaged_corr += corr

        if parallel:
            root_corr = np.array(np.zeros(averaged_corr.shape), dtype=np.complex128)
            comm.Reduce(averaged_corr, root_corr, MPI.SUM, root=0)

        else:
            root_corr = averaged_corr

        if rank == 0:
            root_corr /= root_divider

            return root_corr
        else:
            return

    def compute_corr(self, timespace, B, D, E=0, state=None):
        """
        EXPERIMENTAL Compute noise autocorrelation function of the noise with generalized CCE
        @param timespace:  1D-ndarray
            time points at which compute density matrix
        @param B: ndarray
            magnetic field as (Bx, By, Bz)
        @param D: float or ndarray with shape (3,3)
            D (longitudinal splitting) parameter of central spin in ZFS tensor of central spin in rad * kHz
            OR total ZFS tensor
        @param E: float
            E (transverse splitting) parameter of central spin in ZFS tensor of central spin in rad * kHz
        @param state: ndarray
            Initial state of the central spin. Defaults to sqrt(1 / 2) * (alpha + beta) if not set
        @return: ndarray
            Autocorrelation function of the noise, in (kHz*rad)^2 of shape (N, 3)
            where N is the number of time points and at each point (Ax, Ay, Az) are noise autocorrelation functions
        """
        if state is None:
            state = np.sqrt(1 / 2) * (self.alpha + self.beta)

        dm0 = np.tensordot(state, state, axes=0)

        corr = decorated_noise_correlation(self.clusters, self.bath, self.ntype,
                                           dm0, self.I, self.S, B, D, E,
                                           timespace,
                                           gyro_e=self.gyro)
        return corr


class SpinType:
    """
    Class which contains properties of each spin type in the bath

    Parameters
    ----------
    @param isotope: str
        Name of the bath spin
    @param s: float
        Total spin
    @param gyro:
        Gyromagnetic ratio in rad/(ms * G)
    @param q:
        Quadrupole moment in millibarn (for s > 1/2)
    """

    def __init__(self, isotope, s=0, gyro=0, q=0):
        self.isotope = isotope
        self.s = s
        self.gyro = gyro
        self.q = q


# Just additional alias for backwards compatibility
QSpin = Simulator
