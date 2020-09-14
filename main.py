import numpy as np
import numpy.ma as ma
from .bath.read_bath import read_pos, read_external, gen_hyperfine
from .coherence_function import decorated_coherence_function
from .find_clusters import make_graph, connected_components, find_subclusters
from .density_matrix import decorated_density_matrix, cluster_dm_direct_approach, compute_dm
from .hamiltonian import generate_SpinMatricies, QSpinMatrix, total_elhamiltonian, mf_hamiltonian
from .mean_field_dm import mean_field_density_matrix
from .correlation_function import mean_field_noise_correlation, decorated_noise_correlation


class QSpin:
    """
     The main object for CCE calculations

     Default Units
     Length: Angstrom, A
     Time: Millisecond, ms
     Magnetic Field: Gaussian, G = 1e-4 Tesla
     Gyromagnetic Ratio: rad/(msec*Gauss)

    """
    _dtype_read = np.dtype([('N', np.unicode_, 16), ('xyz', np.float64, (3,))])
    _dtype_bath = np.dtype([('N', np.unicode_, 16),
                            ('xyz', np.float64, (3,)),
                            ('A', np.float64, (3, 3))])

    def __init__(self, spin=1, position=None,
                 alpha=None, beta=None,
                 gyro=-17608.597050):

        if position is None:
            position = np.zeros(3)

        self.position = np.asarray(position, dtype=np.float64)

        self.ntype = {}

        self.spin = spin

        if alpha is None:
            alpha = np.zeros(int(round(2 * spin + 1)), dtype=np.complex128)
            alpha[1] = 1

        if beta is None:
            beta = np.zeros(int(round(2 * spin + 1)), dtype=np.complex128)
            beta[1] = 1

        self.alpha = np.asarray(alpha)
        self.beta = np.asarray(beta)
        self.gyro = gyro

        self.bath = None
        self.graph = None

        self.r_bath = None
        self.r_dipole = None

        self.clusters = []

    def add_spintype(self, *args):
        for nuc in args:
            self.ntype[nuc[0]] = SpinType(*nuc)

        return self.ntype

    def read_bath(self, nspin, r_bath, *,
                  skiprows=1,
                  external_bath=None,
                  hf_positions=None,
                  hf_dipole=None,
                  hf_contact=None,
                  error_range=0.2,
                  ext_r_bath=None):

        self.bath = None

        atoms = read_pos(nspin, center=self.position,
                         r_bath=r_bath, skiprows=skiprows)

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

    def generate_graph(self, r_dipole, r_inner=0):
        self.graph = None
        self.graph = make_graph(self.bath, r_dipole, R_inner=r_inner)

        return self.graph

    def generate_clusters(self, CCE_order, r_dipole=None, r_inner=0, strong=False):
        if self.graph is None:
            assert r_dipole is not None, "Graph generation failed: r_dipole is not provided"
            self.graph = make_graph(self.bath, r_dipole, R_inner=r_inner)

        self.clusters = None
        n_components, labels = connected_components(csgraph=self.graph, directed=False,
                                                    return_labels=True)

        clusters = find_subclusters(
            CCE_order, self.graph, labels, n_components, strong=strong)

        self.clusters = clusters

        return self.clusters

    def compute_coherence(self, timespace, B, N, as_delay=False):
        I = generate_SpinMatricies(self.ntype)
        S = QSpinMatrix(self.spin, self.alpha, self.beta)
        L = decorated_coherence_function(self.clusters, self.bath, self.ntype, I, S, B,
                                         timespace, N, as_delay=as_delay)

        return L

    def compute_dmatrix(self, timespace, B, D, E, pulse_sequence, as_delay=False, state=None, check=True):
        if state is None:
            state = np.sqrt(1 / 2) * (self.alpha + self.beta)

        dm0 = np.tensordot(state, state, axes=0)

        I = generate_SpinMatricies(self.ntype)
        S = QSpinMatrix(self.spin, self.alpha, self.beta)

        H0, dimensions0 = total_elhamiltonian(np.array([]), self.ntype,
                                              I, B, S, self.gyro, D, E)

        dms = compute_dm(dm0, dimensions0, H0, S, timespace, pulse_sequence,
                         as_delay=as_delay)

        dms = ma.masked_array(dms, mask=(dms == 0), fill_value=0j, dtype=np.complex128)

        if check:
            dms_c = decorated_density_matrix(self.clusters, self.bath, self.ntype,
                                             dm0, I, S, B, D, E,
                                             timespace, pulse_sequence, gyro_e=self.gyro,
                                             as_delay=as_delay, zeroth_cluster=dms)

        else:
            dms_c = cluster_dm_direct_approach(self.clusters, self.bath, self.ntype,
                                               dm0, I, S, B, self.gyro, D, E,
                                               timespace, pulse_sequence, as_delay=as_delay)
        dms *= dms_c

        return dms

    def compute_mf_dm(self, timespace, B, D, E, pulse_sequence, as_delay=False, state=None,
                      nbstates=100, seed=None, masked=True, normalized=None, parallel=False,
                      fixstates=None):
        if parallel:
            try:
                from mpi4py import MPI
            except ImportError:
                print('Parallel failed: mpi4py is not found. Running serial')
                parallel = False

        if state is None:
            state = np.sqrt(1 / 2) * (self.alpha + self.beta)

        dm0 = np.tensordot(state, state, axes=0)
        I = generate_SpinMatricies(self.ntype)
        S = QSpinMatrix(self.spin, self.alpha, self.beta)

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
            H0, d0 = mf_hamiltonian(np.array([]), self.ntype,
                                    I, B, S, self.gyro, D, E, self.bath, bath_state)

            dmzero = compute_dm(dm0, d0, H0, S, timespace, pulse_sequence, as_delay=as_delay)
            dmzero = ma.array(dmzero, mask=(dmzero == 0), fill_value=0j, dtype=np.complex128)
            # avdm0 += dmzero
            dms = mean_field_density_matrix(self.clusters, self.bath, self.ntype,
                                            dm0, I, S, B, D, E,
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

    def mean_field_corr(self, timespace, B, D, E, state=None,
                        nbstates=100, seed=None, parallel=False):
        if parallel:
            try:
                from mpi4py import MPI
            except ImportError:
                print('Parallel failed: mpi4py is not found. Running serial')
                parallel = False

        if state is None:
            state = np.sqrt(1 / 2) * (self.alpha + self.beta)

        dm0 = np.tensordot(state, state, axes=0)
        I = generate_SpinMatricies(self.ntype)
        S = QSpinMatrix(self.spin, self.alpha, self.beta)

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
                                                dm0, I, S, B, D, E, timespace,
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

    def compute_corr(self, timespace, B, D, E, state=None):
        if state is None:
            state = np.sqrt(1 / 2) * (self.alpha + self.beta)

        dm0 = np.tensordot(state, state, axes=0)
        I = generate_SpinMatricies(self.ntype)
        S = QSpinMatrix(self.spin, self.alpha, self.beta)

        corr = decorated_noise_correlation(self.clusters, self.bath, self.ntype,
                                           dm0, I, S, B, D, E,
                                           timespace,
                                           gyro_e=self.gyro)
        return corr


class SpinType:
    def __init__(self, isotope, s=0, gyro=0):
        self.isotope = isotope
        self.s = s
        self.gyro = gyro
