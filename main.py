import numpy as np
import numpy.ma as ma
from .bath.read_bath import read_pos, read_external, gen_hyperfine
from .coherence_function import cluster_L
from .find_clusters import make_graph, connected_components, find_subclusters
from .elham import cluster_dm, cluster_dm_direct_approach, compute_dm
from .hamiltonian import generate_SpinMatricies, QSpinMatrix, total_elhamiltonian


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
                 gyro=-17608.597050):

        if position is None:
            position = np.zeros(3)

        self.position = np.asarray(position, dtype=np.float64)

        self.ntype = {}

        self.spin = spin
        self.alpha = np.zeros(round(2 * spin + 1), dtype=np.complex128)
        self.beta = np.zeros(round(2 * spin + 1), dtype=np.complex128)

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
                  hf_positions=None,
                  hf_dipole=None,
                  hf_contact=None,
                  error_range=0.5):

        self.bath = None
        external_atoms = None
        atoms = read_pos(nspin, center=self.position,
                             r_bath=r_bath, skiprows=skiprows)

        if hf_positions and (hf_dipole or hf_contact):
            external_atoms = read_external(hf_positions, hf_dipole, hf_contact)

        self.bath = gen_hyperfine(atoms, self.ntype,
                                  center=self.position,
                                  gamma_e=self.gyro,
                                  error_range=error_range,
                                  external_atoms=external_atoms)
        return self.bath

    def generate_graph(self, r_dipole, r_inner=0):
        self.graph = None
        self.graph = make_graph(self.bath, r_dipole, R_inner=r_inner)

        return self.graph

    def generate_clusters(self, CCE_order, strong=False):

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

        L = cluster_L(self.clusters, self.bath, self.ntype, I, S, B, timespace, N,
                      as_delay=as_delay)

        return L

    def compute_dmatrix(self, timespace, B, D, E, pulse_sequence, as_delay=False, state=None, check=False):
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
            dms_c = cluster_dm(self.clusters, self.bath, self.ntype,
                               dm0, I, S, B, self.gyro, D, E,
                               timespace, pulse_sequence, as_delay=False)
        else:
            dms_c = cluster_dm_direct_approach(self.clusters, self.bath, self.ntype,
                                               dm0, I, S, B, self.gyro, D, E,
                                               timespace, pulse_sequence, as_delay=False)
        dms *= dms_c

        return dms


class SpinType:
    def __init__(self, isotope, s=0, gyro=0):
        self.isotope = isotope
        self.s = s
        self.gyro = gyro
