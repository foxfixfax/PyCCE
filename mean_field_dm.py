import numpy as np
import numpy.ma as ma
import scipy.linalg
from .hamiltonian import mf_hamiltonian, expand
from .cluster_expansion import cluster_expansion_decorator
from .density_matrix import compute_dm, full_dm, generate_dm0

hbar = 1.05457172  # When everything else in rad, kHz, ms, G, A

def compute_mf_dm(dm0, dimensions, states, H, S, timespace,
                  pulse_sequence=None, as_delay=False):
    if not as_delay and pulse_sequence:
        N = len(pulse_sequence)
        timespace = timespace / (2 * N)

    dmtotal0 = generate_dm0(dm0, dimensions, states)
    dm = full_dm(dmtotal0, dimensions, H, S, timespace, pulse_sequence=pulse_sequence)

    initial_shape = dm.shape
    dm.shape = (initial_shape[0], *dimensions, *dimensions)
    for d in range(len(dimensions) + 1, 2, -1):  # The last one is el spin
        dm = np.trace(dm, axis1=1, axis2=d)
    return dm


@cluster_expansion_decorator
def mean_field_density_matrix(nspin, ntype,
                              dm0, I, S, B, D, E,
                              timespace, pulse_sequence, allspins, bath_state,
                              gyro_e=-17608.597050,
                              as_delay=False, zeroth_cluster=None):
    others_mask = np.isin(allspins, nspin)

    others = allspins[~others_mask]
    others_state = bath_state[~others_mask]

    states = bath_state[others_mask]

    if zeroth_cluster is None:
        H, dimensions = mf_hamiltonian(np.array([]), ntype,
                                       I, B, S, gyro_e, D, E, others, others_state)
        zeroth_cluster = compute_dm(dm0, dimensions, H, S, timespace, pulse_sequence,
                                    as_delay=as_delay)
        zeroth_cluster = ma.masked_array(zeroth_cluster, mask=(zeroth_cluster == 0))

    H, dimensions = mf_hamiltonian(nspin, ntype,
                                   I, B, S, gyro_e, D, E, others, others_state)

    dms = compute_mf_dm(dm0, dimensions, states, H, S, timespace,
                        pulse_sequence, as_delay=as_delay) / zeroth_cluster

    return dms
