import numpy as np
from pycce.sm import _smc
from numba import jit
from numba.typed import List
import numba.types

def expand(matrix, i, dim):
    """
    Expand matrix M from it's own dimensions to the total Hilbert space
    :param matrix: ndarray
        Inital matrix
    :param i: int
        Index of the spin in dim
    :param dim: list
        list of dimensions of all spins present in the cluster
    :return: ndarray
        Expanded matrix
    """
    dbefore = np.asarray(dim[:i]).prod()
    dafter = np.asarray(dim[i + 1:]).prod()

    expanded_matrix = np.kron(np.kron(np.eye(dbefore, dtype=np.complex128), matrix),
                              np.eye(dafter, dtype=np.complex128))

    return expanded_matrix


def dimensions_spinvectors(nspin, central_spin=None):
    """
    Generate two arrays, containing dimensions of the spins in the cluster and the spinvectors
    :param nspin: BathArray
        BathArray subclass, containing spins within cluster
    :param central_spin: float, optional
        if provided, include dimensions of the central spin with the total spin s
    :return:
    """
    ntype = nspin.types
    spins = [ntype[n['N']].s for n in nspin]
    dimensions = [_smc[s].dim for s in spins]

    if central_spin is not None:
        dimensions += [_smc[central_spin].dim]
        spins += [central_spin]

    dimensions = np.asarray(dimensions, dtype=np.int32)
    vectors = []
    for j, s in enumerate(spins):
        vectors.append(spinvec(s, j, dimensions))
    return dimensions, vectors


def spinvec(s, j, dimensions):
    """
    Generate spin vector, containing 3 spin matrices in the total basis of the system
    :param s: float
        spin of the particle
    :param j: int
        particle number in dimensions array
    :param dimensions: ndarray
        ndarray containing dimensions of all spins
    :return: ndarray
    """
    vec = np.array([expand(_smc[s].x, j, dimensions),
                    expand(_smc[s].y, j, dimensions),
                    expand(_smc[s].z, j, dimensions)],
                   dtype=np.complex128)
    return vec


def generate_projections(state):
    """
    Generate <Sx>, <Sy>, and <Sz> projections of the given central spin state
    :param state: ndarray
        state of the central spin in Sz basis
    :return: ndarray of shape (3,)
    <Sx>, <Sy>, and <Sz> projections
    """
    spin = (state.size - 1) / 2
    sm = _smc[spin]

    projections = np.array([state.conj() @ sm.x @ state,
                            state.conj() @ sm.y @ state,
                            state.conj() @ sm.z @ state],
                           dtype=np.complex128)
    return projections


def zfs_tensor(D, E=0):
    """
    Generate (3,3) ZFS tensor from observables D and E parameters
    :param D: float or ndarray with shape (3,3)
        D parameter in central spin ZFS OR total ZFS tensor
    :param E: float
        E parameter in central spin ZFS
    :return: ndarray of shape (3,3)
    """
    if isinstance(D, (np.floating, float, int)):

        tensor = np.zeros((3, 3), dtype=np.float64)
        tensor[2, 2] = 2 / 3 * D
        tensor[1, 1] = -D / 3 - E
        tensor[0, 0] = -D / 3 + E
    else:
        tensor = D
    return tensor


def project_bath_states(states):
    ndstates = np.asarray(states)
    if len(ndstates.shape) > 1:
        spin = (ndstates.shape[1] - 1) / 2
        projected_bath_state = np.empty((ndstates.shape[0], 3))

        projected_bath_state[:, 0] = np.trace(np.matmul(ndstates, _smc[spin].x), axis1=1, axis2=2)
        projected_bath_state[:, 1] = np.trace(np.matmul(ndstates, _smc[spin].y), axis1=1, axis2=2)
        projected_bath_state[:, 2] = np.trace(np.matmul(ndstates, _smc[spin].z), axis1=1, axis2=2)


    elif ndstates.dtype == object:
        projected_bath_state = loop_trace(list(states))

    else:
        projected_bath_state = ndstates

    if len(projected_bath_state.shape) > 1 and not np.any(projected_bath_state[:, :2]):
        projected_bath_state = projected_bath_state[:, 2]

    return projected_bath_state


@jit(nopython=True)
def loop_trace(states):
    proj_states = np.empty((len(states), 3), dtype=np.complex128)
    dims = List()

    sx = List()
    sy = List()
    sz = List()

    for j, dm in enumerate(states):
        dm = dm.astype(np.complex128)
        dim = dm.shape[0]
        try:
            ind = dims.index(dim)
        except:
            sxnew, synew, sznew = gen_sm(dim)

            sx.append(sxnew)
            sy.append(synew)
            sz.append(sznew)
            dims.append(dim)

            ind = -1

        xproj = np.trace(dm @ sx[ind])
        yproj = np.trace(dm @ sy[ind])
        zproj = np.trace(dm @ sz[ind])

        proj_states[j, 0] = xproj
        proj_states[j, 1] = yproj
        proj_states[j, 2] = zproj

    return proj_states


@jit(nopython=True)
def gen_sm(dim):
    s = (dim - 1) / 2
    projections = np.linspace(s, -s, dim).astype(np.complex128)
    plus = np.zeros((dim, dim), dtype=np.complex128)

    for i in range(dim - 1):
        plus[i, i + 1] += np.sqrt(s * (s + 1) -
                                  projections[i] * projections[i + 1])

    minus = plus.conj().T
    x = 1 / 2. * (plus + minus)
    y = 1 / 2j * (plus - minus)
    z = np.diag(projections[::-1])
    return x, y, z
