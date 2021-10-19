import warnings

import numpy as np
from numba import jit
from numba.typed import List
from pycce.sm import _smc


def rotmatrix(initial_vector, final_vector):
    r"""
    Generate 3D rotation matrix which applied on initial vector will produce vector, aligned with final vector.

    Examples:

        >>> R = rotmatrix([0,0,1], [1,1,1])
        >>> R @ np.array([0,0,1])
        array([0.577, 0.577, 0.577])

    Args:
        initial_vector (ndarray with shape(3, )): Initial vector.
        final_vector (ndarray with shape (3, )): Final vector.

    Returns:
        ndarray with shape (3, 3): Rotation matrix.
    """

    iv = np.asarray(initial_vector)
    fv = np.asarray(final_vector)
    a = iv / np.linalg.norm(iv)
    b = fv / np.linalg.norm(fv)  # Final vector

    c = a @ b  # Cosine between vectors
    # if they're antiparallel
    if c == -1.:
        raise ValueError('Vectors are antiparallel')

    v = np.cross(a, b)
    screw_v = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r = np.eye(3) + screw_v + np.dot(screw_v, screw_v) / (1 + c)

    return r


@jit(nopython=True)
def expand(matrix, i, dim):
    """
    Expand matrix M from it's own dimensions to the total Hilbert space.

    Args:
        matrix (ndarray with shape (dim[i], dim[i])): Inital matrix.
        i (int): Index of the spin dimensions in ``dim`` parameter.
        dim (ndarray): Array pf dimensions of all spins present in the cluster.

    Returns:
        ndarray with shape (prod(dim), prod(dim)): Expanded matrix.
    """
    dbefore = dim[:i].prod()
    dafter = dim[i + 1:].prod()

    expanded_matrix = np.kron(np.kron(np.eye(dbefore, dtype=np.complex128), matrix),
                              np.eye(dafter, dtype=np.complex128))

    return expanded_matrix


def dimensions_spinvectors(bath=None, central_spin=None):
    """
    Generate two arrays, containing dimensions of the spins in the cluster and the vectors with spin matrices.

    Args:
        bath (BathArray with shape (n,)): Array of the n spins within cluster.
        central_spin (CenterArray, optional): If provided, include dimensions of the central spins.

    Returns:
        tuple: *tuple* containing:

            * **ndarray with shape (n,)**: Array with dimensions for each spin.

            * **list**: List with vectors of spin matrices for each spin in the cluster
              (Including central spin if ``central_spin`` is not None). Each with  shape (3, N, N) where
              ``N = prod(dimensions)``.
    """
    dimensions = []

    if bath is not None:
        dimensions += [n.dim for n in bath]

    if central_spin is not None:
        try:
            for c in central_spin:
                dimensions += [c.dim]

        except TypeError:
            dimensions += [central_spin.dim]

    dimensions = np.array(dimensions, dtype=np.int32)
    vectors = vecs_from_dims(dimensions)

    return dimensions, vectors


@jit(cache=True, nopython=True)
def vecs_from_dims(dimensions):
    td = dimensions.prod()
    vectors = np.zeros((len(dimensions), 3, td, td), dtype=np.complex128)
    for j, d in enumerate(dimensions):
        vectors[j] = spinvec(j, dimensions)
    return vectors


@jit(nopython=True)
def spinvec(j, dimensions):
    x, y, z = _gen_sm(dimensions[j])
    vec = np.stack((expand(x, j, dimensions),
                    expand(y, j, dimensions),
                    expand(z, j, dimensions))
                   )
    return vec


def generate_projections(state_a, state_b=None, spins=None):
    r"""
    Generate vector with the spin projections of the given spin states:

    .. math::

        [\bra{a}\hat{S}_x\ket{b}, \bra{a}\hat{S}_y\ket{b}, \bra{a}\hat{S}_z\ket{b}],

    where :math:`\ket{a}` and :math:`\ket{b}` are the given spin states.

    Args:
        state_a (ndarray): state `a` of the central spin in :math:`\hat{S}_z` basis.
        state_b (ndarray): state `b` of the central spin in :math:`\hat{S}_z` basis.

    Returns:
        ndarray with shape (3,): :math:`[\braket{\hat{S}_x}, \braket{\hat{S}_y}, \braket{\hat{S}_z}]` projections.
    """
    if state_b is None:
        state_b = state_a
    if spins is None:
        spin = (state_a.size - 1) / 2
        sm = _smc[spin]

        projections = np.array([state_a.conj() @ sm.x @ state_b,
                                state_a.conj() @ sm.y @ state_b,
                                state_a.conj() @ sm.z @ state_b],
                               dtype=np.complex128)
    else:
        projections = []
        dim = (np.asarray(spins) * 2 + 1 + 1e-8).astype(int)

        for i, s in enumerate(spins):
            sm = _smc[s]
            smx = expand(sm.x, i, dim)
            smy = expand(sm.y, i, dim)
            smz = expand(sm.z, i, dim)

            p = np.array([state_a.conj() @ smx @ state_b,
                          state_a.conj() @ smy @ state_b,
                          state_a.conj() @ smz @ state_b],
                         dtype=np.complex128)
            projections.append(p)

    return projections


def zfs_tensor(D, E=0):
    """
    Generate (3, 3) ZFS tensor from observable parameters D and E.

    Args:
        D (float or ndarray with shape (3, 3)): Longitudinal splitting (D) in ZFS **OR** total ZFS tensor.
        E (float): Transverse splitting (E) in ZFS.

    Returns:
        ndarray with shape (3, 3): Total ZFS tensor.
    """
    if isinstance(D, (np.floating, float, int)):

        tensor = np.zeros((3, 3), dtype=np.float64)
        tensor[2, 2] = 2 / 3 * D
        tensor[1, 1] = -D / 3 - E
        tensor[0, 0] = -D / 3 + E
    else:
        tensor = D
    return tensor


def project_bath_states(states, single=False):
    r"""
    Generate projections of bath states on :math:`S_z` axis from any type of states input.
    Args:
        states (array-like): Array of bath spin states.

    Returns:
        ndarray: Array of :math:`S_z` projections of the bath states
    """
    # Ask for single b/c check against shape cannot distinguish 2x2 dm and 2 vectors of 2
    # Other checks are kinda expensive

    ndstates = np.asarray(states)

    if not ndstates.shape and ndstates.dtype == object:
        ndstates = ndstates[()]
        single = True

    projected_bath_state = None

    if ndstates.dtype == object:

        try:
            ndstates = np.stack(ndstates)

        except ValueError:

            with warnings.catch_warnings(record=True) as w:
                projected_bath_state = _loop_trace(list(states))

    if projected_bath_state is None:
        spin = (ndstates.shape[1] - 1) / 2

        if len(ndstates.shape) == 2 + (not single):
            # projected_bath_state = np.empty((ndstates.shape[0], 3))

            # projected_bath_state[:, 0] = np.trace(np.matmul(ndstates, _smc[spin].x), axis1=-2, axis2=-1)
            # projected_bath_state[:, 1] = np.trace(np.matmul(ndstates, _smc[spin].y), axis1=-2, axis2=-1)
            projected_bath_state = np.trace(np.matmul(ndstates, _smc[spin].z), axis1=-2, axis2=-1).real  # [:, 2]

        else:
            # Assume vectors
            z_psi = np.einsum('ij,...j->...i', _smc[spin].z, ndstates)
            projected_bath_state = np.einsum('...j,...j->...', ndstates.conj(), z_psi).real
            # projected_bath_state = ndstates

    # if len(projected_bath_state.shape) > 1 and not np.any(projected_bath_state[:, :2]):
    #     projected_bath_state = projected_bath_state[:, 2]

    return projected_bath_state


@jit(nopython=True)
def _loop_trace(states):
    proj_states = np.empty((len(states),), dtype=np.float64)  # (len(states), 3)
    dims = List()

    # sx = List()
    # sy = List()
    sz = List()

    for j, dm in enumerate(states):
        dm = dm.astype(np.complex128)
        dim = dm.shape[0]
        try:
            ind = dims.index(dim)
        except:
            # sxnew, synew, sznew = _gen_sm(dim)
            sznew = _gen_sz(dim)
            # sx.append(sxnew)
            # sy.append(synew)
            sz.append(sznew)
            dims.append(dim)

            ind = -1
        if len(dm.shape) == 2:
            # xproj = np.trace(dm @ sx[ind])
            # yproj = np.trace(dm @ sy[ind])
            zproj = np.diag(dm @ sz[ind]).sum().real
        else:
            # xproj = dm.conj() @ sx[ind] @ dm
            # yproj = dm.conj() @ sy[ind] @ dm
            zproj = (dm.conj() @ sz[ind] @ dm).real

        # proj_states[j, 0] = xproj
        # proj_states[j, 1] = yproj
        proj_states[j] = zproj  # [j, 2]

    return proj_states


@jit(nopython=True)
def _gen_sz(dim):
    s = (dim - 1) / 2
    projections = np.linspace(-s, s, dim).astype(np.complex128)
    return np.diag(projections[::-1])


@jit(nopython=True)
def _gen_sm(dim):
    """
    Numba-friendly spin matrix.
    Args:
        dim (int): dimensions of the spin marix.

    Returns:
        ndarray:
    """
    s = (dim - 1) / 2
    projections = np.linspace(-s, s, dim).astype(np.complex128)
    plus = np.zeros((dim, dim), dtype=np.complex128)

    for i in range(dim - 1):
        plus[i, i + 1] += np.sqrt(s * (s + 1) -
                                  projections[i] * projections[i + 1])

    minus = plus.conj().T
    x = 1 / 2. * (plus + minus)
    y = 1 / 2j * (plus - minus)
    z = np.diag(projections[::-1])
    return x, y, z


def partial_inner_product(avec, total, dimensions, index=-1):
    r"""
    Returns partial inner product :math:`\ket{b}=\bra{a}\ket{\psi}`, where :math:`\ket{a}` provided by
    ``avec`` contains degrees of freedom to be "traced out" and :math:`\ket{\psi}` provided by ``total``
    is the total statevector.

    Args:
        avec (ndarray with shape (a,)):
        total (ndarray with shape (a*b,)):
        dimensions (ndarray with shape (n,)):
        index ():

    Returns:

    """
    if len(total.shape) == 1:
        matrix = np.moveaxis(total.reshape(dimensions), index, -1)
        matrix = matrix.reshape([np.prod(np.delete(dimensions, index)), dimensions[index]])
    else:
        total = total.reshape(total.shape[0], *dimensions)
        matrix = np.moveaxis(total, index, -1)
        matrix = matrix.reshape([total.shape[0], np.prod(np.delete(dimensions, index)), dimensions[index]])
    return avec @ matrix


@jit(nopython=True)
def shorten_dimensions(dimensions, central_number):
    if central_number > 1:
        shortdims = dimensions[:-central_number + 1].copy()
        # reduced dimension so all central spin dimensions are gathered in one
        shortdims[-1] = np.prod(dimensions[-central_number:])
    else:
        shortdims = dimensions
    return shortdims


@jit(nopython=True)
def gen_state_list(states, dims):
    list_of_vectors = List()
    for s, d in zip(states, dims):
        list_of_vectors.append(vector_from_s(s, d))
    return list_of_vectors


@jit(nopython=True)
def vector_from_s(s, d):
    vec_nucleus = np.zeros(d, dtype=np.complex128)
    state_number = np.int32((d - 1) / 2 - s)
    vec_nucleus[state_number] = 1
    return vec_nucleus


@jit(nopython=True)
def from_central_state(dimensions, central_state):

    return expand(central_state, len(dimensions) - 1, dimensions) / dimensions[:-1].prod()


@jit(nopython=True)
def from_none(dimensions):
    tdim = np.prod(dimensions)
    return np.eye(tdim) / tdim


@jit(nopython=True)
def from_states(states):
    cluster_state = states[0]
    for i in range(1, len(states)):
        cluster_state = np.kron(cluster_state, states[i])

    return cluster_state


def combine_cluster_central(cluster_state, central_state):
    lcs = len(cluster_state.shape)
    ls = len(central_state.shape)

    if lcs != ls:
        return noneq_cc(cluster_state, central_state)
    else:
        return eq_cc(cluster_state, central_state)


@jit(nopython=True)
def noneq_cc(cluster_state, central_state):
    if len(cluster_state.shape) == 1:
        matrix = np.outer(cluster_state, cluster_state)
        return np.kron(matrix, central_state)

    else:
        matrix = np.outer(central_state, central_state)
        return np.kron(cluster_state, matrix)


@jit(nopython=True)
def eq_cc(cluster_state, central_state):
    return np.kron(cluster_state, central_state)


@jit(nopython=True)
def rand_state(d):
    np.eye(d, dtype=np.complex128) / d


@jit(nopython=True)
def outer(s1, s2):
    return np.outer(s1, s2)


def generate_initial_state(dimensions, states=None, central_state=None):
    if states is None:
        if central_state is None:
            return from_none(dimensions)
        else:
            if len(central_state.shape) == 1:
                central_state = outer(central_state, central_state)
            return from_central_state(dimensions, central_state)

    has_none = not states.has_state.all()
    all_pure = False
    all_mixed = False

    if not has_none:
        all_pure = states.pure.all()
        if not all_pure:
            all_mixed = (~states.pure).all()

    if has_none:
        for i in range(states.size):
            if states[i] is None:
                states[i] = rand_state(dimensions[i])

    if not (all_pure or all_mixed):
        for i in range(states.size):
            if len(states[i].shape) < 2:
                states[i] = outer(states[i], states[i])

    cluster_state = from_states(tuple(states))

    if central_state is not None:
        cluster_state = combine_cluster_central(cluster_state, central_state)

    return cluster_state


@jit(nopython=True)
def tensor_vdot(tensor, ivec):
    result = np.zeros((tensor.shape[1], *ivec.shape[1:]), dtype=ivec.dtype)
    for i, row in enumerate(tensor):
        for j, a_ij in enumerate(row):
            result[i] += a_ij * ivec[j]
    return result

@jit(nopython=True)
def vvdot(vec_1, vec_2):
    result = np.zeros(vec_1.shape[1:], vec_1.dtype)
    for v1, v2 in zip(vec_1, vec_2):
        result += v1 @ v2
    return result



# oldfuncs


def generate_initial_state_old(dimensions, states=None, state0=None):
    if states is None:
        if state0 is not None:
            return expand(state0, len(dimensions) - 1, dimensions) / dimensions[:-1].prod()
        else:
            tdim = np.prod(dimensions)
            dmtotal0 = np.eye(tdim) / tdim

            return dmtotal0

    cluster_state = None
    check = False
    for i, s in enumerate(states):

        if s is None:
            s = np.eye(dimensions[i], dtype=np.complex128) / dimensions[i]

            if check:
                cluster_state = np.kron(cluster_state, s)
            else:
                cluster_state = s
                check = True

        else:
            # If we already started iterating
            if check:

                lcs = len(cluster_state.shape)
                ls = len(s.shape)

                if lcs != ls:

                    if len(cluster_state.shape) == 1:
                        cluster_state = np.outer(cluster_state, cluster_state)

                    else:
                        s = np.outer(s, s)
                cluster_state = np.kron(cluster_state, s)
            else:
                cluster_state = s
                check = True

    if state0 is not None:

        lcs = len(cluster_state.shape)
        ls = len(state0.shape)

        if lcs != ls:

            if len(cluster_state.shape) == 1:
                cluster_state = np.outer(cluster_state, cluster_state)

            else:
                state0 = np.outer(state0, state0)

        cluster_state = np.kron(cluster_state, state0)

    return cluster_state


def dimensions_spinvectors_old(bath=None, central_spin=None):
    """
    Generate two arrays, containing dimensions of the spins in the cluster and the vectors with spin matrices.

    Args:
        bath (BathArray with shape (n,)): Array of the n spins within cluster.
        central_spin (CenterArray, optional): If provided, include dimensions of the central spins.

    Returns:
        tuple: *tuple* containing:

            * **ndarray with shape (n,)**: Array with dimensions for each spin.

            * **list**: List with vectors of spin matrices for each spin in the cluster
              (Including central spin if ``central_spin`` is not None). Each with  shape (3, N, N) where
              ``N = prod(dimensions)``.
    """
    spins = []
    dimensions = []

    if bath is not None:
        types = bath.types

        spins += [types[n].s for n in bath.N]
        dimensions += [_smc[s].dim for s in spins]

    if central_spin is not None:
        try:
            for c in central_spin:
                dimensions += [c.dim]
                spins += [c.s]

        except TypeError:
            dimensions += [central_spin.dim]
            spins += [central_spin.s]

    dimensions = np.array(dimensions, dtype=np.int32)

    vectors = []

    for j, s in enumerate(spins):
        vectors.append(spinvec(s, j, dimensions))

    vectors = np.array(vectors)

    return dimensions, vectors


def spinvec_old(s, j, dimensions):
    """
    Generate spin vector for the particle, containing 3 spin matrices in the total basis of the system.

    Args:
        s (float): Spin of the particle.
        j (j): Particle index in ``dimensions`` array.
        dimensions (ndarray): Array with dimensions of all spins in the cluster.

    Returns:
        ndarray with shape (3, prod(dimensions), prod(dimensions)):
            Vector of spin matrices for the given spin in the cluster.
    """
    vec = np.array([expand(_smc[s].x, j, dimensions),
                    expand(_smc[s].y, j, dimensions),
                    expand(_smc[s].z, j, dimensions)],
                   dtype=np.complex128)
    return vec
