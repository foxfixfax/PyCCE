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


@jit(cache=True, nopython=True)
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
    """
    Generate ndarray of spin vectors, given the array of spin dimensions.

    Args:
        dimensions (ndarray with shape (n,)): Dimensions of spins.

    Returns:
        ndarray with shape (n, 3, X, X): Array of spin vectors in full Hilbert space.
    """
    td = dimensions.prod()
    vectors = np.zeros((len(dimensions), 3, td, td), dtype=np.complex128)
    for j, d in enumerate(dimensions):
        vectors[j] = spinvec(j, dimensions)
    return vectors


@jit(cache=True, nopython=True)
def spinvec(j, dimensions):
    """
    Generate single spin vector, given the index and dimensions of all spins in the cluster.

    Args:
        j (int): Index of the spin.
        dimensions (ndarray with shape (n,)): Dimensions of spins.

    Returns:
        ndarray with shape (3, X, X): Spin vector of :math:`j`-sth spin in full Hilbert space.

    """
    x, y, z = numba_gen_sm(dimensions[j])
    vec = np.stack((expand(x, j, dimensions),
                    expand(y, j, dimensions),
                    expand(z, j, dimensions))
                   )
    return vec


def generate_projections(state_a, state_b=None, spins=None):
    r"""
    Generate vector or list of vectors (if ``spins`` is not None) with the spin projections of the given spin states:

    .. math::

        [\bra{a}\hat{S}_x\ket{b}, \bra{a}\hat{S}_y\ket{b}, \bra{a}\hat{S}_z\ket{b}],

    where :math:`\ket{a}` and :math:`\ket{b}` are the given spin states.

    Args:
        state_a (ndarray): State :math:`\ket{a}` of the central spin or spins in :math:`\hat{S}_z` basis.
        state_b (ndarray): State :math:`\ket{b}` of the central spin or spins in :math:`\hat{S}_z` basis.
        spins (ndarray, optional): Array of spins, comprising the given state vectors.
            If provided, assumes that states correspond to a Hilbert space of several spins, and projections
            of states are computed for each spin separately.
    Returns:
        ndarray with shape (3,) or list:
            :math:`[\braket{\hat{S}_x}, \braket{\hat{S}_y}, \braket{\hat{S}_z}]` projections or list of projections.
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
    D = np.asarray(D)

    if D.size == 1:

        tensor = np.zeros((3, 3), dtype=np.float64)
        tensor[2, 2] = 2 / 3 * D
        tensor[1, 1] = -D / 3 - E
        tensor[0, 0] = -D / 3 + E

    else:
        tensor = D

    return tensor


@jit(cache=True, nopython=True)
def numba_gen_sm(dim):
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


@jit(cache=True, nopython=True)
def shorten_dimensions(dimensions, central_number):
    """
    Combine the dimensions, corresponding to the central spins.

    Args:
        dimensions (ndarray with shape (n, )): Array of the dimensions of the spins in the cluster.
        central_number (int): Number of central spins.

    Returns:
        ndarray with shape (n - central_number): Array of the shortened dimensions;

    """
    if central_number > 1:
        shortdims = dimensions[:-central_number + 1].copy()
        # reduced dimension so all central spin dimensions are gathered in one
        shortdims[-1] = np.prod(dimensions[-central_number:])
    else:
        shortdims = dimensions
    return shortdims


@jit(cache=True, nopython=True)
def gen_state_list(states, dims):
    """
    Generate list of states from :math:`S_z` projections of the pure states.

    Args:
        states (ndarray with shape (n,)): Array of :math:`S_z` projections.
        dims (ndarray with shape (n,)): Array of the dimensions of the spins in the cluster.

    Returns:
        List: list of state vectors.

    """
    list_of_vectors = List()
    for s, d in zip(states, dims):
        list_of_vectors.append(vector_from_s(s, d))
    return list_of_vectors


@jit(cache=True, nopython=True)
def vector_from_s(s, d):
    """
    Generate vector state from :math:`S_z` projection.

    Args:
        s (float): :math:`S_z` projection.
        d (int): Dimensions of the given spin.

    Returns:
        ndarray with shape (d, ): State vector of a pure state.
    """
    vec_nucleus = np.zeros(d, dtype=np.complex128)
    state_number = np.int32((d - 1) / 2 - s)
    vec_nucleus[state_number] = 1
    return vec_nucleus


@jit(cache=True, nopython=True)
def from_central_state(dimensions, central_state):
    """
    Generate density matrix of the system if all spins apart from central spin are in completely mixed state.

    Args:
        dimensions (ndarray with shape (n,)): Array of the dimensions of the spins in the cluster.
        central_state (ndarray with shape (x,)): Density matrix of central spins.

    Returns:
        ndarray with shape (N, N): Density matrix for the whole cluster.
    """

    return expand(central_state, len(dimensions) - 1, dimensions) / dimensions[:-1].prod()


@jit(cache=True, nopython=True)
def from_none(dimensions):
    """
    Generate density matrix of the systems if all spins are in completely mixed state.
    Args:
        dimensions (ndarray with shape (n,)): Array of the dimensions of the spins in the cluster.

    Returns:
        ndarray with shape (N, N): Density matrix for the whole cluster.

    """
    tdim = np.prod(dimensions)
    return np.eye(tdim) / tdim


def from_states(states):
    """
    Generate density matrix of the systems if all spins are in pure states.
    Args:
        states (array-like): Array of the pure spin states.

    Returns:
        ndarray with shape (N, N): Spin vector for the whole cluster.

    """
    cluster_state = states[0]
    for s in states[1:]:
        cluster_state = np.kron(cluster_state, s)

    return cluster_state


def combine_cluster_central(cluster_state, central_state):
    """
    Combine bath spin states and the state of central spin.
    Args:
        cluster_state (ndarray with shape (n,) or (n, n)): State vector or density matrix of the bath spins.
        central_state (ndarray with shape (m,) or (m, m)): State vector or density matrix of the central spins.

    Returns:
        ndarray with shape (mn, ) or (mn, mn): State vector or density matrix of the full system.
    """
    lcs = len(cluster_state.shape)
    ls = len(central_state.shape)

    if lcs != ls:
        return _noneq_cc(cluster_state, central_state)
    else:
        return _eq_cc(cluster_state, central_state)


@jit(cache=True, nopython=True)
def _noneq_cc(cluster_state, central_state):
    if len(cluster_state.shape) == 1:
        matrix = outer(cluster_state, cluster_state)
        return np.kron(matrix, central_state)

    else:
        matrix = outer(central_state, central_state)
        return np.kron(cluster_state, matrix)


@jit(cache=True, nopython=True)
def _eq_cc(cluster_state, central_state):
    return np.kron(cluster_state, central_state)


@jit(cache=True, nopython=True)
def rand_state(d):
    """
    Generate random state of the spin.

    Args:
        d (int): Dimensions of the spin.

    Returns:
        ndarray with shape (d, d): Density matrix of the random state.
    """
    return np.eye(d, dtype=np.complex128) / d


@jit(cache=True, nopython=True)
def outer(s1, s2):
    """
    Outer product of two complex vectors :math:`\ket{s_1}\bra{s_2}`.

    Args:
        s1 (ndarray with shape (n, )): First vector.
        s2 (ndarray with shape (m, )): Second vector.

    Returns:
        ndarray with shape (n, m): Outer product.
    """
    return np.outer(s1, s2.conj())


def generate_initial_state(dimensions, states=None, central_state=None):
    """
    Generate initial state of the cluster.

    Args:
        dimensions (ndarray with shape (n, )): Dimensions of all spins in the cluster.
        states (BathState, optional): States of the bath spins. If None, assumes completely random state.
        central_state (ndarray): State of the central spin. If None, assumes that no central spin is present
            in the Hilbert space of the cluster.

    Returns:
        ndarray with shape (N,) or (N, N): State vector or density matrix of the cluster.

    """
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

    cluster_state = from_states(states)

    if central_state is not None:
        cluster_state = combine_cluster_central(cluster_state, central_state)

    return cluster_state


@jit(cache=True, nopython=True)
def tensor_vdot(tensor, ivec):
    """
    Compute product of the tensor and spin vector.

    Args:
        tensor ():
        ivec ():

    Returns:

    """
    result = np.zeros((tensor.shape[1], *ivec.shape[1:]), dtype=ivec.dtype)
    for i, row in enumerate(tensor):
        for j, a_ij in enumerate(row):
            result[i] += a_ij * ivec[j]
    return result


@jit(cache=True, nopython=True)
def vvdot(vec_1, vec_2):
    """
    Compute product of two spin vectors.

    Args:
        vec_1 (ndarray with shape (3, N, N)): First spin vector.
        vec_2 (ndarray with shape (3, N, N)): Second spin vector.

    Returns:
        ndarray with shape (N, N): Product of two vectors.
    """
    result = np.zeros(vec_1.shape[1:], vec_1.dtype)
    for v1, v2 in zip(vec_1, vec_2):
        result += v1 @ v2
    return result


def rotate_tensor(tensor, rotation=None, style='col'):
    """
    Rootate tensor in real space, given rotation matrix.

    Args:
        tensor (ndarray with shape (3, 3)): Tensor to be rotated.
        rotation (ndarray with shape (3, 3)): Rotation matrix.
        style (str): Can be 'row' or 'col'. Determines how rotation matrix is initialized.

    Returns:
        ndarray with shape (3, 3): Rotated tensor.
    """
    if rotation is None:
        return tensor
    if style.lower == 'row':
        rotation = rotation.T
    if np.isclose(np.linalg.inv(rotation), rotation.T, rtol=1e-04).all():
        invrot = rotation.T
    else:
        warnings.warn(f"Rotation {rotation} changes distances. Is that desired behavior?", stacklevel=2)
        invrot = np.linalg.inv(rotation)
    tensor_rotation = np.matmul(tensor, rotation)
    res = np.matmul(invrot, tensor_rotation)
    #  Suppress very small deviations
    res[np.isclose(res, 0)] = 0
    return (res + np.swapaxes(res, -1, -2)) / 2


def rotate_coordinates(xyz, rotation=None, cell=None, style='col'):
    """
    Rootate coordinates in real space, given rotation matrix.

    Args:
        xyz (ndarray with shape (..., 3)): Array of coordinates.
        rotation (ndarray with shape (3, 3)): Rotation matrix.
        cell (ndarray with shape (3, 3)): Cell matrix if coordinates are given in cell coordinates.
        style (str): Can be 'row' or 'col'. Determines how rotation matrix and cell matrix are initialized.

    Returns:
        ndarray with shape (..., 3)): Array of rotated coordinates.
    """
    if style.lower() == 'row':
        if rotation is not None:
            rotation = rotation.T
        if cell is not None:
            cell = cell.T
    if cell is not None:
        xyz = np.einsum('jk,...k->...j', cell, xyz)
    if rotation is not None:
        if np.isclose(np.linalg.inv(rotation), rotation.T).all():
            invrot = rotation.T
        else:
            warnings.warn(f"Rotation {rotation} changes distances. Is that desired behavior?", stacklevel=2)
            invrot = np.linalg.inv(rotation)

        xyz = np.einsum('jk,...k->...j', invrot, xyz)
    #  Suppress very small deviations
    xyz[np.isclose(xyz, 0)] = 0

    return xyz


def normalize(vec):
    """
    Normalize vector to 1.

    Args:
        vec (ndarray with shape (n, )): Vector to be normalized.

    Returns:
        ndarray with shape (n, ): Normalized vector.
    """
    vec = np.asarray(vec, dtype=np.complex128)
    return vec / np.linalg.norm(vec)
