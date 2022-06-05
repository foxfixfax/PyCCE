import warnings

import numpy as np
from numba import jit
from numba.typed import List


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


def partial_trace(dmarray, dimensions, sel):
    """
    Compute partial trace of the operator (or array of operators).

    Args:
        dmarray (ndarray with shape (N, N) or (m, N, N):
        dimensions (array-like): Array of all dimensions of the system.
        sel (int or array-like): Index or indexes of dimensions to keep.

    Returns:
        ndarray with shape (n, n) or (m, n, n): Partially traced operator.
    """
    sel = np.asarray(sel, dtype=int).reshape(-1)
    dmarray = dmarray.copy()
    dimensions = np.asarray(dimensions, dtype=int)
    lendim = len(dimensions)

    initial_shape = dmarray.shape

    if len(initial_shape) > 2:
        dmarray.shape = (initial_shape[0], *dimensions, *dimensions)
        add = 1
    else:
        add = 0
        dmarray.shape = (*dimensions, *dimensions)

    indexes = np.delete(np.arange(lendim, dtype=int), sel)
    dims = dimensions.copy()
    for ind in indexes:
        dims[ind] = 1
        dmarray = np.trace(dmarray, axis1=ind + add, axis2=lendim + ind + add)
        if add:
            dmarray = dmarray.reshape(initial_shape[0], *dims, *dims)
        else:
            dmarray = dmarray.reshape(*dims, *dims)
    after_tr = dimensions[sel].prod()
    if add:
        dmarray = dmarray.reshape(initial_shape[0], after_tr, after_tr)
    else:
        dmarray = dmarray.reshape(after_tr, after_tr)

    return dmarray

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


@jit(cache=True, nopython=True)
def tensor_vdot(tensor, ivec):
    """
    Compute product of the tensor and spin vector.

    Args:
        tensor (ndarray with shape (3, 3)): Tensor in real space.
        ivec (ndarray with shape (3, n, n)): Spin vector.

    Returns:
        ndarray with shape (3, n, n): Right-side tensor vector product :math:`Tv`.
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


def _add_args(after):
    def prepare_with_args(func):
        if func.__doc__ is None:
            func.__doc__ = after
        else:
            func.__doc__ = func.__doc__ + after
        return func

    return prepare_with_args


@jit(cache=True, nopython=True)
def vec_tensor_vec(v1, tensor, v2):
    """
    Compute product v @ T @ v.
    Args:
        v1 (ndarray with shape (3, n, n)): Leftmost expanded spin vector.
        tensor (ndarray with shape (3, 3)): 3x3 interaction tensor in real space.
        v2 (ndarray with shape (3, n, n)): Rightmost expanded spin vector.

    Returns:
        ndarray with shape (n, n): Product :math:`vTv`.

    """
    t_vec = tensor_vdot(tensor, v2)
    return vvdot(v1, t_vec)


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
