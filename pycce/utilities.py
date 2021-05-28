import numpy as np
from pycce.sm import _smc
from numba import jit
from numba.typed import List


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
    dbefore = np.asarray(dim[:i]).prod()
    dafter = np.asarray(dim[i + 1:]).prod()

    expanded_matrix = np.kron(np.kron(np.eye(dbefore, dtype=np.complex128), matrix),
                              np.eye(dafter, dtype=np.complex128))

    return expanded_matrix


def dimensions_spinvectors(nspin, central_spin=None):
    """
    Generate two arrays, containing dimensions of the spins in the cluster and the vectors with spin matrices.

    Args:
        nspin (BathArray with shape (n,)): Array of the n spins within cluster.
        central_spin (float, optional): If provided, include dimensions of the central spin with the total spin s.

    Returns:
        tuple: *tuple* containing:
                * **ndarray with shape (n,)**: Array with dimensions for each spin.
                * **list**: List with vectors of spin matrices for each spin in the cluster
                  (Including central spin if ``central_spin`` is not None). Each with  shape (3, N, N) where
                  ``N = prod(dimensions)``.
    """

    ntype = nspin.types
    spins = [ntype[n].s for n in nspin['N']]
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


def generate_projections(state_a, state_b=None):
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

    spin = (state_a.size - 1) / 2
    sm = _smc[spin]

    projections = np.array([state_a.conj() @ sm.x @ state_b,
                            state_a.conj() @ sm.y @ state_b,
                            state_a.conj() @ sm.z @ state_b],
                           dtype=np.complex128)
    return projections


def zfs_tensor(D, E=0):
    """
    Generate (3, 3) ZFS tensor from observables D and E parameters.

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


def project_bath_states(states):
    """
    Generate projections of bath states on Sz axis from any type of states input.
    Args:
        states (array-like): Array of bath spin states.

    Returns:
        ndarray: Array of Sz projections of the bath states
    """
    ndstates = np.asarray(states)
    if len(ndstates.shape) > 1:
        spin = (ndstates.shape[1] - 1) / 2
        projected_bath_state = np.empty((ndstates.shape[0], 3))

        projected_bath_state[:, 0] = np.trace(np.matmul(ndstates, _smc[spin].x), axis1=1, axis2=2)
        projected_bath_state[:, 1] = np.trace(np.matmul(ndstates, _smc[spin].y), axis1=1, axis2=2)
        projected_bath_state[:, 2] = np.trace(np.matmul(ndstates, _smc[spin].z), axis1=1, axis2=2)

    elif ndstates.dtype == object:
        projected_bath_state = _loop_trace(list(states))

    else:
        projected_bath_state = ndstates

    if len(projected_bath_state.shape) > 1 and not np.any(projected_bath_state[:, :2]):
        projected_bath_state = projected_bath_state[:, 2]

    return projected_bath_state


@jit(nopython=True)
def _loop_trace(states):
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
            sxnew, synew, sznew = _gen_sm(dim)

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
def _gen_sm(dim):
    """
    Numba-friendly spin matrix.
    Args:
        dim (int): dimensions of the spin marix.

    Returns:
        ndarray:
    """
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


def transform(atoms, center=None, cell=None, rotation_matrix=None, style='col', inplace=True):
    """
    Coordinate transformation of BathArray.

    Args:
        atoms (BathArray): Array to be transformed.
        center (ndarray with shape (3,)): (0, 0, 0) position of new coordinates in the initial frame.
        cell (ndarray with shape (3, 3)): Cell vectors in cartesian coordinates
            if initial coordinates of the ``atoms`` are in crystallographic reference frame.
        rotation_matrix (ndarray with shape (3, 3)):
            Rotation matrix R of the **coordinate system**.

            E.g. ``R @ [0, 0, 1] = [a, b, c]`` where ``[a, b, c]`` are coordinates of the z axis of the new coordinate
            system in the old coordinate system.

            Note, that rotaton is applied after transition from cell coordinates to the cartesian coordinates,
            in which cell vectors are stored.

        style (str): Can have two values: 'col' or 'row'.
            Shows how ``cell`` and ``rotation_matrix`` matrices are given:

                * if 'col', each column of the matrix is a vector in previous coordinates;
                * if 'row' - each row is a new vector.

            Default 'col'.

        inplace (bool): If true, makes inplace changes to the provided array.

    Returns:
        BathArray: Transformed array with bath spins.
    """

    styles = ['col', 'row']
    if style not in styles:
        raise ValueError('Unsupported style of matrices. Available styles are: ' + ', '.join(*styles))

    if not inplace:
        atoms = atoms.copy()

    if len(atoms.shape) == 0:
        atoms = atoms[np.newaxis]

    if center is None:
        center = np.zeros(3)

    if cell is None:
        cell = np.eye(3)

    if rotation_matrix is None:
        rotation_matrix = np.eye(3)

    if style.lower() == 'row':
        cell = cell.T
        rotation_matrix = rotation_matrix.T

    if not atoms.dtype.names:
        atoms -= np.asarray(center)
        atoms = np.einsum('jk,ik->ij', cell, atoms)
        atoms = np.einsum('jk,ik->ij', np.linalg.inv(rotation_matrix), atoms)

        return atoms

    atoms['xyz'] -= np.asarray(center)

    atoms['xyz'] = np.einsum('jk,ik->ij', cell, atoms['xyz'])
    atoms['xyz'] = np.einsum('jk,ik->ij', np.linalg.inv(rotation_matrix), atoms['xyz'])

    if 'A' in atoms.dtype.names:
        atoms['A'] = np.matmul(atoms['A'], rotation_matrix)
        atoms['A'] = np.matmul(np.linalg.inv(rotation_matrix), atoms['A'])

    if 'Q' in atoms.dtype.names:
        atoms['Q'] = np.matmul(atoms['Q'], rotation_matrix)
        atoms['Q'] = np.matmul(np.linalg.inv(rotation_matrix), atoms['Q'])

    return atoms


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

#TODO Implement partial inner product instead of trace when MC bath state sampling.
def _partial_inner_product(a, total, dimensions, index=-1):
    matrix = np.moveaxis(total.reshape(dimensions), index, -1)
    matrix = matrix.reshape([np.prod(np.delete(dimensions, index)), dimensions[index]])
    return a @ matrix