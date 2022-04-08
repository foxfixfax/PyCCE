import warnings

import numpy as np
from numba import jit

from .array import BathArray, check_gyro
from ..constants import BOHR_TO_ANGSTROM, HBAR_MU0_O4PI, ELECTRON_GYRO, PI2

# Copied from ASE
chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'spin_matrix', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']


class Cube:
    """
    Class to process the .cube datafiles with spin polarization.

    Args:
        filename (str): Name of the .cube file.

    Attributes:
        comments (str): First two lines of the .cube file.
        origin (ndarray with shape (3,)): Coordinates of the origin in angstrom.
        voxel (ndarray with shape (3,3)): Parameters of the voxel - unit of the 3D grid in angstrom.
        size (ndarray with shape (3,)): Size of the cube.
        atoms (BathArray with shape (n)): Array of atoms in the cube.
        data (ndarray with shape (size[0], size[1], size[2]): Data stored in cube.
        grid (ndarray with shape (size[0], size[1], size[2], 3): Coordinates of the points at which data is computed.
        integral (float): Data integrated over cube.
        spin (float): integral / 2 - total spin.
    """
    _dt = np.dtype([('N', np.unicode_, 16), ('xyz', np.float64, (3,))])

    def __init__(self, filename):

        with open(filename, "r") as content:
            # first two lines are comments
            self.comments = next(content).strip() + "\n" + next(content).strip()

            # total number of bath | xyz of the cube origin
            tot = next(content).split()
            natoms = int(tot[0])

            self.origin = np.array([float(x) for x in tot[1:]]) * BOHR_TO_ANGSTROM
            self.voxel = np.empty([3, 3], dtype=np.float64)
            self.size = np.empty(3, dtype=np.int32)

            self.atoms = BathArray((natoms,))

            for i in range(3):
                tot = next(content).split()
                self.size[i] = int(tot[0])

                if self.size[i] < 0:
                    self.voxel[i] = [float(x) for x in tot[1:]]

                else:
                    self.voxel[i] = [float(x) * BOHR_TO_ANGSTROM for x in tot[1:]]
            with warnings.catch_warnings(record=True):
                for j in range(natoms):
                    tot = next(content).split()
                    self.atoms[j]['N'] = chemical_symbols[int(tot[0])]
                    self.atoms[j]['xyz'] = [float(x) for x in tot[2:]]

            if self.size[0] > 0:
                self.atoms['xyz'] *= BOHR_TO_ANGSTROM

            data = [float(x) for line in content for x in line.split()]

        if self.size[0] < 0:
            self.data = np.array(data).reshape(np.abs(self.size))
        else:
            self.data = np.array(data).reshape(np.abs(self.size)) / (BOHR_TO_ANGSTROM ** 3)

        # detect if diagonal
        # (see https://stackoverflow.com/questions/43884189/check-if-a-large-matrix-is-diagonal-matrix-in-python)
        if np.any(self.voxel.reshape(-1)[:-1].reshape(2, 4)[:, 1:]):
            warnings.warn('Voxel might be non-orthogonal. Correctness of the results is not tested')

        na = np.newaxis
        a = np.arange(self.size[0])[:, na] * self.voxel[0][na, :]
        b = np.arange(self.size[1])[:, na] * self.voxel[1][na, :]
        c = np.arange(self.size[2])[:, na] * self.voxel[2][na, :]
        mesh = a[:, na, na, :] + b[na, :, na, :] + c[na, na, :, :]

        self.grid = mesh + self.origin
        self.integral = np.trapz(np.trapz(np.trapz(self.data))) * np.linalg.det(self.voxel)
        self.spin = round(self.integral) * 0.5

    def transform(self, rotmatrix=None, shift=None):
        r"""
        Changes coordinates of the grid. DOES NOT ASSUME PERIODICITY.

        Args:
            rotmatrix (ndarray with shape (3, 3)): Rotation matrix `R`:

                .. math::

                    R =  &[n_1^{(1)} n_1^{(2)} n_1^{(3)}]\\
                         &[n_2^{(1)} n_2^{(2)} n_2^{(3)}]\\
                         &[n_3^{(1)} n_3^{(2)} n_3^{(3)}]

                where :math:`n_i^{(j)}` corresponds to the coefficient of initial basis vector :math:`i`
                for :math:`j` new basis vector:

                .. math::

                    e'_j = n_1^{(j)} \vec{e}_1 + n_2^{(j)} \vec{e}_2 + n_3^{(j)} \vec{e}_3

                In other words, columns of `R` are coordinates of the new
                basis in the old basis.

                Given vector in initial basis v = [v1, v2, v3],
                vector in new basis is given as v' = R.T @ v.

            shift (ndarray with shape (3,)): Shift in the origin of coordinates (in the old basis).

        """

        if shift is not None:
            shift = np.asarray(shift)
            self.grid = self.grid + shift

            self.atoms.xyz += shift
            self.origin += shift

        if rotmatrix is not None:
            assert (np.isclose(np.linalg.det(rotmatrix), 1.)), 'Determinant of R is not equal to 1'
            self.grid = np.einsum('ij,abcj->abci', rotmatrix.T, self.grid)
            self.origin = rotmatrix.T @ self.origin
            self.atoms.transform(rotation_matrix=rotmatrix)
        return

    def integrate(self, position, gyro_n, gyro_e=ELECTRON_GYRO, spin=None, parallel=False, root=0):
        """
        Integrate over polarization data, stored in Cube object,
        to obtain hyperfine dipolar-dipolar tensor.

        Args:
            position (ndarray with shape (3,) or (n, 3)): Position of the bath spin at which to compute
                hyperfine tensor or array of positions.
            gyro_n (float or ndarray with shape (n,) ): Gyromagnetic ratio of the bath spin or array of the ratios.
            gyro_e (float): Gyromagnetic ratio of central spin.
            spin (float): Total spin of the central spin. If not given, taken from the integral of the polarization.

        Returns:
            ndarray with shape (3, 3) or (n, 3, 3): Hyperfine tensor or array of hyperfine tensors.
        """
        if parallel:
            try:
                import mpi4py
            except ImportError:
                warnings.warn('Could not find mpi4py. Using the serial implementation instead')
                parallel = False

        if parallel:
            comm = mpi4py.MPI.COMM_WORLD

            size = comm.Get_size()
            rank = comm.Get_rank()

        else:
            rank = root

        if spin is None:
            spin = self.spin

        if np.around(spin * 2) != np.around(self.integral):
            warnings.warn(f'provided spin: {spin:.2f} is not equal to one from spin density: {self.integral / 2:.2f}')

        if rank == root:
            gyro_e, _ = check_gyro(gyro_e)
            gyro_n, _ = check_gyro(gyro_n)

            gyro_e = np.asarray(gyro_e)
            gyro_n = np.asarray(gyro_n)
            position = np.asarray(position)

        if parallel:
            gyro_e = comm.bcast(gyro_e, root)
            gyro_n = comm.bcast(gyro_n, root)
            position = comm.bcast(position, root)

        if len(position.shape) <= 1:
            return _cube_integrate(self.data, self.grid, self.voxel, spin,
                                   position, gyro_n, gyro_e)

        npos = position.shape[0]
        nshape = (npos, *gyro_n.shape[1:])
        gyro_n = np.array(np.broadcast_to(gyro_n, nshape))
        if gyro_n.ndim < 3:
            gyro_n = gyro_n[:, np.newaxis, ...]
        if parallel:
            start, end = _distribute(rank, size, npos)
            position = position[start:end]
            gyro_n = gyro_n[start:end]

        if position.size:
            hyperfine = _cube_integrate_array(self.data, self.grid, self.voxel, spin, position,
                                              gyro_n, gyro_e)
        else:
            hyperfine = np.zeros((0, 3, 3), dtype=np.float64)

        if parallel:

            total_hyperfine = np.zeros((npos * 3 * 3), dtype=np.float64)

            start_ends = tuple(_distribute(i, size, npos) for i in range(size))

            sendcounts = tuple((end - start) * 3 * 3 for (start, end) in start_ends)
            displacements = tuple(start * 3 * 3 for (start, end) in start_ends)

            comm.Allgatherv([hyperfine.flatten(), mpi4py.MPI.DOUBLE],
                            [total_hyperfine, sendcounts, displacements, mpi4py.MPI.DOUBLE])

            total_hyperfine = total_hyperfine.reshape(npos, 3, 3)
            comm.Barrier()

        else:
            total_hyperfine = hyperfine

        return total_hyperfine

        # d['A'] = -(3 * np.outer(pos, pos) - identity * r ** 2) / (r ** 5) * pre


def _distribute(rank, size, number):
    remainder = number % size
    add = int(rank < remainder)
    each = number // size
    start = rank * each + rank if rank < remainder else rank * each + remainder
    end = start + each + add

    return start, end


@jit(nopython=True)
def _cube_integrate(data, grid, voxel, spin, position, gyro_n, gyro_e=ELECTRON_GYRO):
    """
    Integrate cube for one position.

    Args:
        data (ndarray): 3D data.
        grid (ndarray): Grid over which to integrate.
        voxel (ndarray): Parameters of the voxel.
        spin (float): Total central spin.
        position (ndarray with shape (3,)): Position of the bath spin at which to compute
            hyperfine tensor
        gyro_n (ndarray): Gyromagnetic ratio of the bath spin.
        gyro_e (ndarray): Gyromagnetic ratio of central spin.

    Returns:
        ndarray with shape (3, 3): Hyperfine tensor.
    """
    pos = grid - position

    dist = np.sqrt(np.sum(pos ** 2, axis=-1))

    hyperfine = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):

            if i == j:
                integrand = - data * (3 * pos[:, :, :, i] * pos[:, :, :, j] - dist ** 2) / dist ** 5
            else:
                integrand = - data * (3 * pos[:, :, :, i] * pos[:, :, :, j]) / dist ** 5

            hyperfine[i, j] = np.trapz(np.trapz(np.trapz(integrand))) * HBAR_MU0_O4PI / (
                    2 * spin * PI2) * np.linalg.det(voxel)

    if gyro_e.ndim < 2:
        hyperfine = hyperfine * gyro_e
    else:
        hyperfine = gyro_e @ hyperfine

    if gyro_n.ndim < 2:
        hyperfine = hyperfine * gyro_n
    else:
        hyperfine = hyperfine @ gyro_n

    return hyperfine


@jit(nopython=True)
def _cube_integrate_array(data, grid, voxel, spin, coordinates, gyros, gyro_e=ELECTRON_GYRO):
    """
    Integrate cube for array of positions.
    Args:
        data (ndarray): 3D data.
        grid (ndarray): Grid over which to integrate.
        voxel (ndarray): Parameters of the voxel.
        spin (float): Total central spin.
        coordinates (ndarray with shape (n, 3)): Positions of the bath spins at which to compute
                hyperfine tensors.
        gyros (ndarray with shape (n,) ): Array of gyromagnetic ratio of the bath spins.
        gyro_e (ndarray): Gyromagnetic ratio of central spin.

    Returns:
        ndarray with shape (n, 3, 3): Array of hyperfine tensors.
    """
    hyperfines = np.zeros((coordinates.shape[0], 3, 3), dtype=np.float64)

    for i, position in enumerate(coordinates):
        hyperfines[i] = _cube_integrate(data, grid, voxel, spin, position, gyros[i, ...], gyro_e)

    return hyperfines
