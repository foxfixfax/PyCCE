import copy
import warnings

import numpy as np
from pycce.bath.array import common_isotopes, common_concentrations
from pycce.constants import BOHR_TO_ANGSTROM

coord_types = ['crystal', 'bohr', 'angstrom', 'alat']


class DFTCoordinates:
    r"""
    Abstract class of a container of the DFT output coordinates.

    Attributes:
        alat (float): The lattice parameter in angstrom.
        cell (ndarray with shape (3, 3)):
            cell is 3x3 matrix with entries:

            .. math::

                [&[a_x\ b_x\ c_x]\\
                &[a_y\ b_y\ c_y]\\
                &[a_z\ b_z\ c_z]]


            where a, b, c are crystallographic vectors
            and x, y, z are their coordinates in the cartesian reference frame.

        coordinates (ndarray with shape (n, 3)): Array with the coordinates of atoms in the cell.

        names (ndarray with shape (n,)): Array with the names of atoms in the cell.

        cell_units (str): Units of cell coordinates: 'bohr', 'angstrom', 'alat'.

        coordinates_units (str): Units of atom coordinates: 'crystal', 'bohr', 'angstrom', 'alat'.
    """

    def __init__(self):
        self.alat = None
        self.cell = None

        self.coordinates = None
        self.names = None
        self.cell_units = None
        self.coordinates_units = None

    def to_angstrom(self, inplace=False):
        """
        Method to transform cell and coordinates units to angstroms.

        Args:
            inplace (bool): if True changes attributes inplace. Otherwise returns copy.

        Returns:
            DFTCoordinates or subclass:
                Instance of the subclass with units of coordinates and cell of Angstroms.
        """
        if inplace:
            obj = self
        else:
            obj = copy.deepcopy(self)

        obj.cell = change_to_angstrom(self.cell.T, self.cell_units, alat=self.alat).T
        obj.coordinates = change_to_angstrom(self.coordinates, self.coordinates_units, alat=obj.alat, cell=obj.cell)
        obj.cell_units = 'angstrom'
        obj.coordinates_units = 'angstrom'
        return obj

    def get_angstrom(self, coordinate, units):
        """
        Change given coordinates to angstrom.

        Args:
            coordinates (ndarray with shape (n, 3) or (3,)): Coordinates to change.
            units (str): Initial units of the coordinates.

        Returns:
            ndarray (n, 3): Coordinates in angstrom.
        """
        return change_to_angstrom(coordinate, units, alat=self.alat, cell=self.cell)

    def __repr__(self):
        m = f"{type(self).__name__} with coordinates in {self.coordinates_units} and cell in {self.cell_units}."
        return m


def change_to_angstrom(coordinates, units, alat=None, cell=None):
    r"""
    Change coordinates to angstrom.

    Args:
        coordinates (ndarray with shape (n, 3) or (3,)): Coordinates to change.
        units (str): Initial units of the coordinates.
        alat (float): The lattice parameter in angstrom.

        cell (ndarray with shape (3,3)):
            cell is 3x3 matrix with entries:

            .. math::

                [&[a_x\ b_x\ c_x]\\
                &[a_y\ b_y\ c_y]\\
                &[a_z\ b_z\ c_z]]

            where a, b, c are crystallographic vectors,
            and x, y, z are their coordinates in the cartesian reference frame.

    Returns:
        ndarray with shape (n, 3): Coordinates in angstrom.

    """
    coordinates = np.asarray(coordinates)

    if units == 'angstrom':
        coordinates = coordinates
    if units == 'bohr':
        coordinates = coordinates * BOHR_TO_ANGSTROM
    if units == 'alat':
        coordinates = coordinates * alat
    if units == 'crystal':
        if len(coordinates.shape) > 1:
            coordinates = np.einsum('jk,ik->ij', cell, coordinates)
        else:
            coordinates = cell @ coordinates

    return coordinates


def fortran_value(value):
    """
    Get value from Fortran-type variable.

    Args:
        value (str): Value read from Fortran-type input.

    Returns:
        value (bool, str, float): value in Python format.

    """
    bools = {".true.": True, ".false": False}
    str_separators = "'" '"'

    if value in bools:
        value = bools[value]

    elif value.strip('+-').isdigit():
        value = int(value)

    elif value[0] in str_separators:
        value = value.strip(str_separators)

    else:
        try:
            value = float(value.replace('d', 'e'))
        except ValueError:
            raise ValueError(f'{value} is incorrect')

    return value


def yield_index(word, lines, start=0, case_sensitive=False):
    """
    Generator which yields indexes of the lines containing specific word.

    Args:
        word (str): Word to find in the line.
        lines (list of str): List of strings contraining lines from the file. Output of open(file).readlines().
        start (int): First index from which to start search.
        case_sensitive (bool): If True looks for the exact match. Otherwise the search is case insensitive.

    Yields:
        i (int): Index of the line containing word.
    """

    if not case_sensitive:
        word = word.lower()

    for i in range(start, len(lines)):
        lin = lines[i]
        if not case_sensitive:
            lin = lin.lower()

        if word in lin:
            yield i


def find_first_index(word, lines, start=0, case_sensitive=False):
    """
    Function to find first appearance of the index in the list of lines.

    Args:
        word (str): Word to find in the line.
        lines (list of str): List of strings contraining lines from the file. Output of open(file).readlines().
        start (int): First index from which to start search.
        case_sensitive (bool): If True looks for the exact match. Otherwise the search is case insensitive.

    Returns:
        i (int): Index of the first line from the start containing word.

    """
    try:
        ind = next(yield_index(word, lines, start=start, case_sensitive=case_sensitive))
    except StopIteration:
        raise KeyError(f'Could not find {word}')
    return ind


def set_isotopes(array, isotopes=None, inplace=True, spin_types=None):
    """
    Function to set the most common isotopes for the array containing DFT output. If some other isotope
    is specified, the A tensors are scaled accordingly.

    Args:
        array (BathArray): Array with DFT spins.
        isotopes (dict): Dictionary with chosen isotopes.
        inplace (bool): True if change the array inplace.
        spin_types (SpinDict): If provided, allows for custom defined ``SpinType`` instances.

    Returns:
        array (BathArray): Array with DFT spins with correct isotopes.

    """
    if not inplace:
        array = array.copy()

    if spin_types is None:
        spin_types = common_isotopes
    else:
        array.add_type(spin_types)

    for n in np.unique(array.N):
        try:
            isotopes_of_n = common_concentrations[n]
            names = list(isotopes_of_n.keys())
            conc = [isotopes_of_n[k] for k in names]
            which = names[np.argmax(conc)]

            if isotopes is not None and n in isotopes:
                factor = spin_types[isotopes[n]].gyro / spin_types[which].gyro
                which = isotopes[n]
            else:
                factor = 1

            where = array.N == n

            array.A[where] *= factor
            array.N[where] = which

            array.types[which]

        except KeyError:
            warnings.warn(f'Unknown element {n}. Skipping')
    return array
