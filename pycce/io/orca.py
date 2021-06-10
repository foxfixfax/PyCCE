from .base import DFTCoordinates, find_first_index, yield_index, set_isotopes
import numpy as np
from pycce.constants import MHZ_TO_KHZ, EFG_CONVERSION
from pycce.bath.array import BathArray, transform
import re
import warnings

def read_orca(fname, isotopes=None, types=None, center=None,
              find_isotopes=True, rotation_matrix=None, rm_style='col'):
    """
    Function to read ORCA output containing the hyperfines couplings and EFG tensors.

    if ``find_isotopes`` is set to True changes the names of the atoms to the most abundant isotopes.
    If that is not the desired outcome, user can define which isotopes to use using keyword isotopes.

    Args:
        fname (str): file name of the ORCA output.
        isotopes (dict): Optional.
            Dictionary with entries::

            {"element" : "isotope"}

            where "element" is the name of the element in DFT output, "isotope" is the name of the isotope.

        types (SpinDict or list of tuples): SpinDict containing SpinTypes of isotopes or input to make one.
        center (ndarray of shape (3,)): position of (0, 0, 0) point in the DFT coordinates.
        rotation_matrix (ndarray of shape (3,3)):
            Rotation matrix to rotate basis. For details see utilities.transform.
        rm_style (str):
            Indicates how rotation matrix should be interpreted.
            Can take values "col" or "row". Default "col"
        find_isotopes (bool): If true, sets isotopes instead of names of the atoms.

    Returns:
        BathArray:
            Array of bath spins with hyperfine couplings and quadrupole tensors from Orca output.

    """
    lines = open(fname).readlines()
    output = ORCACoordinates(lines)
    with warnings.catch_warnings(record=True):
        atoms = BathArray(array=output.coordinates, names=output.names, types=types)

    start = find_first_index('ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE', lines)

    efgs = []
    hyperfines = []
    a_indexes = []
    q_indexes = []
    for ind in yield_index('Nucleus', lines, start=start, case_sensitive=True):
        sline = lines[ind].split()
        atom_index = int(re.sub('\D', '', sline[1]))
        atom_name = re.sub("\d+", "", sline[1])

        # assert atoms[atom_index]['N'] == atom_name, (f"Name mismatch at {atom_index}{atom_name}"
        #                                              f" {atoms[atom_index]['N']} expected")

        try:
            n = find_first_index('Raw HFC matrix', lines, start=ind) + 2
            tensor = []
            for _ in range(3):
                line = lines[n]
                tensor.append([float(x) * MHZ_TO_KHZ for x in line.split()])
                n += 1
            hyperfines.append(tensor)
            a_indexes.append(atom_index)
        except KeyError:
            pass

        try:
            n = find_first_index('Raw EFG matrix', lines, start=ind) + 1
            tensor = []
            for _ in range(3):
                line = lines[n]
                tensor.append([float(x) * EFG_CONVERSION for x in line.split()])
                n += 1
            efgs.append(tensor)
            q_indexes.append(atom_index)
        except KeyError:
            pass

    if hyperfines:
        atoms.A[a_indexes] = hyperfines

    if find_isotopes:
        set_isotopes(atoms, isotopes=isotopes, spin_types=types)

    if efgs:
        atoms[q_indexes] = atoms[q_indexes].from_efg(efgs)

    atoms = transform(atoms, center, rotation_matrix=rotation_matrix, style=rm_style)

    return atoms


class ORCACoordinates(DFTCoordinates):
    r"""
    Coordinates of the system from the ORCA output. Subclass of the DFTCoordinates.

    With initialization reads output of the ORCA.

    Args:
        orca_output (str or list of str): either name of the output file or list of lines read from that file.
    Attributes:
        alat (float): The lattice parameter in angstrom.
        cell (ndarray with shape (3, 3)):
            cell is 3x3 matrix with entries:

            .. math::

                [&[a_x\ b_x\ c_x]\\
                &[a_y\ b_y\ c_y]\\
                &[a_z\ b_z\ c_z]]

            where a, b, c are crystallographic vectors,
            and x, y, z are their coordinates in the cartesian reference frame.
        coordinates (ndarray with shape (n, 3)): array with the coordinates of atoms in the cell.
        names (ndarray with shape (n,)): array with the names of atoms in the cell.
        cell_units (str): Units of cell coordinates: 'bohr', 'angstrom', 'alat'.
        coordinates_units (str): Units of atom coordinates: 'crystal', 'bohr', 'angstrom', 'alat'.
    """

    def __init__(self, orca_output):
        super().__init__()
        self.cell = np.eye(3)
        self.cell_units = 'angstrom'
        self.read_output(orca_output)

    def read_output(self, orca_output):
        """
        Method to read coordinates of atoms from ORCA output into the ORCACoordinates instance.

        Args:
            orca_output (str or list of str): either name of the output file or list of lines read from that file.
        """
        try:
            lines = open(orca_output).readlines()

        except TypeError:
            lines = orca_output

        index = find_first_index('CARTESIAN COORDINATES (ANGSTROEM)', lines) + 1
        names = []
        coordinates = []

        while True:
            try:
                index += 1

                row_split = lines[index].split()
                name = row_split[0]
                crow = [float(x) for x in row_split[1:]]

                names.append(name)
                coordinates.append(crow)
            except (IndexError, ValueError):
                break

        self.coordinates_units = 'angstrom'
        self.coordinates = np.asarray(coordinates)
        self.names = np.array(names, dtype='<U16')
