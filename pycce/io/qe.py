import warnings

import numpy as np
from pycce.io.base import DFTCoordinates, fortran_value, find_first_index, set_isotopes
from pycce.bath.array import BathArray, transform
from pycce.constants import MHZ_TO_KHZ, BOHR_TO_ANGSTROM, EFG_CONVERSION

qe_coord_types = ['crystal', 'bohr', 'angstrom', 'alat']



def read_qe(pwfile, hyperfine=None, efg=None, s=1, pwtype=None, types=None, isotopes=None,
            center=None, center_type=None, rotation_matrix=None, rm_style='col', find_isotopes=True):
    r"""
    Function to read PW/GIPAW output from Quantum Espresso into BathArray.

    Changes the names of the atoms to the most abundant isotopes if ``find_isotopes`` set to True.
    If that is not the desired outcome, user can define which isotopes to use using keyword isotopes.
    If ``find_isotopes`` is False, then keep the original names even when ``isotopes`` argument is provided.

    Args:
        pwfile (str):
            Name of PW input or output file.
            If the file doesn't have proper extension, parameter pw_type should indicate the type.
        hyperfine (str): name of the GIPAW hyperfine output.
        efg (str): Name of the gipaw electric field tensor output.
        s (float): Spin of the central spin. Default 1.
        pwtype (str): Type of the ``pwfile``. if not listed, will be inferred from extension of pwfile.
        types (SpinDict or list of tuples): SpinDict containing SpinTypes of isotopes or input to make one.
        isotopes (dict): Optional.
            Dictionary with entries: {"element" : "isotope"}, where "element" is the name of the element
            in DFT output, "isotope" is the name of the isotope.
        center (ndarray of shape (3,)): Position of (0, 0, 0) point in input coordinates.
        center_type (str): Type of the coordinates provided in center argument.
            Possible value include: 'bohr', 'angstrom', 'crystal', 'alat'. Default assumes the same as in PW file.
        rotation_matrix (ndarray of shape (3,3)):
            Rotation matrix to rotate basis. For details see utilities.transform.
        rm_style (str):
            Indicates how rotation matrix should be interpreted.
            Can take values "col" or "row". Default "col"
        find_isotopes (bool): If true, sets isotopes instead of names of the atoms.

    Returns:
        BathArray:
            BathArray containing atoms with hyperfine couplings and quadrupole tensors from QE output.

    """

    couplings = None
    gradients = None

    if hyperfine is not None:
        contact, dipolar = read_hyperfine(hyperfine, spin=s)
        couplings = dipolar + np.eye(3)[np.newaxis, :, :] * contact[:, np.newaxis, np.newaxis]

    if efg is not None:
        gradients = read_efg(efg)

    pwoutput = PWCoordinates(pwfile, pwtype=pwtype)

    if center is not None and center_type is None:
        center_type = pwoutput.coordinates_units

    pwoutput.to_angstrom(inplace=True)
    coord, names = pwoutput.coordinates, pwoutput.names

    with warnings.catch_warnings(record=True) as w:
        atoms = BathArray(array=coord, names=names,
                          hyperfines=couplings,
                          types=types)

    if find_isotopes:
        set_isotopes(atoms, isotopes=isotopes, spin_types=types)

    if gradients is not None:
        atoms.from_efg(gradients)

    if center is not None:
        center = pwoutput.get_angstrom(center, center_type)

    atoms = transform(atoms, center, rotation_matrix=rotation_matrix, style=rm_style)
    return atoms


class PWCoordinates(DFTCoordinates):
    """
    Coordinates of the system from the PW data of Quantum Espresso. Subclass of the DFTCoordinates.

    With initiallization reads either output or input of PW module of QE.

    Args:
        filename (str): name of the PW input or output.
        pwfile (str):
            Name of PW input or output file.
            If the file doesn't have proper extension, parameter pw_type should indicate the type.
        pwtype (str): Type of the coord_f. if not listed, will be inferred from extension of pwfile.
        to_angstrom (bool): True if automatically convert the units of ``cell`` and ``coordinates`` to Angstrom.

    """

    def __init__(self, filename, pwtype=None, to_angstrom=False):
        super().__init__()

        if not pwtype:
            pwtype = filename.split('.')[-1]

        if pwtype == 'in':
            self.parse_input(filename, to_angstrom=to_angstrom)
        elif pwtype == 'out':
            self.parse_output(filename, to_angstrom=to_angstrom)

        else:
            raise TypeError('Unsupported pw_type! Only .out or .in are supported')

    def parse_output(self, filename, to_angstrom=False):
        """
        Method to read coordinates of atoms from PW output into the PWCoordinates instance.

        Args:
            filename (str): the name of the output file.
            to_angstrom (bool): True if automatically convert the units of ``cell`` and ``coordinates`` to Angstrom.

        Returns:
            None

        """
        lines = open(filename).readlines()
        alat_index = find_first_index('lattice parameter (alat)', lines)

        self.alat = float(lines[alat_index].split()[-2]) * BOHR_TO_ANGSTROM

        cell_index = find_first_index('crystal axes', lines) + 1
        cell = []
        for index in range(cell_index, cell_index + 3):
            row_split = lines[index].split()
            cell_row = [float(x) for x in row_split[-4:-1]]
            cell.append(cell_row)

        self.cell = np.asarray(cell).T
        self.cell_units = 'alat'

        # read coordinates

        names = []
        coordinates = []
        try:
            final_kw = 'Begin final coordinates'
            start = find_first_index(final_kw, lines)
            coord_kw = 'ATOMIC_POSITIONS'
            index = find_first_index(coord_kw, lines, start=start)
            coord_units = get_ctype(lines[index])

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

        except KeyError:
            coord_units = 'alat'

            coord_kw = 'Cartesian axes'
            index = find_first_index(coord_kw, lines) + 2
            while True:
                try:
                    index += 1

                    row_split = lines[index].split()
                    name = row_split[1]
                    crow = [float(x) for x in row_split[-4:-1]]

                    names.append(name)
                    coordinates.append(crow)

                except IndexError:
                    break

        self.coordinates_units = coord_units
        self.coordinates = np.asarray(coordinates)
        self.names = np.array(names, dtype='<U16')

        if to_angstrom:
            self.to_angstrom(inplace=True)
        return self

    def parse_input(self, filename, to_angstrom=False):
        """
        Method to read coordinates of atoms from PW input into the PWCoordinates instance.

        Args:
            filename (str): the name of the output file.
            to_angstrom (bool): True if automatically convert the units of ``cell`` and ``coordinates`` to Angstrom.

        """
        input_string = open(filename).read()
        namelists = read_qe_namelists(input_string.lower())
        lines = input_string.splitlines()

        alat = namelists['system'].get('celldm(1)', None)

        if alat is None:
            alat = namelists['system'].get('a', None)
        else:
            alat *= BOHR_TO_ANGSTROM

        cell = cell_from_system(namelists['system'])
        if cell is not None:
            cell_units = 'bohr'
        else:
            index = find_first_index('CELL_PARAMETERS', lines)
            cell_units = get_ctype(lines[index])
            self.cell_units = cell_units
            cell = []

            for _ in range(3):
                index += 1
                line = lines[index]
                cell.append([float(x) for x in line.split()])

            cell = np.asarray(cell).T

            if cell_units == 'alat' and alat is None:
                raise ValueError('alat was not found')

            if alat is None:
                assert cell_units in ['bohr', 'angstrom']
                alat = cell[0, 0] * BOHR_TO_ANGSTROM ** (cell_units == 'bohr')

        self.cell = cell
        self.cell_units = cell_units
        self.alat = alat

        index = find_first_index('ATOMIC_POSITIONS', lines)
        coord_units = get_ctype(lines[index])

        names = []
        coords = []

        for i in range(index + 1, index + 1 + namelists['system']['nat']):

            row_split = lines[i].split()
            names.append(row_split[0])
            coords.append([float(x) for x in row_split[1:]])

        self.names = np.array(names, dtype='<U16')
        self.coordinates = np.asarray(coords)
        self.coordinates_units = coord_units

        if to_angstrom:
            self.to_angstrom(inplace=True)


def cell_from_system(sdict):
    """
    Function to obtain cell from namelist SYSTEM read from PW input.

    Args:
        sdict (dict): Dictinary generated from namelist SYSTEM of PW input.

    Returns:
        ndarray with shape (3,3):
            Cell is 3x3 matrix with entries::

                [[a_x b_x c_x]
                 [a_y b_y c_y]
                 [a_z b_z c_z]],

            where a, b, c are crystallographic vectors,
            and x, y, z are their coordinates in the cartesian reference frame.

    """
    ibrav = sdict.get('ibrav', None)
    if ibrav == 0:
        return None
    params = ['a', 'b', 'c', 'cosab', 'cosac', 'cosbc']
    celldm = [sdict.get(f'celldm({i + 1})', 0) for i in range(6)]
    if not any(celldm):
        abc = [sdict.get(a, 0) for a in params]
        celldm = celldms_from_abc(ibrav, abc)

    if not any(celldm):
        return None

    if ibrav == 1:
        cell = np.eye(3) * celldm[0]
        return cell

    elif ibrav == 2:
        v1 = celldm[0] / 2 * np.array([-1, 0, 1])
        v2 = celldm[0] / 2 * np.array([0, 1, 1])
        v3 = celldm[0] / 2 * np.array([-1, 1, 0])

    elif ibrav == 3:
        v1 = celldm[0] / 2 * np.array([1, 1, 1])
        v2 = celldm[0] / 2 * np.array([-1, 1, 1])
        v3 = celldm[0] / 2 * np.array([-1, -1, 1])

    elif ibrav == -3:
        v1 = celldm[0] / 2 * np.array([-1, 1, 1])
        v2 = celldm[0] / 2 * np.array([1, -1, 1])
        v3 = celldm[0] / 2 * np.array([1, 1, -1])

    elif ibrav == 4:
        v1 = celldm[0] * np.array([1, 0, 0])
        v2 = celldm[0] * np.array([-1 / 2, np.sqrt(3) / 2, 0])
        v3 = celldm[0] * np.array([0, 0, celldm[2]])

    elif ibrav == 5:
        term_1 = np.sqrt(1 + 2 * celldm[3])
        term_2 = np.sqrt(1 - celldm[3])
        v1 = celldm[0] * np.array([term_2 / np.sqrt(2), -term_2 / np.sqrt(6), term_1 / np.sqrt(3)])
        v2 = celldm[0] * np.array([0, term_2 * np.sqrt(2 / 3), term_1 / np.sqrt(3)])
        v3 = celldm[0] * np.array([-term_2 / np.sqrt(2), -term_2 / np.sqrt(6), term_1 / np.sqrt(3)])

    elif ibrav == -5:
        term_1 = np.sqrt(1 + 2 * celldm[3])
        term_2 = np.sqrt(1 - celldm[3])
        v1 = celldm[0] * np.array([(term_1 - 2 * term_2) / 3, (term_1 + term_2) / 3, (term_1 + term_2) / 3])
        v2 = celldm[0] * np.array([(term_1 + term_2) / 3, (term_1 - 2 * term_2) / 3, (term_1 + term_2) / 3])
        v3 = celldm[0] * np.array([(term_1 + term_2) / 3, (term_1 + term_2) / 3, (term_1 - 2 * term_2) / 3])

    elif ibrav == 6:
        v1 = celldm[0] * np.array([1, 0, 0])
        v2 = celldm[0] * np.array([0, 1, 0])
        v3 = celldm[0] * np.array([0, 0, celldm[2]])
    elif ibrav == 7:
        v1 = celldm[0] / 2 * np.array([1, -1, celldm[2]])
        v2 = celldm[0] / 2 * np.array([1, 1, celldm[2]])
        v3 = celldm[0] / 2 * np.array([-1, -1, celldm[2]])
    elif ibrav == 8:
        v1 = celldm[0] * np.array([1, 0, 0])
        v2 = celldm[0] * np.array([0, celldm[1], 0])
        v3 = celldm[0] * np.array([0, 0, celldm[2]])
    elif ibrav == 9:
        v1 = celldm[0] / 2 * np.array([1, celldm[1], 0])
        v2 = celldm[0] / 2 * np.array([-1, celldm[1], 0])
        v3 = celldm[0] * np.array([0, 0, celldm[2]])

    elif ibrav == -9:
        v1 = celldm[0] / 2 * np.array([1, -celldm[1], 0])
        v2 = celldm[0] / 2 * np.array([+1, celldm[1], 0])
        v3 = celldm[0] * np.array([0, 0, celldm[2]])

    elif ibrav == 91:
        v1 = celldm[0] * np.array([1, 0, 0])
        v2 = celldm[0] / 2 * np.array([0, celldm[1], -celldm[2]])
        v3 = celldm[0] / 2 * np.array([0, celldm[1], celldm[2]])
    elif ibrav == 10:
        v1 = celldm[0] / 2 * np.array([1, 0, celldm[2]])
        v2 = celldm[0] / 2 * np.array([1, celldm[1], 0])
        v3 = celldm[0] / 2 * np.array([0, celldm[1], celldm[2]])
    elif ibrav == 11:
        v1 = celldm[0] / 2 * np.array([1, celldm[1], celldm[2]])
        v2 = celldm[0] / 2 * np.array([-1, celldm[1], celldm[2]])
        v3 = celldm[0] / 2 * np.array([-1, -celldm[1], celldm[2]])
    elif ibrav == 12:
        sen = np.sqrt(1 - celldm[3] ** 2)
        v1 = celldm[0] * np.array([1, 0, 0])
        v2 = celldm[0] * np.array([celldm[1] * celldm[3], celldm[1] * sen, 0])
        v3 = celldm[0] * np.array([0, 0, celldm[2]])
    elif ibrav == -12:
        sen = np.sqrt(1 - celldm[4] ** 2)
        v1 = celldm[0] * np.array([1, 0, 0])
        v2 = celldm[0] * np.array([0, celldm[1], 0])
        v3 = celldm[0] * np.array([celldm[2] * celldm[4], 0, celldm[2] * sen])

    elif ibrav == 13:
        sen = np.sqrt(1 - celldm[3] ** 2)
        v1 = celldm[0] / 2 * np.array([1, 0, -celldm[2]])
        v2 = celldm[0] * np.array([celldm[1] * celldm[3], celldm[1] * sen, 0])
        v3 = celldm[0] / 2 * np.array([1, 0, celldm[2]])

    elif ibrav == -13:
        sen = np.sqrt(1 - celldm[4] ** 2)
        v1 = celldm[0] / 2 * np.array([1, celldm[1], 0])
        v2 = celldm[0] / 2 * np.array([-1, celldm[1], 0])
        v3 = celldm[0] * np.array([celldm[2] * celldm[4], 0, celldm[2] * sen])

    elif ibrav == 14:
        singam = np.sqrt(1 - celldm[5] ** 2)
        term = (1 + 2 * celldm[3] * celldm[4] * celldm[5] - celldm[3] ** 2 - celldm[4] ** 2 - celldm[5] ** 2)
        term = np.sqrt(term / (1 - celldm[5] ** 2))

        v1 = celldm[0] * np.array([1,
                                   0,
                                   0])
        v2 = celldm[0] * np.array([celldm[1] * celldm[5],
                                   celldm[1] * singam,
                                   0])
        v3 = celldm[0] * np.array([celldm[2] * celldm[4],
                                   celldm[2] * (celldm[3] - celldm[4] * celldm[5]) / singam,
                                   celldm[2] * term])
    else:
        raise ValueError('Unsupported ibrav')

    cell = np.stack([v1, v2, v3], axis=1)

    return cell


def celldms_from_abc(ibrav, abc_list):
    """
    Obtain celldms from ibrav value and a, b, c, cosab, cosac, cosbc parameters.

    Using ibrav value and abc parameters from PW input generate celldm array, necessary to construct cell parameters.
    For details about abc and ibrav values see PW input documentation.

    Args:
        ibrav (int): ibrav parameter of PW input.
        abc_list (list): List, of 6 parameters:  a, b, c, cosab, cosac, cosbc

    Returns:
        celldm (list): list of 6 values, from which cell can be generated.

    """
    a, b, c, cosab, cosac, cosbc = abc_list

    celldm = [0.] * 6
    celldm[0] = a / BOHR_TO_ANGSTROM
    celldm[1] = b / a
    celldm[2] = c / a

    if ibrav in [0, 14]:

        celldm[3] = cosbc
        celldm[4] = cosac
        celldm[5] = cosab

    elif ibrav in [-12, -13]:
        celldm[3] = 0.0
        celldm[4] = cosac
        celldm[5] = 0.0

    elif ibrav in [-5, 5, 12, 13]:
        celldm[3] = cosab
        celldm[4] = 0.0
        celldm[5] = 0.0

    return celldm

def read_gipaw_tensors(lines, keyword=None, start=None, conversion=1):
    """
    Helper function to read GIPAW tensors from the list of lines.

    Args:
        lines (list of str): List of strings contraining lines from the file. Output of open(file).readlines().
        keyword (str): Keyword in the line which indicates the beginning of the tensor data block.
        start (int): Index of the line which indicates the beginning of the tensor data block.
        conversion (float): Conversion factor from GIPAW units to the ones, used in this package.

    Returns:
        ndarray with shape (n, 3, 3): Array of tensors.

    """
    if keyword is not None:
        start = find_first_index(keyword, lines)

    if start is None:
        raise ValueError

    all_tensors = []

    n = start
    while True:
        n += 1
        tensor = []
        try:
            for _ in range(3):
                line = lines[n]
                tensor.append([float(x) * conversion for x in line.split()[2:]])
                n += 1

        except (ValueError, IndexError):
            break

        all_tensors.append(tensor)

    return all_tensors


def read_hyperfine(filename, spin=1):
    """
    Function to read hyperfine couplings from GIPAW output.

    Args:
        filename (str): Name of the GIPAW hyperfine output.
        spin (float): Spin of the central spin. Default 1.

    Returns:
        tuple: Tuple containing:

            * *ndarray with shape (n,)*: Array of Fermi contact terms.
            * *ndarray with shape (n, 3,3)*: Array of spin dipolar hyperfine tensors.

    """
    conversion = MHZ_TO_KHZ / (2 * spin)

    lines = open(filename).readlines()

    dipol_keyword = 'total dipolar (symmetrized)'
    contact_keyword = 'Fermi contact in MHz'

    dipolars = read_gipaw_tensors(lines, dipol_keyword, conversion=conversion)
    start = find_first_index(contact_keyword, lines) + 2

    contacts = []

    for index in range(start, start + len(dipolars)):
        line = lines[index]
        # divided by spin, b/c NI in the GIPAW code
        cont = float(line.split()[-1]) * conversion
        contacts.append(cont)

    return np.asarray(contacts), np.asarray(dipolars)


def read_efg(filename):
    """
    Function to read electric field gradient tensors from GIPAW output.

    Args:
        filename (str): Name of the GIPAW EFG-containing output.

    Returns:
        ndarray with shape (n, 3,3): Array of EFG tensors.

    """
    efg_kw = 'total EFG (symmetrized)'
    lines = open(filename).readlines()
    tensors = read_gipaw_tensors(lines, keyword=efg_kw, conversion=EFG_CONVERSION)

    return np.asarray(tensors)


def read_qe_namelists(input_string):
    """
    Read Fortran-like namelists from the large string.

    Args:
        input_string (str): String representation of the QE input file.

    Returns:
        dict: Dictionary, containing dicts for each namelist found in the input string.

    """
    namelists = {}

    for block in input_string.split('/\n'):

        if not '&' in block:
            # means we are out of namelists
            break

        lines = [s.strip() for s in block.splitlines() if (s.strip() and s[0] != '!')]
        index = find_first_index('&', lines)
        block_name = lines[index].strip('&\t ')

        namelists[block_name] = {}
        for row in lines[index + 1:]:
            for pair in row.split(','):
                if not pair.strip():
                    continue
                name, _, value = (x.strip(',\t ').lower() for x in pair.partition("="))
                namelists[block_name][name] = fortran_value(value)

    return namelists


def get_ctype(lin):
    """
    Get coordinates type from the line of QE input/output.

    Args:
        str: Line from QE input/output containing string with coordinates type.

    Returns:
        str: type of the coordinates.
    """

    try:
        coord_type = next(filter(lambda x: x in lin.lower(), qe_coord_types))
    except IndexError:
        raise ValueError(f'{lin} type is not supported.\nAllowed types: ', ' '.join(*qe_coord_types))

    return coord_type

# ELSEIF (ibrav == 7) THEN
#    !
#    !     body centered tetragonal lattice
#    !
#    IF (celldm (3) <= 0.d0) CALL errore ('latgen', 'wrong celldm(3)', ibrav)
#    !
#    cbya=celldm(3)
#    a2(1)=celldm(1)/2.d0
#    a2(2)=a2(1)
#    a2(3)=cbya*celldm(1)/2.d0
#    a1(1)= a2(1)
#    a1(2)=-a2(1)
#    a1(3)= a2(3)
#    a3(1)=-a2(1)
#    a3(2)=-a2(1)
#    a3(3)= a2(3)
#    !
# ELSEIF (ibrav == 8) THEN
#    !
#    !     Simple orthorhombic lattice
#    !
#    IF (celldm (2) <= 0.d0) CALL errore ('latgen', 'wrong celldm(2)', ibrav)
#    IF (celldm (3) <= 0.d0) CALL errore ('latgen', 'wrong celldm(3)', ibrav)
#    !
#    a1(1)=celldm(1)
#    a2(2)=celldm(1)*celldm(2)
#    a3(3)=celldm(1)*celldm(3)
#    !
# ELSEIF ( abs(ibrav) == 9) THEN
#    !
#    !     One face (base) centered orthorhombic lattice  (C type)
#    !
#    IF (celldm (2) <= 0.d0) CALL errore ('latgen', 'wrong celldm(2)', &
#                                                                abs(ibrav))
#    IF (celldm (3) <= 0.d0) CALL errore ('latgen', 'wrong celldm(3)', &
#                                                                abs(ibrav))
#    !
#    IF ( ibrav == 9 ) THEN
#       !   old PWscf description
#       a1(1) = 0.5d0 * celldm(1)
#       a1(2) = a1(1) * celldm(2)
#       a2(1) = - a1(1)
#       a2(2) = a1(2)
#    ELSE
#       !   alternate description
#       a1(1) = 0.5d0 * celldm(1)
#       a1(2) =-a1(1) * celldm(2)
#       a2(1) = a1(1)
#       a2(2) =-a1(2)
#    ENDIF
#    a3(3) = celldm(1) * celldm(3)
#    !
# ELSEIF ( ibrav == 91 ) THEN
#    !
#    !     One face (base) centered orthorhombic lattice  (A type)
#    !
#    IF (celldm (2) <= 0.d0) CALL errore ('latgen', 'wrong celldm(2)', ibrav)
#    IF (celldm (3) <= 0.d0) CALL errore ('latgen', 'wrong celldm(3)', ibrav)
#    !
#    a1(1) = celldm(1)
#    a2(2) = celldm(1) * celldm(2) * 0.5_DP
#    a2(3) = - celldm(1) * celldm(3) * 0.5_DP
#    a3(2) = a2(2)
#    a3(3) = - a2(3)
#    !
# ELSEIF (ibrav == 10) THEN
#    !
#    !     All face centered orthorhombic lattice
#    !
#    IF (celldm (2) <= 0.d0) CALL errore ('latgen', 'wrong celldm(2)', ibrav)
#    IF (celldm (3) <= 0.d0) CALL errore ('latgen', 'wrong celldm(3)', ibrav)
#    !
#    a2(1) = 0.5d0 * celldm(1)
#    a2(2) = a2(1) * celldm(2)
#    a1(1) = a2(1)
#    a1(3) = a2(1) * celldm(3)
#    a3(2) = a2(1) * celldm(2)
#    a3(3) = a1(3)
#    !
# ELSEIF (ibrav == 11) THEN
#    !
#    !     Body centered orthorhombic lattice
#    !
#    IF (celldm (2) <= 0.d0) CALL errore ('latgen', 'wrong celldm(2)', ibrav)
#    IF (celldm (3) <= 0.d0) CALL errore ('latgen', 'wrong celldm(3)', ibrav)
#    !
#    a1(1) = 0.5d0 * celldm(1)
#    a1(2) = a1(1) * celldm(2)
#    a1(3) = a1(1) * celldm(3)
#    a2(1) = - a1(1)
#    a2(2) = a1(2)
#    a2(3) = a1(3)
#    a3(1) = - a1(1)
#    a3(2) = - a1(2)
#    a3(3) = a1(3)
#    !
# ELSEIF (ibrav == 12) THEN
#    !
#    !     Simple monoclinic lattice, unique (i.e. orthogonal to a) axis: c
#    !
#    IF (celldm (2) <= 0.d0) CALL errore ('latgen', 'wrong celldm(2)', ibrav)
#    IF (celldm (3) <= 0.d0) CALL errore ('latgen', 'wrong celldm(3)', ibrav)
#    IF (abs(celldm(4))>=1.d0) CALL errore ('latgen', 'wrong celldm(4)', ibrav)
#    !
#    sen=sqrt(1.d0-celldm(4)**2)
#    a1(1)=celldm(1)
#    a2(1)=celldm(1)*celldm(2)*celldm(4)
#    a2(2)=celldm(1)*celldm(2)*sen
#    a3(3)=celldm(1)*celldm(3)
#    !
# ELSEIF (ibrav ==-12) THEN
#    !
#    !     Simple monoclinic lattice, unique axis: b (more common)
#    !
#    IF (celldm (2) <= 0.d0) CALL errore ('latgen', 'wrong celldm(2)',-ibrav)
#    IF (celldm (3) <= 0.d0) CALL errore ('latgen', 'wrong celldm(3)',-ibrav)
#    IF (abs(celldm(5))>=1.d0) CALL errore ('latgen', 'wrong celldm(5)',-ibrav)
#    !
#    sen=sqrt(1.d0-celldm(5)**2)
#    a1(1)=celldm(1)
#    a2(2)=celldm(1)*celldm(2)
#    a3(1)=celldm(1)*celldm(3)*celldm(5)
#    a3(3)=celldm(1)*celldm(3)*sen
#    !
# ELSEIF (ibrav == 13) THEN
#    !
#    !     One face centered monoclinic lattice unique axis c
#    !
#    IF (celldm (2) <= 0.d0) CALL errore ('latgen', 'wrong celldm(2)', ibrav)
#    IF (celldm (3) <= 0.d0) CALL errore ('latgen', 'wrong celldm(3)', ibrav)
#    IF (abs(celldm(4))>=1.d0) CALL errore ('latgen', 'wrong celldm(4)', ibrav)
#    !
#    sen = sqrt( 1.d0 - celldm(4) ** 2 )
#    a1(1) = 0.5d0 * celldm(1)
#    a1(3) =-a1(1) * celldm(3)
#    a2(1) = celldm(1) * celldm(2) * celldm(4)
#    a2(2) = celldm(1) * celldm(2) * sen
#    a3(1) = a1(1)
#    a3(3) =-a1(3)
# ELSEIF (ibrav == -13) THEN
#    !
#    !     One face centered monoclinic lattice unique axis b
#    !
#    IF (celldm (2) <= 0.d0) CALL errore ('latgen', 'wrong celldm(2)',-ibrav)
#    IF (celldm (3) <= 0.d0) CALL errore ('latgen', 'wrong celldm(3)',-ibrav)
#    IF (abs(celldm(5))>=1.d0) CALL errore ('latgen', 'wrong celldm(5)',-ibrav)
#    !
#    sen = sqrt( 1.d0 - celldm(5) ** 2 )
#    a1(1) = 0.5d0 * celldm(1)
#    a1(2) =-a1(1) * celldm(2)
#    a2(1) = a1(1)
#    a2(2) =-a1(2)
#    a3(1) = celldm(1) * celldm(3) * celldm(5)
#    a3(3) = celldm(1) * celldm(3) * sen
#    !
# ELSEIF (ibrav == 14) THEN
#    !
#    !     Triclinic lattice
#    !
#    IF (celldm (2) <= 0.d0) CALL errore ('latgen', 'wrong celldm(2)', ibrav)
#    IF (celldm (3) <= 0.d0) CALL errore ('latgen', 'wrong celldm(3)', ibrav)
#    IF (abs(celldm(4))>=1.d0) CALL errore ('latgen', 'wrong celldm(4)', ibrav)
#    IF (abs(celldm(5))>=1.d0) CALL errore ('latgen', 'wrong celldm(5)', ibrav)
#    IF (abs(celldm(6))>=1.d0) CALL errore ('latgen', 'wrong celldm(6)', ibrav)
#    !
#    singam=sqrt(1.d0-celldm(6)**2)
#    term= (1.d0+2.d0*celldm(4)*celldm(5)*celldm(6)             &
#         -celldm(4)**2-celldm(5)**2-celldm(6)**2)
#    IF (term < 0.d0) CALL errore &
#       ('latgen', 'celldm do not make sense, check your data', ibrav)
#    term= sqrt(term/(1.d0-celldm(6)**2))
#    a1(1)=celldm(1)
#    a2(1)=celldm(1)*celldm(2)*celldm(6)
#    a2(2)=celldm(1)*celldm(2)*singam
#    a3(1)=celldm(1)*celldm(3)*celldm(5)
#    a3(2)=celldm(1)*celldm(3)*(celldm(4)-celldm(5)*celldm(6))/singam
#    a3(3)=celldm(1)*celldm(3)*term
