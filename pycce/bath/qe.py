import numpy as np
import warnings

from ..units import MHZ_TO_RADKHZ, HARTREE_TO_MHZ, M_TO_BOHR, BOHR_TO_ANGSTROM
from .array import BathArray
FLOAT_ERROR_RANGE = 1e-6
BARN_TO_BOHR2 = M_TO_BOHR ** 2 * 1E-28
EFG_CONVERSION = BARN_TO_BOHR2 * HARTREE_TO_MHZ * MHZ_TO_RADKHZ  # units to convert EFG


# units of Hyperfine are kHz * rad
# units of EFGs are kHz * rad / barn
# necessary units of Q are barn


# Qs = { # isotope | Q in Q/barn
#     "N": ("14", 0.02044),
#     "C": ("11", 0.03327),
# }
# QeVzz = Qs['N'][1] * Vzz * EFG_CONVERSION
# V in units of rad * kHz / barn

def find_line(file, keyw):
    """find line in file with keyw in it"""
    for lin in file:
        if keyw.lower() in lin.lower():
            break
    else:
        raise RuntimeError('The {} was not found in {}'.format(keyw, file))

    return lin


def read_qe(coord_f, hyperfine=None, efg=None, s=1, pw_type=None, spin_types=None, which_isotopes=None,
            center=None, rotation_matrix=None, rm_style='col'):
    """
    @param coord_f: str
         pw input or output file. if doesn't have proper extension, parameter pw_type should indicate the type
    @param hyperfine: str
        name of the gipaw hyperfine output
    @param efg: str
        name of the gipaw electric field tensor output
    @param s: int
        spin of the central spin. Default 1
    @param pw_type: str
        type of the coord_f. if not listed, will be inferred from extension of coord_f
    @param spin_types: SpinDict
        optional. SpinDict containing SpinTypes of isotopes
    @param which_isotopes: dict
        optional if no efg. dictionary with entries: "element" : "isotope", where "element" is the name of the element
        in PW/GIPAW output, "isotope" is the name of the isotope
    @param center: ndarray of shape (3,)
        position of the (0,0,0) in final coordinates
    @param rotation_matrix: ndarray of shape (3,3)
        rotation matrix to rotate coordinates. For details see bath.transform
    @param rm_style: str
        indicates how rotation matrix should be interpreted. Can take values "col" or "row". Default "col"
    @return: BathArray
    """
    qe_coord_types = ['crystal', 'bohr', 'angstrom', 'alat']

    dipolars = None
    gradients = None

    if hyperfine is not None:
        dipolars = []

    if efg is not None:
        gradients = []

    spin_names = []
    coordinates = []

    if not pw_type:
        pw_type = coord_f.split('.')[-1]

    if pw_type not in ('out', 'in'):
        raise TypeError('Unsupported pw_type! Only .out or .in are supported')

    with open(coord_f) as coord:
        if hyperfine is not None:
            hf = open(hyperfine)
            gipaw_keyword = 'total dipolar (symmetrized)'
            find_line(hf, gipaw_keyword)

        if efg is not None:
            efg = open(efg)
            efg_kw = 'total EFG (symmetrized)'
            find_line(efg, efg_kw)

        if pw_type == 'out':
            fcoord = 'Begin final coordinates'
            find_line(coord, fcoord)

        pw_keyword = 'ATOMIC_POSITIONS'
        lin = find_line(coord, pw_keyword)
        try:
            coord_type = next(filter(lambda x: x in lin.lower(), qe_coord_types))
        except IndexError:
            raise ValueError('ATOMIC_POSITIONS type is not supported.\nAllowed types: ', ' '.join(*qe_coord_types))

        for pl in coord:
            # Important only for pw out
            if 'end final coordinates' in pl.lower():
                break
            if not pl.strip():
                break
            typ = pl.split()[0]
            try:
                coord = np.array([float(x) for x in pl.split()[1:]])
            except ValueError:
                break

            if hyperfine is not None:
                A = []
                # divided by spin, b/c NI in the GIPAW code
                for _ in range(3):
                    gl = next(hf)
                    A.append([float(x) / (2 * s) * MHZ_TO_RADKHZ for x in gl.split()[2:]])
                dipolars.append(A)
                next(hf)

            if efg is not None:
                v = []
                for _ in range(3):
                    fl = next(efg)
                    v.append([float(x) * EFG_CONVERSION for x in fl.split()[2:]])
                gradients.append(v)
                next(efg)

            spin_names.append(typ)
            coordinates.append(coord)

    spin_names = np.asarray(spin_names, dtype='<U16')
    if which_isotopes is not None:
        for gipaw_name in which_isotopes:
            spin_names[spin_names == gipaw_name] = which_isotopes[gipaw_name]

    atoms = BathArray(array=coordinates, spin_names=spin_names,
                      hyperfines=dipolars, quadrupoles=gradients,
                      types=spin_types)
    if efg is not None:
        pref = atoms.types[atoms].q / (2 * s * (2 * s - 1))
        atoms['Q'] *= pref[:, np.newaxis, np.newaxis]

    cell = None
    if coord_type == 'bohr':
        atoms['xyz'] *= BOHR_TO_ANGSTROM

    elif coord_type == 'crystal':
        coord = open(coord_f).readlines()
        cell = []

        if pw_type == 'in':
            lin = find_line(coord, 'CELL_PARAMETERS')
            cell_type = next(filter(lambda x: x in lin.lower(), qe_coord_types))
            cindex = coord.index(lin)
            for _ in range(cindex + 1, cindex + 4):
                lin = coord[_]
                cell.append([float(x) for x in lin.split()])

            cell = np.asarray(cell)

            if cell_type == 'bohr':
                cell *= BOHR_TO_ANGSTROM
            elif cell_type == 'alat':
                lin = find_line(coord, 'celldm(1)')
                alat = float(lin.split()[-1].strip())
                cell *= alat
            elif cell_type != 'angstrom':
                warnings.warn('CELL_PARAMETERS units are not recognized. Assumed angstrom')
        if pw_type == 'out':

            lin = find_line(coord, 'lattice parameter (alat)')
            alat = float(lin.split()[-2])
            lin = find_line(coord, 'crystal axes')

            for _ in range(3):
                lin = next(coord)
                cell.append([float(x) for x in lin.split()[3:-1]])

            cell = np.asarray(cell) * alat * BOHR_TO_ANGSTROM

    if hyperfine is not None:
        find_line(hf, 'Fermi contact in MHz')
        next(hf)

        for a in atoms:
            gl = next(hf)
            # divided by spin, b/c NI in the GIPAW code
            cont = float(gl.split()[-1]) / (2 * s)
            a['A'] += np.eye(3) * cont * MHZ_TO_RADKHZ
        hf.close()

    if rm_style == 'col' and cell is not None:
        cell = cell.T

    atoms = transform(atoms, center, cell, rotation_matrix, rm_style)
    return atoms


def transform(atoms, center=None, cell=None, rotation_matrix=None, style='col', inplace=True):
    """
    Coordinate transformation of BathArray
    @param atoms: BathArray
        bath to be rotated
    @param center: ndarray
        position of (0,0,0) in new coordinates
    @param cell: ndarray of shape (3,3)
        cell vectors in cartesian coordinates
    @param rotation_matrix: ndarray of shape (3,3)
         rotation matrix, which rotates the coordinate system, in which cell vectors are stored.
         Note, that rotaton is applied after transition from cell coordinates to the cartesian coordinates,
         in which cell vectors are stored
    @param style: str
        can have two values: 'col' or 'row'. Shows how cell and rotate matrices are stored:
        if 'col', each column of the matrix is a vector in previous coordinates, if 'row' - each row is a new vector
        default 'col'
    @param inplace:  bool
        whether to make changes to existing BathArray or create a new one. Default True
    @return:
    """

    styles = ['col', 'row']
    if not style in styles:
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

    if not np.all(cell - np.diag(np.diag(cell)) < FLOAT_ERROR_RANGE) and 'A' in atoms.dtype.names:
        mes = ('Changes to A tensor are supported only when cell is diagonal matrix.',
               ' Otherwise expect the unexpected')
        warnings.warn(mes, RuntimeWarning)

    atoms['xyz'] -= np.asarray(center)

    if style.lower() == 'row':
        cell = cell.T
        rotation_matrix = rotation_matrix.T

    atoms['xyz'] = np.einsum('jk,ik->ij', cell, atoms['xyz'])
    atoms['xyz'] = np.einsum('jk,ik->ij', np.linalg.inv(rotation_matrix), atoms['xyz'])

    if 'A' in atoms.dtype.names:
        atoms['A'] = np.matmul(atoms['A'], rotation_matrix)
        atoms['A'] = np.matmul(np.linalg.inv(rotation_matrix), atoms['A'])

    if 'Q' in atoms.dtype.names:
        atoms['Q'] = np.matmul(atoms['Q'], rotation_matrix)
        atoms['Q'] = np.matmul(np.linalg.inv(rotation_matrix), atoms['Q'])

    return atoms
