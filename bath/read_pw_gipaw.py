import numpy as np
import warnings

FLOAT_ERROR_RANGE = 1e-6

def find_line(file, keyw):
    """find line in file with keyw in it"""
    for lin in file:
        if keyw in lin:
            break
    else:
        raise RuntimeError('The {} was not found in {}'.format(keyw, file))

    return lin


def read_qe(coord_f, hf_f, s=1, pw_type=None):
    """from quantum espresso output of pw and gipaw read atoms
       into np.ndarray of dtype [('N', np.unicode_, 16), ('xyz', np.float64, (3,)), ('contact', np.float64), \
       ('A', np.float64, (3, 3))])
       :param coord_f: pw input or output file. if doesn't have proper extension, pw_type should indicate the type
       :param hf_f: gipaw output
       :param s: spin of the qubit spin
       :param pw_type: type of the coord_f. if not listed, will be inferred from extension of coord_f
       :return: np.ndarray containing types of nuclei, their position, contact HF term and dipolar-dipolar term"""

    dt_out = np.dtype([('N', np.unicode_, 16),
                       ('xyz', np.float64, (3,)),
                       ('contact', np.float64),
                       ('A', np.float64, (3, 3))])

    types = []
    coordinates = []
    dipolars = []

    if not pw_type:
        pw_type = coord_f.split('.')[-1]

    if pw_type not in ('out', 'in'):
        raise TypeError('Unsupported pw_type!')

    with open(hf_f) as hf, open(coord_f) as coord:

        if pw_type == 'out':
            fcoord = 'Begin final coordinates'
            find_line(coord, fcoord)

        pw_keyword = 'ATOMIC_POSITIONS'

        gipaw_keyword = 'total dipolar (symmetrized)'

        find_line(hf, gipaw_keyword)

        find_line(coord, pw_keyword)

        for pl in coord:
            # Important only for pw out
            if 'end final coordinates' in pl.lower():
                break

            typ = pl.split()[0]
            try:
                coord = np.array([float(x) for x in pl.split()[1:]])
            except ValueError:
                break
            A = []
            # divided by spin, b/c fucked in the GIPAW code
            for i in range(3):
                gl = next(hf)
                A.append([float(x) / (2 * s) for x in gl.split()[2:]])

            types.append(typ)
            dipolars.append(A)
            coordinates.append(coord)

            next(hf)

        atoms = np.array([*zip(types,
                               coordinates,
                               [0] * len(types),
                               dipolars)],
                         dtype=dt_out)

        find_line(hf, 'Fermi contact in MHz')
        next(hf)

        for a in atoms:
            gl = next(hf)
            # divided by spin, b/c fucked in the GIPAW code
            cont = float(gl.split()[-1]) / (2 * s)
            a['contact'] = cont

        print("finished reading")

    return atoms


def transform(atoms, center=None, cell=None, rotate=None, style='col', inplace=True):
    """
    :param atoms: array of nuclei for which the transformation will be applied. Coordinates should be stored in cell
    coordinates,
    :param center: position of center in cell coordinates
    :param cell: cell vectors in cartesian coordinates
    :param rotate: rotation matrix, which rotates the coordinate system, in which cell vectors are stored.
    Note, that rotate is applied after transition from cell coordinates to the cartesian coordinates, in which cell
    vectors are stored in matrix cell
    :param style: can have two values: 'col' or 'row'. Shows how cell and rotate matrices are stored:
    if 'col', each column of the matrix is a vector in previous coordinates, if 'row' - each row is a new vector
    :param copy:
    :return:
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

    if rotate is None:
        rotate = np.eye(3)

    if not np.all(cell - np.diag(np.diag(cell)) < FLOAT_ERROR_RANGE) and 'A' in atoms.dtype.names:
        mes = ('Changes to A tensor are supported only when cell is diagonal matrix.',
               ' Otherwise expect the unexpected')
        warnings.warn(mes, RuntimeWarning)

    atoms['xyz'] -= np.asarray(center)

    if style.lower() == 'row':
        cell = cell.T
        rotate = rotate.T

    atoms['xyz'] = np.einsum('jk,ik->ij', cell, atoms['xyz'])
    atoms['xyz'] = np.einsum('jk,ik->ij', np.linalg.inv(rotate), atoms['xyz'])

    if 'A' in atoms.dtype.names:
        atoms['A'] = np.matmul(atoms['A'], rotate)
        atoms['A'] = np.matmul(np.linalg.inv(rotate), atoms['A'])

    return atoms
