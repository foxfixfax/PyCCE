import numpy as np

from pycce.bath.array import BathArray
from pycce.bath.read_cube import Cube
from pycce.units import MHZ_TO_RADKHZ, HBAR, ELECTRON_GYRO


# hbar mu0 /4pi I have no idea about units, from mengs code
# UPDATE: derived, it checks out
# HBAR = 1.054571729
# External HF given in MHz, transform to kHz * rad
# MHZ_TO_RADKHZ = 2 * np.pi * 1000


def read_xyz(nspin, skiprows: int = 2, spin_types=None):
    """
    read positions of bath within r_bath
    :param nspin: ndarray or str
        either np.ndarray with dtype [('N', np.unicode_, 16), ('xyz', np.float64, (3,))], which
        contains isotope type and position or name of the file in which they are stored
    :param center: ndarray with shape (3,)
        position of the qubit spin
    :param r_bath: float
        maximum distance from the qubit spin, at which nuclei are added to the bath
    :param skiprows:
        used when nspin is filename. Number of rows to skip in the file
    :return: ndarray
        array with positions and names of the bath with dtype [('N', np.unicode_, 16), ('xyz', np.float64, (3,))]
    """

    if isinstance(nspin, BathArray):
        atoms = nspin.copy()
        if spin_types is not None:
            try:
                atoms.add_type(**spin_types)
            except TypeError:
                atoms.add_type(*spin_types)

    elif isinstance(nspin, np.ndarray):
        dataset = nspin
        atoms = BathArray(array=dataset, types=spin_types)
    else:
        dt_read = np.dtype([('N', np.unicode_, 16), ('xyz', np.float64, (3,))])
        dataset = np.loadtxt(nspin, dtype=dt_read, skiprows=skiprows)
        atoms = BathArray(array=dataset, types=spin_types)

    return atoms


def gen_hyperfine(atoms: np.ndarray, ntype: dict, center: np.ndarray = None,
                  gyro_e: float = ELECTRON_GYRO, external_atoms: np.ndarray = None,
                  error_range: float = 0.2, cube: Cube = None) -> np.ndarray:
    """
    Generate hyperfine values for array of bath

    :param atoms: ndarray with shape (natoms,)
        dtype should include [('N', np.unicode_, 16), ('xyz', np.float64, (3,))] containing the
        coordinates (xyz) of the nuclear isotope and it's type (N)
    :param ntype: dict
        contains instances of SpinType class with nuclear types, present in the bath array
    :param center: ndarray with shape (3,)
        position of central spin
    :param gyro_e: float
        gyromagnetic ratio of the qubit spin (rad/kHz/G)
    :param external_atoms: ndarray
        contains bath with predefined hyperfine values and EFG
    :param error_range: float
        error range within which the coordinates of bath in external_atoms are considered the same
        as in the atoms_inside array
    :param cube:
    :return: BathArray
        array of bath with dtype [('N', np.unicode_, 16),
                                   ('xyz', np.float64, (3,)),
                                   ('A', np.float64, (3, 3))]
        contains isotope type N, coordinates xyz, hyperfine tensor A
    """

    if center is None:
        center = [0, 0, 0]

    identity = np.eye(3)

    if type(atoms) != BathArray:
        atoms = BathArray(array=atoms)

    for d in atoms:
        pos = d['xyz'] - center
        r = np.linalg.norm(pos)

        d['A'] = -(3 * np.outer(pos, pos) - identity * r ** 2) / (r ** 5) * gyro_e * ntype[d['N']].gyro * HBAR

    if external_atoms is not None:
        # counter_ext = 0
        dist_matrix = np.linalg.norm(atoms['xyz'][:, np.newaxis, :] - external_atoms['xyz'][np.newaxis, :, :], axis=-1)
        anames = np.core.defchararray.strip(atoms['N'], '1234567890')
        same_names = anames[:, np.newaxis] == external_atoms['N'][np.newaxis, :]
        criteria = np.logical_and(dist_matrix < error_range, same_names)

        indexes, ext_indexes = np.nonzero(criteria)
        # Check for uniqueness. If several follow the criteria, use the first one appearing.
        _, uind = np.unique(indexes, return_index=True)
        indexes = indexes[uind]
        ext_indexes = ext_indexes[uind]
        atoms['xyz'][indexes] = external_atoms['xyz'][ext_indexes]

        if 'A' in external_atoms.dtype.names:
            # print('found A')
            atoms['A'][indexes] = external_atoms['A'][ext_indexes].copy()
        if 'contact' in external_atoms.dtype.names:
            # print('found contact')
            atoms['A'][indexes] += (identity[np.newaxis, :, :] *
                                    external_atoms['contact'][ext_indexes][:, np.newaxis, np.newaxis])
        if 'Q' in external_atoms.dtype.names:
            atoms['Q'][indexes] = external_atoms['Q'][ext_indexes].copy()
            # pref = ntype[n['N']].q / (6 * s * (2 * s - 1))
            # delI2 = np.sum(np.diag(n['Q'])) * np.eye(I[s].x.shape[0]) * s * (s + 1)

        newcounter = ext_indexes.size
        print('Number of bath with external HF: {}'.format(newcounter))

    if cube is not None:
        where = np.ones(atoms.shape, dtype=bool) if external_atoms is None else ~criteria
        for a in atoms[where]:
            a['A'] = cube.intergate(a['xyz'] - center, ntype[a['N']].gyro, gyro_e)

    print('Number of overall Nuclear spins is {}'.format(atoms.shape[0]))
    return atoms


def read_external(coord_f: str, hf_f: str = None, cont_f: str = None,
                  skiprows: int = 1, erbath=None,
                  center=None) -> np.ndarray:
    """
    Function to read inserts from GIPAW. Does not renormalize the data by the spin
    Exists for backwards compatibility with Meng Code. To be removed in the release version

    :param coord_f: str
        name of the file containing coord of bath with external hyperfine
    :param hf_f: str
        external dipolar-dipolar
    :param cont_f: str
        external contact terms
    :param skiprows: int
        number of rows to skip in the three files
    :param erbath: float
        maximum distance from the center to be included
    :param center: ndarray with shape (3,)
        position of the central spin
    :return: ndarray
        ndarray with bath
    """
    dt_read = np.dtype([('N', np.unicode_, 16), ('xyz', np.float64, (3,))])
    dataset = np.loadtxt(coord_f, dtype=dt_read, skiprows=skiprows)

    dt_out = np.dtype([('N', np.unicode_, 16),
                       ('xyz', np.float64, (3,)),
                       ('A', np.float64, (3, 3))])

    atoms = np.zeros(dataset.shape, dtype=dt_out)

    atoms['N'][:] = dataset['N']
    atoms['xyz'][:] = dataset['xyz']

    if hf_f:
        with open(hf_f) as hf:
            for i in range(skiprows):
                next(hf)

            for a in atoms:

                for i in range(3):
                    gl = next(hf)
                    a['A'][i] = [np.float64(x) * MHZ_TO_RADKHZ for x in gl.split()[2:]]

                next(hf)

    if cont_f:
        with open(cont_f) as contact:
            for i in range(skiprows):
                next(contact)

            for a in atoms:
                cl = next(contact)
                a['A'] += np.eye(3) * np.float64(cl.split()[-1]) * MHZ_TO_RADKHZ

    if erbath is not None and center is not None:
        atoms = atoms[np.linalg.norm(atoms['xyz'] - center, axis=-1) < erbath]

    print('{} external Hypefine tensors were found.'.format(atoms.shape[0]))
    return atoms
