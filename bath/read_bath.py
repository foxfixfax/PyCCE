import numpy as np

from ..units import MHZ_TO_RADKHZ, HBAR


# hbar mu0 /4pi I have no idea about units, from mengs code
# UPDATE: derived, it checks out
# HBAR = 1.054571729
# External HF given in MHz, transform to kHz * rad
# MHZ_TO_RADKHZ = 2 * np.pi * 1000


def read_pos(nspin, r_bath: float, center: np.array = None, skiprows: int = 1):
    """
    read positions of atoms within r_bath
    @param nspin: ndarray or str
        either np.ndarray with dtype [('N', np.unicode_, 16), ('xyz', np.float64, (3,))], which
        contains isotope type and position or name of the file in which they are stored
    @param center: ndarray with shape (3,)
        position of the qubit spin
    @param r_bath: float
        maximum distance from the qubit spin, at which nuclei are added to the bath
    @param skiprows:
        used when nspin is filename. Number of rows to skip in the file
    @return: ndarray
        array with positions and names of the atoms with dtype [('N', np.unicode_, 16), ('xyz', np.float64, (3,))]
    """

    if center is None:
        center = [0, 0, 0]
    if isinstance(nspin, np.ndarray):
        dataset = nspin
    else:
        dt_read = np.dtype([('N', np.unicode_, 16), ('xyz', np.float64, (3,))])
        dataset = np.loadtxt(nspin, dtype=dt_read, skiprows=skiprows)

    mask = np.linalg.norm(dataset['xyz'] - np.asarray(center), axis=-1) < r_bath

    atoms_inside = dataset[mask]

    return atoms_inside


def gen_hyperfine(atoms_inside: np.ndarray, ntype: dict, center: np.ndarray = None,
                  gyro_e: float = -17608.597050, external_atoms: np.ndarray = None,
                  error_range: float = 0.2) -> np.ndarray:
    """
    Generate hyperfine values for array of atoms

    @param atoms_inside: ndarray with shape (natoms,)
        dtype should include [('N', np.unicode_, 16), ('xyz', np.float64, (3,))] containing the
        coordinates (xyz) of the nuclear isotope and it's type (N)
    @param ntype: dict
        contains instances of SpinType class with nuclear types, present in the atoms array
    @param center: ndarray with shape (3,)
        position of central spin
    @param gyro_e: float
        gyromagnetic ratio of the qubit spin (rad/kHz/G)
    @param external_atoms: ndarray
        contains atoms with predefined hyperfine values and EFG
    @param error_range: float
        error range within which the coordinates of atoms in external_atoms are considered the same
        as in the atoms_inside array
    @return: ndarray
        array of atoms with dtype [('N', np.unicode_, 16),
                                   ('xyz', np.float64, (3,)),
                                   ('A', np.float64, (3, 3))]
        contains isotope type N, coordinates xyz, hyperfine tensor A
    """

    if center is None:
        center = [0, 0, 0]

    identity = np.eye(3)

    dt_out = np.dtype([('N', np.unicode_, 16),
                       ('xyz', np.float64, (3,)),
                       ('A', np.float64, (3, 3)),
                       ('V', np.float64, (3, 3))])

    atoms = np.zeros(atoms_inside.shape[0], dtype=dt_out)

    atoms['N'] = atoms_inside['N']
    atoms['xyz'] = atoms_inside['xyz']

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
        if 'V' in external_atoms.dtype.names:
            atoms['V'][indexes] = external_atoms['V'][ext_indexes].copy()

        newcounter = ext_indexes.size
        print('Number of atoms with external HF: {}'.format(newcounter))

    print('Number of overall Nuclear spins is {}'.format(atoms.shape[0]))
    return atoms


def read_external(coord_f: str, hf_f: str = None, cont_f: str = None, skiprows: int = 1, erbath=None,
                  center=None) -> np.ndarray:
    """
    Function to read inserts from GIPAW. Does not renormalize the data by the spin
    Exists for backwards compatibility with Meng Code. To be removed in the release version

    @param coord_f: str
        name of the file containing coord of atoms with external hyperfine
    @param hf_f: str
        external dipolar-dipolar
    @param cont_f: str
        external contact terms
    @param skiprows: int
        number of rows to skip in the three files
    @param erbath: float
        maximum distance from the center to be included
    @param center: ndarray with shape (3,)
        position of the central spin
    @return: ndarray
        ndarray with atoms
    """
    dt_read = np.dtype([('N', np.unicode_, 16), ('xyz', np.float64, (3,))])
    dataset = np.loadtxt(coord_f, dtype=dt_read, skiprows=skiprows)

    dt_out = np.dtype([('N', np.unicode_, 16),
                       ('xyz', np.float64, (3,)),
                       ('contact', np.float64),
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
                a['contact'] = np.float64(cl.split()[-1]) * MHZ_TO_RADKHZ

    if erbath is not None and center is not None:
        atoms = atoms[np.linalg.norm(atoms['xyz'] - center, axis=-1) < erbath]

    print('{} external Hypefine tensors were found.'.format(atoms.shape[0]))
    return atoms
