import numpy as np

# hbar mu0 /4pi I have no idea about units, from mengs code
# UPDATE: derived, it checks out
hbar = 1.054571729
# External HF given in MHz, transform to kHz * rad
prefactor = 2 * np.pi * 1000


def read_pos(nspin, center: np.array = None,
                 r_bath: float = 100, skiprows: int = 1):
    """
    :param nspin: nspin is either np.ndarray with dtype [('N', np.unicode_, 16), ('xyz', np.float64, (3,))], which \
    contains isotope type and position or name of the file in which they are stored
    :param center: position of the qubit spin
    :param r_bath: maximum distance from the qubit spin, at which nuclei are added to the bath
    :param skiprows: used when nspin is filename. Number of rows to skip in the file
    :return: np.ndarray with dtype [('N', np.unicode_, 16), ('xyz', np.float64, (3,))]
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
                  gamma_e: float = -17608.597050, external_atoms: np.ndarray = None,
                  error_range: float = 0.5) -> np.ndarray:
    """
    :param atoms_inside: numpy array with dtype [('N', np.unicode_, 16), ('xyz', np.float64, (3,))] containing the \
    coordinates of the nuclear isotope N and it's type
    :param ntype: dictionary, which contains instances of SpinType class with nuclear types, present in the atoms array
    :param gamma_e: gyromagnetic ratio of the qubit spin (rad/kHz/G)
    :param error_range: error range within which the coordinates of atoms in dft cell are considered the same as in \
    the atoms_inside array
    :param external_atoms: np.ndarray containing atoms with predefined hf
    :param center: coordinates of center - position of qubit spin
    :return: np.array with dtype [('N', np.unicode_, 16), ('xyz', np.float64, (3,)), ('A', np.float64, (3, 3))] \
    containing isotope type N, coordinates xyz, hyperfine tensor A
    """

    if center is None:
        center = [0, 0, 0]

    I = np.eye(3)

    dt_out = np.dtype([('N', np.unicode_, 16),
                       ('xyz', np.float64, (3,)),
                       ('A', np.float64, (3, 3))])

    atoms = np.zeros(atoms_inside.shape[0], dtype=dt_out)

    atoms['N'] = atoms_inside['N']
    atoms['xyz'] = atoms_inside['xyz']

    for d in atoms:
        pos = d['xyz'] - center
        r = np.linalg.norm(pos)

        d['A'] = -(3 * np.outer(pos, pos) - I * r ** 2) / (r ** 5) * gamma_e * ntype[d['N']].gyro * hbar

    if external_atoms is not None:
        counter_ext = 0

        for ea in external_atoms:
            possible = np.linalg.norm(atoms['xyz'] - ea['xyz'], axis=1) < error_range
            arrindex = np.nonzero(possible)[0]
            for cindex in arrindex:

                if ea['N'] in atoms[cindex]['N']:
                    atoms[cindex]['xyz'] = ea['xyz']  # Change position to extenal

                    if 'A' in ea.dtype.names:
                        atoms[cindex]['A'] = ea['A'] * prefactor

                    if 'contact' in ea.dtype.names:
                        atoms[cindex]['A'] += I * ea['contact'] * prefactor

                counter_ext += 1

        print('Number of atoms with external HF in dataset: {}'.format(counter_ext))

    print('Number of overall Nuclear spins is {}'.format(atoms.shape[0]))
    return atoms


def read_external(coord_f: str, hf_f: str, cont_f: str, skiprows: int = 1) -> np.ndarray:
    """
    :param coord_f: hf_pos, containing coord of atoms with external hyperfine
    :param hf_f: external dipolar-dipolar
    :param cont_f: external contact terms
    :param skiprows: number of rows to skip in the three files
    :return: np.array with dtype [('N', np.unicode_, 16), ('xyz', np.float64, (3,)), ('contact', np.float64), \
    ('A', np.float64, (3, 3))] containing isotope type N, coordinates xyz, contact term, and hyperfine tensor A
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
                    a['A'][i] = [np.float64(x) for x in gl.split()[2:]]

                next(hf)

    if cont_f:
        with open(cont_f) as contact:
            for i in range(skiprows):
                next(contact)

            for a in atoms:
                cl = next(contact)
                a['contact'] = np.float64(cl.split()[-1])

    print('{} external Hypefine tensors were found.'.format(atoms.shape[0]))
    return atoms
