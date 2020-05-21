import numpy as np
import sys
from string import digits
import collections

err_range = 0.1
FLOAT_ERROR_RANGE = 1e-10


class NSpinCell:
    _conv = {'rad': 1, 'deg': 2 * np.pi / 360}

    def __init__(self, a=0, b=0, c=0,
                 alpha=None, beta=None, gamma=None,
                 units='rad'):

        if units not in self._conv:
            raise KeyError('Only units available are: ' +
                           ', '.join(self._conv.keys()))

        if alpha is None:
            alpha = np.pi / 2
        else:
            alpha = alpha * self._conv[units]

        if beta is None:
            beta = np.pi / 2
        else:
            beta = beta * self._conv[units]

        if gamma is None:
            gamma = np.pi / 2
        else:
            gamma = gamma * self._conv[units]

        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.a, self.b, self.c = a, b, c

        inbr = (1 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma) -
                np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2)

        self.volume = a * b * c * inbr ** (1 / 2)

        self.cell = np.zeros((3, 3))
        self.cell[0, 0] = a
        self.cell[0, 1] = b * np.cos(gamma)
        self.cell[1, 1] = b * np.sin(gamma)
        self.cell[0, 2] = c * np.cos(beta)
        self.cell[1, 2] = c * (np.cos(alpha) - np.cos(beta) *
                               np.cos(gamma)) / np.sin(gamma)
        # cell is 3x3 matrix with entries:
        # cell = [a_x b_x c_x]
        #        [a_y b_y c_y]
        #        [a_z b_z c_z]

        self.cell[2, 2] = self.volume / a / b / np.sin(gamma)

        if np.linalg.cond(self.cell) < 1 / sys.float_info.epsilon:
            zdr = np.linalg.inv(self.cell) @ np.array([0, 0, 1])
            zdr = zdr / np.linalg.norm(zdr)
        else:
            zdr = np.zeros(3)

        self._zdir = zdr

        self.atoms = {}
        self.isotopes = {}

    @property
    def zdir(self):
        return self._zdir

    @zdir.setter
    def zdir(self, direction):
        self._zdir = np.asarray(direction)  # Stored in the cell coordinates

        a = np.array([0, 0, 1])  # Initial vector

        ud = self.cell @ self._zdir
        b = ud / np.linalg.norm(ud)  # Final vector

        # Rotational matrix
        # If z direction is opposite
        if abs(a @ b + 1) < FLOAT_ERROR_RANGE:
            R = np.array([[0, 1, 0],
                          [1, 0, 0],
                          [0, 0, -1]])

        elif abs(a @ b - 1) < FLOAT_ERROR_RANGE:
            R = np.eye(3)

        else:
            R = rotmatrix(a, b)

        self.cell = np.linalg.inv(R) @ self.cell

        # R =  [n_1^(1) n_1^(2) n_1^(3)]
        #      [n_2^(1) n_2^(2) n_2^(3)]
        #      [n_3^(1) n_3^(2) n_3^(3)]
        # n_i^(j) corresponds to coeff of initial basis vector i
        # for j new basis vector:

        # e'_j = n_1^(j)*e_1 + n_2^(j)*e_2 + n_3^(j)*e_3
        # in other words, columns of M are coordinates of the new
        # basis in the old basis.

    def add_atoms(self, *args):

        for tup in args:

            if tup[0] in self.atoms:
                self.atoms[tup[0]].append(np.asarray(tup[1]))
            else:
                self.atoms[tup[0]] = [np.asarray(tup[1])]

        return self.atoms

    def add_isotopes(self, *args):
        remove_digits = str.maketrans('', '', digits)

        for tup in args:
            isotope_name = tup[0]
            atom_name = isotope_name.translate(remove_digits)

            if atom_name in self.isotopes:
                self.isotopes[atom_name][isotope_name] = tup[1]
            else:
                self.isotopes[atom_name] = {isotope_name: tup[1]}
        return self.isotopes

    def gen_supercell(self, size, add=None, remove=None):
        axb = np.cross(self.cell[:, 0], self.cell[:, 1])
        bxc = np.cross(self.cell[:, 1], self.cell[:, 2])
        cxa = np.cross(self.cell[:, 2], self.cell[:, 0])

        anumber = int(size * np.linalg.norm(bxc) / (bxc @ self.cell[:, 0]) + 1)
        bnumber = int(size * np.linalg.norm(cxa) / (cxa @ self.cell[:, 1]) + 1)
        cnumber = int(size * np.linalg.norm(axb) / (axb @ self.cell[:, 2]) + 1)
        # print(anumber, bnumber, cnumber)

        dt = np.dtype([('N', np.unicode_, 16), ('xyz', np.float64, (3,))])
        atoms = []

        for a in self.isotopes:
            nsites = len(self.atoms[a])
            # print(nsites)
            sites_xyz = np.asarray(self.atoms[a]) @ self.cell.T
            # print(sites_xyz)
            maxind = np.array([anumber,
                               bnumber,
                               cnumber,
                               nsites], dtype=np.int32)

            natoms = np.prod(maxind, dtype=np.int32)
            atom_seedsites = np.arange(natoms, dtype=np.int32)
            mask = np.zeros(natoms, dtype=bool)

            for i in self.isotopes[a]:
                conc = self.isotopes[a][i]
                nisotopes = int(round(natoms * conc))
                seedsites = np.sort(np.random.choice(atom_seedsites[~mask],
                                                     nisotopes, replace=False))

                mask += np.isin(atom_seedsites, seedsites)
                bcn = bnumber * cnumber * nsites
                cn = cnumber * nsites

                aindexes = seedsites // bcn - (anumber - 1) // 2  # recenter at 0
                bindexes = (seedsites % bcn) // cn - (bnumber - 1) // 2
                cindexes = ((seedsites % bcn) % cn) // nsites - (cnumber - 1) // 2

                # indexes of the sites
                nindexes = ((seedsites % bcn) % cn) % nsites

                indexes = np.column_stack((aindexes,
                                           bindexes,
                                           cindexes))

                uc_positions = np.einsum('jk,ik->ij', self.cell, indexes)

                subatoms = np.zeros(indexes.shape[0], dtype=dt)
                subatoms['N'] = i
                subatoms['xyz'] = uc_positions + sites_xyz[nindexes]

                atoms.append(subatoms)

        atoms = np.concatenate(atoms)
        atoms = atoms[np.linalg.norm(atoms['xyz'], axis=1) <= size]

        defective_atoms = defect(self.cell, atoms, add=add, remove=remove)
        return defective_atoms

    def cell_to_cartesian(self, coord):
        return self.cell @ np.asarray(coord)

    @classmethod
    def from_ase_Atoms(cls, atoms_object):
        self = cls()
        self.cell = atoms_object.cell[:].T
        positions = atoms_object.get_scaled_positions(wrap=True)
        symbols = atoms_object.get_chemical_symbols()

        zdr = np.linalg.inv(self.cell) @ np.array([0, 0, 1])
        zdr = zdr / np.linalg.norm(zdr)
        self._zdir = zdr
        for s in symbols:
            self.atoms[s] = []
        for sym, pos in zip(symbols, positions):
            self.atoms[sym].append(pos)
        return self


def defect(cell, atoms, add=None, remove=None):
    defective_atoms = atoms.copy()
    dt = atoms.dtype

    if isinstance(remove[0], str):
        name = remove[0]
        position_cc = np.asarray(remove[1])  # Given in the cell coordinates

        position = cell @ position_cc

        offsets = np.linalg.norm((atoms['xyz'] - position), axis=1)
        where = np.logical_and(atoms['N'] == name,
                               offsets <= err_range)

        defective_atoms = atoms[~where]

    elif remove is not None:
        where = np.zeros(defective_atoms.shape, dtype=bool)

        for removals in remove:
            name = removals[0]
            position_cc = np.asarray(removals[1])  # Given in the cell coordinates

            position = cell @ position_cc
            # print(name, position)
            print(np.core.defchararray.find(atoms['N'], name) != -1)
            offsets = np.linalg.norm((atoms['xyz'] - position), axis=1)
            # print(offsets <= err_range)
            where += np.logical_and(np.core.defchararray.find(atoms['N'], name) != -1,
                                    offsets <= err_range)

        print('I see {} removals'.format(np.count_nonzero(where)))
        print('Removing: \n', atoms[where])

        defective_atoms = atoms[~where]

    if isinstance(add, str):
        name = add[0]
        position_cc = np.asarray(add[1])  # Given in the cell coordinates

        position = cell @ position_cc

        newentry = np.array([name, position], dtype=dt)

        defective_atoms = np.append(defective_atoms, newentry)

    elif add is not None:
        newlist = []

        for addition in add:
            name = addition[0]
            position_cc = np.asarray(addition[1])  # Given in the cell coordinates

            position = cell @ position_cc

            newentry = np.array([name, position], dtype=dt)

            newlist.append(newentry)

        defective_atoms = np.append(defective_atoms, newlist)

    return defective_atoms


def rotmatrix(initial_vector, final_vector):
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
