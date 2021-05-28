import sys
from string import digits

import numpy as np
from pycce.utilities import rotmatrix

from .array import BathArray
from .conc import common_concentrations

err_range = 0.1
FLOAT_ERROR_RANGE = 1e-10


class BathCell:
    r"""
    Generator of the bath spins positions from the unit cell of the material.

    Args:
        a (float): `a` parameter of the primitive cell.
        b (float): `b` parameter of the primitive cell.
        c (float): `c` parameter of the primitive cell.
        alpha (float): :math:`\alpha` angle of the primitive cell.
        beta (float): :math:`\beta` angle of the primitive cell.
        gamma (float): :math:`\gamma` angle of the primitive cell.
        angle (str): units of the :math:`\alpha`, :math:`\beta`, :math:`\gamma` angles.
            Can be either radians (``'rad'``), or degrees (``'deg'``).
        cell (ndarray with shape (3, 3)): Parameters of the cell.

            ``cell`` is 3x3 matrix with columns of coordinates of crystallographic vectors
            in the cartesian reference frame. See ``cell`` attribute.

            If provided, overrides `a`, `b`, and `c`.

    Attributes:

        cell (ndarray with shape (3, 3)): Parameters of the cell.
            ``cell`` is 3x3 matrix with entries:

            .. math::

                [&[a_x\ b_x\ c_x]\\
                &[a_y\ b_y\ c_y]\\
                &[a_z\ b_z\ c_z]]

            where a, b, c are crystallographic vectors
            and x, y, z are their coordinates in the cartesian reference frame.

        atoms (dict): Dictionary containing coordinates and occupancy of each lattice site::

                {atom_1: [array([x1, y1, z1]), array([x2, y2, z2])],
                 atom_2: [array([x3, y3, z3]), ...]}

        isotopes (dict):
            Dictionary containing spin types and their concentration for each lattice site type::

                {atom_1: {spin_1: concentration, spin_2: concentration},
                 atom_2: {spin_3: concentration ...}}

            where ``atom_i`` are lattice site types, and ``spin_i`` are spin types.
    """

    _conv = {'rad': 1, 'deg': 2 * np.pi / 360}
    _coord_types = ['angstrom', 'cell']

    def __init__(self, a=None, b=None, c=None,
                 alpha=None, beta=None, gamma=None,
                 angle='rad', cell=None):

        if angle not in self._conv:
            raise KeyError('Only angle available are: '
                           ', '.join(self._conv.keys()))

        if alpha is None:
            alpha = np.pi / 2
        else:
            alpha = alpha * self._conv[angle]

        if beta is None:
            beta = np.pi / 2
        else:
            beta = beta * self._conv[angle]

        if gamma is None:
            gamma = np.pi / 2
        else:
            gamma = gamma * self._conv[angle]
        if b is None:
            b = a
        if c is None:
            c = a
        # self.state, self.beta, self.gamma = state, beta, gamma
        # self.a, self.b, self.c = a, b, c
        if cell is not None:
            self.cell = np.asarray(cell)
        else:
            inbr = (1 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma) -
                    np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2)

            volume = a * b * c * inbr ** (1 / 2)

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

            self.cell[2, 2] = volume / a / b / np.sin(gamma)

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
        """
        ndarray: z-direction of the reference cartesian coordinate frame in cell coordinates.
        """
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
            rotation = np.array([[0, 1, 0],
                                 [1, 0, 0],
                                 [0, 0, -1]])

        elif abs(a @ b - 1) < FLOAT_ERROR_RANGE:
            rotation = np.eye(3)

        else:
            # a -> b
            rotation = rotmatrix(a, b)

        self.cell = np.linalg.inv(rotation) @ self.cell

        # R =  [n_1^(1) n_1^(2) n_1^(3)]
        #      [n_2^(1) n_2^(2) n_2^(3)]
        #      [n_3^(1) n_3^(2) n_3^(3)]
        # n_i^(j) corresponds to coeff of initial basis vector i
        # for j new basis vector:

        # e'_j = n_1^(j)*e_1 + n_2^(j)*e_2 + n_3^(j)*e_3
        # in other words, columns of matrix are coordinates of the new
        # basis in the old basis.

    def set_zdir(self, direction, type='cell'):
        """
        Set z-direction of the cell.

        Args:
            direction (ndarray with shape (3,)): Direction of the z axis.

            type (str): How coordinates in ``direction`` are stored. If ``type="cell"``,
                assumes crystallographic coordinates. If ``type="angstrom"`` assumes that z direction is given
                in the cartresian reference frame.

        """
        if type == 'cell':
            direction = np.asarray(direction)
        elif type == 'angstrom':
            direction = self.to_cell(np.asarray(direction))
        else:
            raise ValueError('Unknown direction type.')
        self.zdir = direction

    def add_atoms(self, *args, type='cell'):
        """
        Add coordinates of the lattice sites to the unit cell.

        Args:
            *args (tuple): List of tuples, each containing the type of atom N (*str*),
                and the xyz coordinates in the format (*float, float, float*): ``(N, [x, y, z])``.

            type (str): Type of coordinates. Can take values of ``['cell', 'angstrom']``.

                If ``type="cell"``, assumes crystallographic coordinates.

                If ``type="angstrom"`` assumes that coordinates are given in the cartresian reference frame.

        Returns:

            dict:
                View of ``cell.atoms`` dictionary, where each key is the type of lattice site, and each value
                is the list of coordinates in crystallographic frame.

        Examples:

            >>> cell = BathCell(10)
            >>> cell.add_atoms(('C', [0, 0, 0]), ('C', [5, 5, 5]), type='angstrom')
            >>> cell.add_atoms(('Si', [0, 0.5, 0.]), type='cell')
            >>> print(cell.atoms)
            {'C': [array([0., 0., 0.]), array([0.5, 0.5, 0.5])], 'Si': [array([0. , 0.5, 0. ])]}

        """

        for tup in args:
            if type == 'cell':
                coord = np.asarray(tup[1])
            elif type == 'angstrom':
                coord = self.to_cell(tup[1])
            else:
                raise ValueError('Unknown coordinates type.Supported:'
                                 '\n'.join(str(x) for x in self._coord_types))
            if tup[0] in self.atoms:
                self.atoms[tup[0]].append(coord)
            else:
                self.atoms[tup[0]] = [coord]

        return self.atoms

    def add_isotopes(self, *args):
        """
        Add spins that can populate each lattice site type.

        Args:

            *args (tuple or list of tuples): Each tuple can have any of the following formats:

                * Name of the lattice site `N` (`str`), name of the spin `X` (`str`),
                  concentration `c` (`float`, in decimal): ``(N, X, c)``.

                * Isotope name `X and concentration `c`: ``(X, c)``.

                  In this case, the name of the isotope is given in the format
                  ``"{}{}".format(digits, atom_name)`` where ``digits`` is any set of digits 0-9,
                  ``atom_name`` is the name of the corresponding lattice site.
                  Convenient when generating nuclear spin bath.

        Returns:
            dict:
                View of ``cell.isotopes`` dictionary which contains information about lattice site types, spin types,
                and their concentrations::

                    {atom_1: {spin_1: concentration, spin_2: concentration},
                     atom_2: {spin_3: concentration ...}}

        Examples:

            >>> cell = BathCell(10)
            >>> cell.add_atoms(('C', [0, 0, 0]), ('C', [5, 5, 5]), type='angstrom')
            >>> cell.add_isotopes(('C', 'X', 0.001), ('13C', 0.0107))
            >>> print(cell.isotopes)
            {'C': {'X': 0.001, '13C': 0.0107}}

        """

        remove_digits = str.maketrans('', '', digits)

        for tup in args:
            try:

                atom_name = tup[0]
                isotope_name = tup[1]
                concentration = tup[2]

                if atom_name in self.isotopes:
                    self.isotopes[atom_name][isotope_name] = concentration

                else:
                    self.isotopes[atom_name] = {isotope_name: concentration}

            except IndexError:

                isotope_name = tup[0]
                concentration = tup[1]
                atom_name = isotope_name.translate(remove_digits)

                if atom_name in self.isotopes:
                    self.isotopes[atom_name][isotope_name] = concentration
                else:
                    self.isotopes[atom_name] = {isotope_name: concentration}

        return self.isotopes

    def gen_supercell(self, size, add=None, remove=None, seed=None):
        """
        Generate supercell populated with spins.

        .. note::

            If ``isotopes`` were not provided, assumes the natural concentration of nuclear spin isotopes for each
            lattice site type. However, if any isotope concentration is provided,
            then uses only user-defined ones.

        Args:
            size (float): Approximate linear size of the supercell. The generated supercell will have
                minimal distance between opposite sides larger than this parameter.

            add (tuple or list of tuples):
                Tuple or list of tuples containing common_isotopes to add as a defect.
                Each tuple contains name of the new isotope
                and its coordinates in the cell basis: ``(isotope_name, x_cell, y_cell, z_cell)``.

            remove (tuple or list of tuples):
                Tuple or list of tuples containing bath to remove in the defect.
                Each tuple contains name of the atom to remove
                and its coordinates in the cell basis: ``(atom_name, x_cell, y_cell, z_cell)``.

            seed (int): Seed for random number generator.

        .. note::

            While ``add`` takes the **spin** name as an argument, ``remove`` takes the lattice site name.

        Returns:
            BathArray: Array of the spins in the given supercell.
        """
        if not self.isotopes:
            isotopes = {}

            for a in self.atoms:

                try:
                    isotopes[a] = common_concentrations[a]
                except KeyError:
                    pass

        else:
            isotopes = self.isotopes

        rgen = np.random.default_rng(seed)

        axb = np.cross(self.cell[:, 0], self.cell[:, 1])
        bxc = np.cross(self.cell[:, 1], self.cell[:, 2])
        cxa = np.cross(self.cell[:, 2], self.cell[:, 0])

        anumber = int(size * np.linalg.norm(bxc) / (bxc @ self.cell[:, 0]) + 1)
        bnumber = int(size * np.linalg.norm(cxa) / (cxa @ self.cell[:, 1]) + 1)
        cnumber = int(size * np.linalg.norm(axb) / (axb @ self.cell[:, 2]) + 1)
        # print(anumber, bnumber, cnumber)

        dt = np.dtype([('N', np.unicode_, 16), ('xyz', np.float64, (3,))])
        atoms = []

        for a in isotopes:
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

            for i in isotopes[a]:
                conc = isotopes[a][i]
                nisotopes = int(round(natoms * conc))
                seedsites = rgen.choice(atom_seedsites[~mask],
                                        nisotopes, replace=False,
                                        shuffle=False)

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
        # bath = bath[np.linalg.norm(bath['xyz'], axis=1) <= size]

        defective_atoms = defect(self.cell, atoms, add=add, remove=remove)
        bath = BathArray(array=defective_atoms)
        return bath

    def to_cartesian(self, coord):
        """
        Transform coordinates from crystallographic basis to the cartesian reference frame.

        Args:
            coord (ndarray with shape (3,) or (n, 3)): Coordinates in crystallographic basis or array of coordinates.

        Returns:
            ndarray with shape (3,) or (n, 3): Cartesian coordinates in angstrom.
        """

        coord = np.asarray(coord)
        if len(coord.shape) == 1:
            return self.cell @ coord

        elif len(coord.shape) == 2:
            return np.einsum('jk,ik->ij', self.cell, coord)
        else:
            raise ValueError('Improper coordinates format')

    def to_cell(self, coord):
        """
        Transform coordinates from the cartesian coordinates of the reference frame to the cell coordinates.

        Args:
            coord (ndarray with shape (3,) or (n, 3)): Cartesian coordinates in angstrom or array of coordinates.

        Returns:
            ndarray with shape (3,) or (n, 3): Coordinates in the cell basis.

        """
        coord = np.asarray(coord)

        if len(coord.shape) == 1:
            return np.linalg.inv(self.cell) @ coord

        elif len(coord.shape) == 2:
            return np.einsum('jk,ik->ij', np.linalg.inv(self.cell), coord)
        else:
            raise ValueError('Improper coordinates format')

    @classmethod
    def from_ase(cls, atoms_object):
        """
        Generate ``BathCell`` instance from ``ase.Atoms`` object of Atomic Simulations Environment (ASE) package.

        Args:
            atoms_object (Atoms): Atoms object, used to generate new ``BathCell`` instance.

        Returns:
            BathCell: New instance of the ``BathCell`` with atoms read from ``ase.Atoms``.
        """

        spin_cell = cls(cell=atoms_object.cell[:].T)
        positions = atoms_object.get_scaled_positions(wrap=True)
        symbols = atoms_object.get_chemical_symbols()

        zdr = np.linalg.inv(spin_cell.cell) @ np.array([0, 0, 1])

        zdr = zdr / np.linalg.norm(zdr)

        spin_cell._zdir = zdr

        for s in symbols:
            spin_cell.atoms[s] = []

        for sym, pos in zip(symbols, positions):
            spin_cell.atoms[sym].append(pos)

        return spin_cell

    def __repr__(self):
        m = f"{type(self).__name__} containing:\n"
        na = 0
        for k in self.atoms:
            na += 1
            m += f"{len(self.atoms[k])} positions for {k}"
            try:
                im = ' with'
                for ik in self.isotopes[k]:
                    im += f" {ik}: {self.isotopes[k][ik]}"
                im += '.\n'
            except KeyError:
                im = ".\n"
            m += im
        if not na:
            m += "No atomic positions.\n"

        m += f"\nCell:\n{self.cell}\n"
        m += f"\nz-direction: {self.zdir}\n"

        return m


def defect(cell, atoms, add=None, remove=None):
    """
    Generate a defect in the given supercell.

    The defect will be located in the unit cell, located roughly
    in the middle of the supercell, generated by ``BathCell``, such that (0, 0, 0) of cartesian reference frame
    is located at (0, 0, 0) position of this unit cell.

    Args:
        cell (ndarray with shape (3, 3)): parameters of the unit cell.

        atoms (BathArray): Array of spins in the supercell.

        add (tuple or list of tuples):
            Add spin type(s) to the supercell at specified positions to create point defect.
            Each tuple contains name of the new isotope
            and its coordinates in the cell basis: ``(isotope_name, x_cell, y_cell, z_cell)``.

        remove (tuple or list of tuples):
            Remove lattice site from the supercell at specified position to create point defect.
            Each tuple contains name of the atom to remove
            and its coordinates in the cell basis: ``(atom_name, x_cell, y_cell, z_cell)``.

    Returns:
        BathArray: Array of spins with the defect added.
    """

    defective_atoms = atoms.copy()
    dt = atoms.dtype

    where = None
    if remove is not None and isinstance(remove[0], str):
        name = remove[0]
        position_cc = np.asarray(remove[1])  # Given in the cell coordinates

        position = cell @ position_cc

        offsets = np.linalg.norm((atoms['xyz'] - position), axis=1)
        where = np.logical_and(np.core.defchararray.find(atoms['N'], name) != -1,
                               offsets <= err_range)

    elif remove is not None:
        where = np.zeros(defective_atoms.shape, dtype=bool)

        for removals in remove:
            name = removals[0]
            position_cc = np.asarray(removals[1])  # Given in the cell coordinates

            position = cell @ position_cc
            # print(name, position)
            # print(np.core.defchararray.find(bath['N'], name) != -1)
            offsets = np.linalg.norm((atoms['xyz'] - position), axis=1)
            # print(offsets <= err_range)
            where += np.logical_and(np.core.defchararray.find(atoms['N'], name) != -1,
                                    offsets <= err_range)
    if np.count_nonzero(where):
        # print('I see {} removals'.format(np.count_nonzero(where)))
        # print('Removing: \n', bath[where])

        defective_atoms = defective_atoms[~where]

    if add is not None and isinstance(add[0], str):
        name = add[0]
        position_cc = np.asarray(add[1])  # Given in the cell coordinates

        position = cell @ position_cc

        newentry = np.array((name, position), dtype=dt)

        # print('Adding: \n', newentry)
        defective_atoms = np.append(defective_atoms, newentry)

    elif add is not None:
        newlist = []

        for addition in add:
            name = addition[0]
            position_cc = np.asarray(addition[1])  # Given in the cell coordinates

            position = cell @ position_cc

            newentry = np.array((name, position), dtype=dt)

            newlist.append(newentry)

        # print('Adding: \n', np.asarray(newlist))

        defective_atoms = np.append(defective_atoms, newlist)

    return defective_atoms
