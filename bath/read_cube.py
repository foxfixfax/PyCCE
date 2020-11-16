import numpy as np
from ..unit_conversion import BOHR_TO_ANGSTROM


# Copied from ASE
chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']


class Cube:
    _dt = np.dtype([('N', np.unicode_, 16), ('xyz', np.float64, (3,))])

    def __init__(self, filename, savecoord=True):
        self.cell = np.empty(3, 3, dtype=np.float64)

        with open(filename, "r") as content:
            # first two lines are comments
            self.comments = next(content).strip() + "\n" + next(content).strip()

            # total number of atoms | xyz of the cube origin
            tot = next(content).split()
            natoms = int(tot[0])

            self.origin = np.array([float(x) for x in tot[1:]])
            self.voxel = np.empty(3,3, dtype=np.float64)
            self.size = np.empty(3, dtype=np.int32)

            if savecoord:
                self.cell = np.empty(natoms, dtype=self._dt)
            else:
                self.cell = None

            for i in range(3):
                tot = next(content).split()
                self.size[i] = int(tot[0])

                if self.size[i] < 0:
                    self.voxel[i] = [float(x) for x in tot[1:]]

                else:
                    self.voxel[i] = [float(x) * BOHR_TO_ANGSTROM for x in tot[1:]]

            for j in range(natoms):
                tot = next(content).split()

                if savecoord:
                    self.cell[j]['N'] = chemical_symbols[int(tot[0])]
                    self.cell[j]['xyz'] = [float(x) for x in tot[1:]]

            data = [[float(x) for x in line.split()] for line in content]

            self.data = np.array(data).reshape(size)






