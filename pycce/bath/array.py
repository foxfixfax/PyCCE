import warnings
from collections import UserDict, Mapping

import numpy as np

from ..units import HBAR, ELECTRON_GYRO


class BathArray(np.ndarray):
    _dtype_bath = np.dtype([('N', np.unicode_, 16),
                            ('xyz', np.float64, (3,)),
                            ('A', np.float64, (3, 3)),
                            ('Q', np.float64, (3, 3))])

    def __new__(subtype, shape=None, array=None,
                spin_names=None, hyperfines=None, quadrupoles=None,
                ca=None, sn=None, hf=None, q=None,
                spin_types=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments. This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__

        if array is None and ca is not None:
            array = ca

        if spin_names is None and sn is not None:
            spin_names = sn

        if hyperfines is None and hf is not None:
            hyperfines = hf

        if quadrupoles is None and q is not None:
            quadrupoles = q

        if shape is not None:
            # set the new 'info' attribute to the value passed
            obj = super(BathArray, subtype).__new__(subtype, shape, dtype=subtype._dtype_bath)
        else:
            for a in (array, hyperfines, quadrupoles):
                if a is not None:
                    obj = super(BathArray, subtype).__new__(subtype, (np.asarray(a).shape[0],),
                                                            dtype=subtype._dtype_bath)
                    break
            else:
                raise ValueError('No shape provided')

        obj.types = SpinDict()

        if spin_types is not None:
            try:
                obj.add_types(**spin_types)
            except TypeError:
                obj.add_types(*spin_types)

        if array is not None:
            array = np.asarray(array)
            if array.dtype.names is not None:
                for n in array.dtype.names:
                    obj[n] = array[n]
            else:
                obj['xyz'] = array.reshape(-1, 3)

        if spin_names is not None:
            obj['N'] = np.asarray(spin_names).reshape(-1)
        if hyperfines is not None:
            obj['A'] = np.asarray(hyperfines).reshape(-1, 3, 3)
        if quadrupoles is not None:
            obj['Q'] = np.asarray(quadrupoles).reshape(-1, 3, 3)

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(BathArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the BathArray.__new__
        #    constructor, and BathArray stuff will be set when we return to
        #    BathArray.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(BathArray):
        #    obj is arr
        #    (type(obj) can be BathArray)
        # From new-from-template - e.g arr[:3]
        #    type(obj) is BathArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'types', because this
        # method sees all creation of default objects - with the
        # BathArray.__new__ constructor, but also with
        # arr.view(BathArray).
        self.types = getattr(obj, 'types', SpinDict())
        # We do not need to return anything

    @property
    def isotope(self):
        return self.types[self].isotope

    @property
    def s(self):
        return self.types[self].s

    @property
    def gyro(self):
        return self.types[self].gyro

    @property
    def q(self):
        return self.types[self].q

    def __setitem__(self, key, val):
        # look at the units - convert the values to what they need to be (in
        # the base_unit) and then delegate to the ndarray.__setitem__
        if isinstance(val, (np.str_, str)):
            if val not in self.types.keys():
                try:
                    self.types[val] = common_isotopes[val]
                except KeyError:
                    warnings.warn('Spin type for {} was not provided and was not found in common isotopes.'.format(val))
        elif isinstance(val, np.ndarray) and isinstance(val[0], (np.str_, str)):
            for n in np.unique(val):
                if n not in self.types.keys():
                    try:
                        self.types[n] = common_isotopes[n]
                    except KeyError:
                        warnings.warn(
                            'Spin type for {} was not provided and was not found in common isotopes.'.format(n))
        elif isinstance(val, np.ndarray) and (val.dtype.names != None) and ('N' in val.dtype.names):
            for n in np.unique(val['N']):
                if n not in self.types.keys():
                    try:
                        self.types[n] = common_isotopes[n]
                    except KeyError:
                        warnings.warn(
                            'Spin type for {} was not provided and was not found in common isotopes.'.format(n))
        return np.ndarray.__setitem__(self, key, val)

    def add_type(self, *args, **kwargs):
        self.types.add_type(*args, **kwargs)

    def update(self, ext_bath, error_range=0.2, ignore_isotopes=True, inplace=True):
        bath = combine_bath(self, ext_bath, error_range, ignore_isotopes, inplace)
        return bath

    def from_point_dipole(self, position, gyro_e=ELECTRON_GYRO):

        identity = np.eye(3, dtype=np.float64)
        pos = self['xyz'] - position

        gyros = self.types[self].gyro
        posxpos = np.einsum('ki,kj->kij', pos, pos)

        r = np.linalg.norm(pos, axis=1)[:, np.newaxis, np.newaxis]

        pref = (gyro_e * gyros * HBAR)[:, np.newaxis, np.newaxis]

        self['A'] = -(3 * posxpos[np.newaxis, :] - identity[np.newaxis, :] * r ** 2) / (r ** 5) * pref
        return

    def from_cube(self, cube, gyro_e=ELECTRON_GYRO):
        gyros = self.types[self].gyro
        self['A'] = cube.integrate(self['xyz'], gyros, gyro_e)
        return

    def from_func(self, func, gyro_e=ELECTRON_GYRO):
        for a in self:
            a['A'] = func(a['xyz'], self.type[a].gyro, gyro_e)
        return

    def dist(self, pos=None):
        if pos is None:
            pos = np.zeros(3)
        else:
            pos = np.asarray(pos)

        return np.linalg.norm(self['xyz'] - pos, axis=-1)

class SpinType:
    """
    Class which contains properties of each spin type in the bath

    Parameters
    ----------
    @param isotope: str
        Name of the bath spin
    @param s: float
        Total spin
    @param gyro:
        Gyromagnetic ratio in rad/(ms * G)
    @param q:
        Quadrupole moment in millibarn (for s > 1/2)
    """

    def __init__(self, isotope, s=0., gyro=0., q=0.):
        self.isotope = isotope
        self.s = s
        self.gyro = gyro
        self.q = q

    def __repr__(self):
        return f'({self.isotope}, {self.s}, {self.gyro}, {self.q})'


class SpinDict(UserDict):
    """
    Wrapper class for dictionary with easier access to the values. Can take np.void or BathArray instances as keys.
    Every entry has to be SpinType

    """

    def __delitem__(self, key):
        if isinstance(key, np.void):
            k = key['N']
            super().__delitem__(k)

        elif isinstance(key, (np.ndarray, BathArray)):
            for k in key['N']:
                super().__delitem__(k)

        else:
            super().__delitem__(key)

    def __setitem__(self, key, value):

        if isinstance(key, np.void):
            if not isinstance(value, SpinType):
                k = key['N']
                if value[0] == k:
                    value = SpinType(*value)
                else:
                    value = SpinType(k, *value)
            super().__setitem__(k, value)

        elif isinstance(key, (np.ndarray, BathArray)):
            for k, v in zip(key['N'], value):
                if not isinstance(v, SpinType):
                    if v[0] == k:
                        v = SpinType(*v)
                    else:
                        v = SpinType(k, *v)
                super().__setitem__(k, v)

        else:
            if not isinstance(value, SpinType):
                if value[0] == key:
                    value = SpinType(*value)
                else:
                    value = SpinType(key, *value)
            super().__setitem__(key, value)

    def __getitem__(self, key):
        if isinstance(key, np.void):
            k = key['N']
            return super().__getitem__(k)
        elif isinstance(key, (np.ndarray, BathArray)):
            spins = np.empty(key.shape, dtype=np.float64)
            gyros = np.empty(key.shape, dtype=np.float64)
            quads = np.empty(key.shape, dtype=np.float64)

            for n in np.unique(key['N']):
                spins[key['N'] == n] = super().__getitem__(n).s
                gyros[key['N'] == n] = super().__getitem__(n).gyro
                quads[key['N'] == n] = super().__getitem__(n).q

            return SpinType(key, s=spins, gyro=gyros, q=quads)
        else:
            return super().__getitem__(key)

    def __repr__(self):
        return f"{type(self).__name__}({self.data})"

    def add_type(self, *args, **kwargs):
        try:
            for nuc in args:
                if isinstance(nuc, SpinType):
                    self[nuc.isotope] = nuc
                elif isinstance(nuc, Mapping):
                    self.update(nuc)
                else:
                    self[nuc[0]] = SpinType(*nuc)
        except TypeError:
            self[args[0]] = SpinType(*args)

        for nuc in kwargs:
            self[nuc] = kwargs[nuc]


def gen_spindict(*spin_types):
    ntype = SpinDict()
    for nuc in spin_types:
        if isinstance(nuc, SpinType):
            ntype[nuc.isotope] = nuc
        else:
            ntype[nuc[0]] = SpinType(*nuc)
    return ntype


def same_bath_indexes(barray_1, barray_2, error_range=0.2, ignore_isotopes=True):
    # counter_ext = 0
    dist_matrix = np.linalg.norm(barray_1['xyz'][:, np.newaxis, :] - barray_2['xyz'][np.newaxis, :, :], axis=-1)
    if ignore_isotopes:
        tb_names = np.core.defchararray.strip(barray_1['N'], '1234567890')
        ab_names = np.core.defchararray.strip(barray_2['N'], '1234567890')
        same_names = tb_names[:, np.newaxis] == ab_names[np.newaxis, :]
    else:
        same_names = barray_1['N'][:, np.newaxis] == barray_2['N'][np.newaxis, :]

    criteria = np.logical_and(dist_matrix < error_range, same_names)
    indexes, ext_indexes = np.nonzero(criteria)

    # Check for uniqueness. If several follow the criteria, use the first one appearing.
    _, uind = np.unique(indexes, return_index=True)
    indexes = indexes[uind]
    ext_indexes = ext_indexes[uind]
    return indexes, ext_indexes


def combine_bath(total_bath, added_bath, error_range=0.2, ignore_isotopes=True, inplace=True):
    if not inplace:
        total_bath = total_bath.copy()

    indexes, ext_indexes = same_bath_indexes(total_bath, added_bath, error_range, ignore_isotopes)
    total_bath[indexes] = added_bath[ext_indexes]

    return total_bath


# Dictionary of the common isotopes. Placed in this file to avoid circular dependency
common_isotopes = SpinDict()

# electron spin
common_isotopes['e'] = SpinType('e', 1 / 2, ELECTRON_GYRO, 0)

# H
common_isotopes['1H'] = SpinType('1H', 1 / 2, 26.7519, 0)
common_isotopes['2H'] = SpinType('2H', 1, 4.1066, 0.00286)
common_isotopes['3H'] = SpinType('3H', 1 / 2, 28.535, 0)

# He
common_isotopes['3He'] = SpinType('3He', 1 / 2, -20.38, 0)

# Li
common_isotopes['6Li'] = SpinType('6Li', 1, 3.9371, -0.000806)
common_isotopes['7Li'] = SpinType('7Li', 3 / 2, 10.3976, -0.0400)

# Be
common_isotopes['9Be'] = SpinType('9Be', 3 / 2, -3.9575, 0.0529)

# B
common_isotopes['10B'] = SpinType('10B', 3, 2.875, 0.0845)
common_isotopes['11B'] = SpinType('11B', 3 / 2, 8.584, 0.04059)

# C
common_isotopes['13C'] = SpinType('13C', 1 / 2, 6.7283, 0)

# N
common_isotopes['14N'] = SpinType('14N', 1, 1.9338, 0.02044)
common_isotopes['15N'] = SpinType('15N', 1 / 2, -2.712, 0)

# O
common_isotopes['17O'] = SpinType('17O', 5 / 2, -3.6279, -0.0256)

# F
common_isotopes['19F'] = SpinType('19F', 1 / 2, 25.181, 0)

# Ne
common_isotopes['21Ne'] = SpinType('21Ne', 3 / 2, -2.113, 0.102)

# Na
common_isotopes['23Na'] = SpinType('23Na', 3 / 2, 7.0801, 0.104)

# Mg
common_isotopes['25Mg'] = SpinType('25Mg', 5 / 2, -1.639, 0.199)

# Al
common_isotopes['27Al'] = SpinType('27Al', 5 / 2, 6.976, 0.1466)

# Si
common_isotopes['29Si'] = SpinType('29Si', 1 / 2, -5.3188, 0)

# P
common_isotopes['31P'] = SpinType('31P', 1 / 2, 10.841, 0)

# spin_matrix
common_isotopes['33S'] = SpinType('33S', 3 / 2, 2.055, -0.0678)

# Cl
common_isotopes['35Cl'] = SpinType('35Cl', 3 / 2, 2.624, -0.0817)
common_isotopes['37Cl'] = SpinType('37Cl', 3 / 2, 2.1842, -0.0644)

# K
common_isotopes['39K'] = SpinType('39K', 3 / 2, 1.2498, 0.0585)
common_isotopes['41K'] = SpinType('41K', 3 / 2, 0.686, 0.0711)

# Ca
common_isotopes['43Ca'] = SpinType('43Ca', 7 / 2, -1.8025, -0.0408)

# Sc
common_isotopes['45Sc'] = SpinType('45Sc', 7 / 2, 6.5081, -0.220)

# Ti
common_isotopes['47Ti'] = SpinType('47Ti', 5 / 2, -1.5105, 0.302)
common_isotopes['49Ti'] = SpinType('49Ti', 7 / 2, -1.5109, 0.247)

# V
common_isotopes['50V'] = SpinType('50V', 6, 2.6717, 0.21)
common_isotopes['51V'] = SpinType('51V', 7 / 2, 7.0453, -0.043)

# Cr
common_isotopes['53Cr'] = SpinType('53Cr', 3 / 2, -1.512, 0.15)

# Mn
common_isotopes['55Mn'] = SpinType('55Mn', 5 / 2, 6.608, 0.330)

# Fe
common_isotopes['57Fe'] = SpinType('57Fe', 1 / 2, 0.8661, 0)

# Co
common_isotopes['59Co'] = SpinType('59Co', 7 / 2, 6.317, 0.42)

# Ni
common_isotopes['61Ni'] = SpinType('61Ni', 3 / 2, -2.394, 0.162)

# Cu
common_isotopes['63Cu'] = SpinType('63Cu', 3 / 2, 7.0974, -0.220)
common_isotopes['65Cu'] = SpinType('65Cu', 3 / 2, 7.6031, -0.204)

# Zn
common_isotopes['67Zn'] = SpinType('67Zn', 5 / 2, 1.6768, 0.150)

# Ga
common_isotopes['69Ga'] = SpinType('69Ga', 3 / 2, 6.4323, 0.171)
common_isotopes['71Ga'] = SpinType('71Ga', 3 / 2, 8.1731, 0.107)

# Ge
common_isotopes['73Ge'] = SpinType('73Ge', 9 / 2, -0.9357, -0.196)

# As
common_isotopes['75As'] = SpinType('75As', 3 / 2, 4.595, 0.314)

# Se
common_isotopes['77Se'] = SpinType('77Se', 1 / 2, 5.12, 0)

# Br
common_isotopes['79Br'] = SpinType('79Br', 3 / 2, 6.7228, 0.313)
common_isotopes['81Br'] = SpinType('81Br', 3 / 2, 7.2468, 0.262)

# Kr
common_isotopes['83Kr'] = SpinType('83Kr', 9 / 2, -1.033, 0.259)

# Rb
common_isotopes['85Rb'] = SpinType('85Rb', 5 / 2, 2.5909, 0.276)
common_isotopes['87Rb'] = SpinType('87Rb', 3 / 2, 8.7807, 0.1335)

# Sr
common_isotopes['87Sr'] = SpinType('87Sr', 9 / 2, -1.163, 0.305)

# Y
common_isotopes['89Y'] = SpinType('89Y', 1 / 2, -1.3155, 0)

# Zr
common_isotopes['91Zr'] = SpinType('91Zr', 5 / 2, -2.4959, -0.176)

# Nb
common_isotopes['93Nb'] = SpinType('93Nb', 9 / 2, 6.564, -0.32)

# Mo
common_isotopes['95Mo'] = SpinType('95Mo', 5 / 2, 1.75, -0.022)
common_isotopes['97Mo'] = SpinType('97Mo', 5 / 2, -1.787, 0.255)

# Ru
common_isotopes['99Ru'] = SpinType('99Ru', 3 / 2, -1.234, 0.079)
common_isotopes['101Ru'] = SpinType('101Ru', 5 / 2, -1.383, 0.46)

# Rh
common_isotopes['103Rh'] = SpinType('103Rh', 1 / 2, -0.846, 0)

# Ag
common_isotopes['107Ag'] = SpinType('107Ag', 1 / 2, -1.087, 0)
common_isotopes['109Ag'] = SpinType('109Ag', 1 / 2, -1.25, 0)

# Cd
common_isotopes['111Cd'] = SpinType('111Cd', 1 / 2, -5.6926, 0)
common_isotopes['113Cd'] = SpinType('113Cd', 1 / 2, -5.955, 0)

# In
common_isotopes['113In'] = SpinType('113In', 9 / 2, 5.8782, 0.759)
common_isotopes['115In'] = SpinType('115In', 9 / 2, 5.8908, 0.770)

# Sn
common_isotopes['115Sn'] = SpinType('115Sn', 1 / 2, -8.8014, 0)
common_isotopes['117Sn'] = SpinType('117Sn', 1 / 2, -9.589, 0)
common_isotopes['119Sn'] = SpinType('119Sn', 1 / 2, -10.0138, 0)

# Sb
common_isotopes['121Sb'] = SpinType('121Sb', 5 / 2, 6.4355, -0.543)
common_isotopes['123Sb'] = SpinType('123Sb', 7 / 2, 3.4848, -0.692)

# Te
common_isotopes['123Te'] = SpinType('123Te', 1 / 2, -7.049, 0)
common_isotopes['125Te'] = SpinType('125Te', 1 / 2, -8.498, 0)

# I
common_isotopes['127I'] = SpinType('127I', 5 / 2, 5.3817, -0.696)

# Xe
common_isotopes['129Xe'] = SpinType('129Xe', 1 / 2, -7.441, 0)
common_isotopes['131Xe'] = SpinType('131Xe', 3 / 2, 2.206, -0.114)

# Cs
common_isotopes['133Cs'] = SpinType('133Cs', 7 / 2, 3.5277, -0.00343)

# Ba
common_isotopes['135Ba'] = SpinType('135Ba', 3 / 2, 2.671, 0.160)
common_isotopes['137Ba'] = SpinType('137Ba', 3 / 2, 2.988, 0.245)

# La
common_isotopes['139La'] = SpinType('139La', 7 / 2, 3.801, 0.200)

# Pr
common_isotopes['141Pr'] = SpinType('141Pr', 5 / 2, 3.2763, -0.077)

# Nd
common_isotopes['143Nd'] = SpinType('143Nd', 7 / 2, -1.45735, -0.61)
common_isotopes['145Nd'] = SpinType('145Nd', 7 / 2, -0.89767, -0.314)

# Sm
common_isotopes['147Sm'] = SpinType('147Sm', 7 / 2, -1.111, -0.27)
common_isotopes['149Sm'] = SpinType('149Sm', 7 / 2, -0.91368, 0.075)

# Eu
common_isotopes['151Eu'] = SpinType('151Eu', 5 / 2, 6.650967, 0.95)
common_isotopes['153Eu'] = SpinType('153Eu', 5 / 2, 2.93572, 2.28)

# Gd
common_isotopes['155Gd'] = SpinType('155Gd', 3 / 2, -0.821225, 1.27)
common_isotopes['157Gd'] = SpinType('157Gd', 3 / 2, -1.0850, 1.35)

# Tb
common_isotopes['159Tb'] = SpinType('159Tb', 3 / 2, 6.34059, 1.432)

# Dy
common_isotopes['161Dy'] = SpinType('161Dy', 5 / 2, -0.919568, 2.51)
common_isotopes['163Dy'] = SpinType('163Dy', 5 / 2, 1.28931, 2.318)

# Ho
common_isotopes['165Ho'] = SpinType('165Ho', 7 / 2, 5.7062478, 3.58)

# Er
common_isotopes['167Er'] = SpinType('167Er', 7 / 2, -0.771575, 3.57)

# Tm
common_isotopes['169Tm'] = SpinType('169Tm', 1 / 2, -2.21, 0)

# Yb
common_isotopes['171Yb'] = SpinType('171Yb', 1 / 2, 4.7248, 0)
common_isotopes['173Yb'] = SpinType('173Yb', 5 / 2, -1.3025, 2.80)

# Lu
common_isotopes['175Lu'] = SpinType('175Lu', 7 / 2, 3.05469, 3.49)
common_isotopes['176Lu'] = SpinType('176Lu', 7, 2.163448, 4.92)

# Hf
common_isotopes['177Hf'] = SpinType('177Hf', 7 / 2, 1.081, 3.37)
common_isotopes['179Hf'] = SpinType('179Hf', 9 / 2, -0.679, 3.79)

# Ta
common_isotopes['181Ta'] = SpinType('181Ta', 7 / 2, 3.22, 3.17)

# W
common_isotopes['183W'] = SpinType('183W', 1 / 2, 1.12, 0)

# Re
common_isotopes['185Re'] = SpinType('185Re', 5 / 2, 6.077, 2.18)
common_isotopes['187Re'] = SpinType('187Re', 5 / 2, 6.138, 2.07)

# Os
common_isotopes['187Os'] = SpinType('187Os', 1 / 2, 0.616, 0)
common_isotopes['189Os'] = SpinType('189Os', 3 / 2, 0.8475, 0.86)

# Ir
common_isotopes['191Ir'] = SpinType('191Ir', 3 / 2, 0.4643, 0.816)
common_isotopes['193Ir'] = SpinType('193Ir', 3 / 2, 0.5054, 0.751)

# Pt
common_isotopes['195Pt'] = SpinType('195Pt', 1 / 2, 5.768, 0)

# Au
common_isotopes['197Au'] = SpinType('197Au', 3 / 2, 0.4625, 0.547)

# Hg
common_isotopes['199Hg'] = SpinType('199Hg', 1 / 2, 4.8154, 0)
common_isotopes['201Hg'] = SpinType('201Hg', 3 / 2, -1.7776, 0.387)

# Tl
common_isotopes['203Tl'] = SpinType('203Tl', 1 / 2, 15.436, 0)
common_isotopes['205Tl'] = SpinType('205Tl', 1 / 2, 15.589, 0)

# Pb
common_isotopes['207Pb'] = SpinType('207Pb', 1 / 2, 5.54, 0)

# Bi
common_isotopes['209Bi'] = SpinType('209Bi', 9 / 2, 4.342, -0.516)

# U
common_isotopes['235U'] = SpinType('235U', 7 / 2, -0.52, 4.936)
