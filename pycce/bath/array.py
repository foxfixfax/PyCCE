import warnings
from collections import UserDict, Mapping

import numpy as np
from numpy.lib.recfunctions import repack_fields

from ..constants import HBAR, ELECTRON_GYRO

HANDLED_FUNCTIONS = {}

_set_str_kinds = {'U', 'S'}


class BathArray(np.ndarray):
    """
    Subclass of ``ndarray`` containing information about the bath spins.

    The subclass has fixed structured datatype::

         _dtype_bath = np.dtype([('N', np.unicode_, 16),
                                 ('xyz', np.float64, (3,)),
                                 ('A', np.float64, (3, 3)),
                                 ('Q', np.float64, (3, 3))])

    Accessing different fields results in the ``ndarray`` view.

    Each of the fields can be accessed as the attribute of the ``BathArray`` instance and modified accordingly.
    In addition to the name fields, the information of the bath spin types is stored in the ``types`` attribute.
    All of the items in ``types`` can be accessed as attributes of the BathArray itself.

    Examples:
        Generate empty ``BathArray`` instance.

        >>> ba = BathArray((3,))
        >>> print(ba)
        [('', [0., 0., 0.], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
         ('', [0., 0., 0.], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
         ('', [0., 0., 0.], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])]

        Generate BathArray from the set of arrays:

        >>> import numpy as np
        >>> ca = np.random.random((3, 3))
        >>> sn = ['1H', '2H', '3H']
        >>> hf = np.random.random((3, 3, 3))
        >>> ba = BathArray(ca=ca, hf=hf, sn=sn)
        >>> print(ba.N, ba.types)
        ['1H' '2H' '3H'] SpinDict(1H: (1H, 0.5, 26.7519), 2H: (2H, 1, 4.1066, 0.00286), 3H: (3H, 0.5, 28.535))

    .. warning::
        Due to how structured arrays work, if one uses a boolean array to access an subarray,
        and then access the name field, the initial array *will not change*.

        Example:

            >>> ba = BathArray((10,), sn='1H')
            >>> print(ba.N)
            ['1H' '1H' '1H' '1H' '1H' '1H' '1H' '1H' '1H' '1H']
            >>> bool_mask = np.arange(10) % 2 == 0
            >>> ba[bool_mask]['N'] = 'e'
            >>> print(ba.N)
            ['1H' '1H' '1H' '1H' '1H' '1H' '1H' '1H' '1H' '1H']

            To achieve the desired result, one should first access the name field and only then apply the boolean mask:

            >>> ba['N'][bool_mask] = 'e'
            >>> print(ba.N)
            ['e' '1H' 'e' '1H' 'e' '1H' 'e' '1H' 'e' '1H']

    Args:
        shape (tuple): Shape of the array.

        array (array-like):
            Either an unstructured array with shape (n, 3) containing coordinates of bath spins as rows OR
            structured ndarray with the same fields as the datatype of the bath.

        name (array-like):
            Array of the bath spin name.

        hyperfines (array-like):
            Array of the hyperfine tensors with shape (n, 3, 3).

        quadrupoles (array-like):
            Array of the quadrupole tensors with shape (n, 3, 3).

        efg (array-like):
            Array of the electric field gradients with shape (n, 3, 3) for each bath spin.
            Used to compute Quadrupole tensors for spins >= 1.
            Requires the spin types either be found in ``common_isotopes`` or specified with ``types`` argument.

        types (SpinDict):
            SpinDict or input to create one.
            Contains either SpinTypes of the bath spins or tuples which will initialize those.
            See ``pycce.bath.SpinDict`` documentation for details.

        ca (array-like):
            Shorthand notation for ``array`` argument.

        sn (array-like):
            Shorthand notation for ``name`` argument.

        hf (array-like):
            Shorthand notation for ``hyperfines`` argument.

        q (array-like):
            Shorthand notation for ``quadrupoles`` argument.

    """
    _dtype_bath = np.dtype([('N', np.unicode_, 16),
                            ('xyz', np.float64, (3,)),
                            ('A', np.float64, (3, 3)),
                            ('Q', np.float64, (3, 3))])

    def __new__(subtype, shape=None, array=None,
                names=None, hyperfines=None, quadrupoles=None,
                ca=None, sn=None, hf=None, q=None, efg=None,
                types=None):

        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments. This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__

        if array is None and ca is not None:
            array = ca

        if names is None and sn is not None:
            names = sn

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

        if types is not None:
            try:
                obj.add_type(**types)
            except TypeError:
                obj.add_type(*types)

        if array is not None:
            array = np.asarray(array)
            if array.dtype.names is not None:
                for n in array.dtype.names:
                    obj[n] = array[n]
            else:
                obj['xyz'] = array.reshape(-1, 3)

        if names is not None:
            obj['N'] = np.asarray(names).reshape(-1)
        if hyperfines is not None:
            obj['A'] = np.asarray(hyperfines).reshape(-1, 3, 3)
        if quadrupoles is not None:
            obj['Q'] = np.asarray(quadrupoles).reshape(-1, 3, 3)
        elif efg is not None:
            obj.from_efg(efg)
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
        if obj.dtype != self._dtype_bath:
            warnings.warn('Trying to view array with unknown dtype as BathArray. '
                          'This can lead to unexpected results.',
                          RuntimeWarning, stacklevel=2)

        self.types = getattr(obj, 'types', SpinDict())

        # We do not need to return anything

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            if not all(issubclass(t, np.ndarray) for t in types):
                # Defer to any non-subclasses that implement __array_function__
                return NotImplemented
            # Use NumPy's private implementation without __array_function__
            # dispatching
            return func._implementation(*args, **kwargs)
        # Note: this allows subclasses that don't override
        # __array_function__ to handle MyArray objects
        if not all(issubclass(t, BathArray) for t in types):
            return NotImplemented

        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    @property
    def name(self):
        """
        ndarray: Array of the ``name`` attribute for each spin in the array from ``types`` dictionary.
            Note that while the output of attribute should be the same as the ``N`` field of the BathArray instance,
            this attribute *should not* be used for production as it creates a *new* array from ``types`` dictionary.
        """
        return self.types[self].name

    @property
    def s(self):
        """
        ndarray: Array of the ``spin`` (spin value) attribute for each spin in the array from ``types`` dictionary.
        """
        return self.types[self].s

    @property
    def gyro(self):
        """
        ndarray: Array of the ``gyro`` (gyromagnetic ratio)
            attribute for each spin in the array from ``types`` dictionary.
        """
        return self.types[self].gyro

    @property
    def q(self):
        """
        ndarray: Array of the ``q`` (quadrupole moment)
            attribute for each spin in the array from ``types`` dictionary.
        """
        return self.types[self].q

    @property
    def detuning(self):
        """
        ndarray: Array of the ``detuning``
            attribute for each spin in the array from ``types`` dictionary.
        """
        return self.types[self].detuning

    @property
    def x(self):
        """
        ndarray: Array of x coordinates for each spin in the array (``bath['xyz'][:, 0]``).
        """
        return self['xyz'][:, 0]

    @x.setter
    def x(self, val):
        self['xyz'][:, 0] = val

    @property
    def y(self):
        """
        ndarray: Array of y coordinates for each spin in the array (``bath['xyz'][:, 1]``).
        """
        return self['xyz'][:, 1]

    @y.setter
    def y(self, val):
        self['xyz'][:, 1] = val

    @property
    def z(self):
        """
        ndarray: Array of z coordinates for each spin in the array (``bath['xyz'][:, 2]``).
        """
        return self['xyz'][:, 2]

    @z.setter
    def z(self, val):
        self['xyz'][:, 2] = val

    @property
    def N(self):
        """
        ndarray: Array of name for each spin in the array (``bath['N']``).
        """
        return self['N']

    @N.setter
    def N(self, val):
        self['N'] = val

    @property
    def xyz(self):
        """
        ndarray: Array of coordinates for each spin in the array (``bath['xyz']``).
        """
        return self['xyz']

    @xyz.setter
    def xyz(self, val):
        self['xyz'] = val

    @property
    def A(self):
        """
        ndarray: Array of hyperfine tensors for each spin in the array (``bath['A']``).
        """
        return self['A']

    @A.setter
    def A(self, val):
        self['A'] = val

    @property
    def Q(self):
        """
        ndarray: Array of quadrupole tensors for each spin in the array (``bath['Q']``).
        """
        return self['Q']

    @Q.setter
    def Q(self, val):
        self['Q'] = val

    def __getitem__(self, item):
        # if string then return ndarray view of the field
        if isinstance(item, int):
            return super().__getitem__((Ellipsis, item))
        elif isinstance(item, (str, np.str_)):
            try:
                return self.view(np.ndarray).__getitem__(item)
            except ValueError:
                return self[self['N'] == item]
        else:
            return super().__getitem__(item)

    def __setitem__(self, key, val):
        np.ndarray.__setitem__(self, key, val)

        if isinstance(val, str):
            if val not in self.types.keys():
                try:
                    self.types[val] = common_isotopes[val]
                except KeyError:
                    warnings.warn(_spin_not_found_message(val), stacklevel=2)
            return

        elif isinstance(val, (int, float, complex)):
            return

        val = np.asarray(val)

        if (val.dtype.names is not None) and ('N' in val.dtype.names):
            val = val['N']

        if val.dtype.kind in _set_str_kinds:
            if not val.shape:
                n = val[()]
                if n not in self.types.keys():
                    try:
                        self.types[n] = common_isotopes[n]
                    except KeyError:
                        warnings.warn(_spin_not_found_message(n), stacklevel=2)
                return

            for n in np.unique(val):
                if n not in self.types.keys():
                    try:
                        self.types[n] = common_isotopes[n]
                    except KeyError:
                        warnings.warn(_spin_not_found_message(n), stacklevel=2)
            return

    def __eq__(self, other):
        try:

            xyzs = (self['xyz'] == other['xyz']).all(axis=1)
            hfs = (self['A'] == other['A']).all(axis=(1, 2))
            qds = (self['Q'] == other['Q']).all(axis=(1, 2))

            return xyzs & hfs & qds

        except (IndexError, TypeError):
            return super().__eq__(other)

    def add_type(self, *args, **kwargs):
        """
        Add spin type to the ``types`` dictionary.

        Args:
            *args: Any number of positional inputs to create ``SpinDict`` entries.
                E.g. the tuples of form (name :obj:`str`, spin :obj:`float`, gyro :obj:`float`, q :obj:`float`).
            **kwargs: Any number of keyword inputs to create ``SpinDict`` entries.
                E.g. name = (spin, gyro, q).

        For details and allowed inputs see ``SpinDict`` documentation.

        Returns:
            SpinDict: A view of ``self.types`` instance.
        """
        self.types.add_type(*args, **kwargs)

    def update(self, ext_bath, error_range=0.2, ignore_isotopes=True, inplace=True):
        """
        Update the properties of the spins in the array using data from other ``BathArray`` instance.
        For each spin in ``ext_bath`` check whether there is such spin in the array that has the same position
        within allowed error range given by ``error_range`` and has the same name.
        If such spins is found in the array, then it's coordinates, hyperfine tensor and quadrupole tensor
        are updated using the values of the spin in the ``ext_bath`` object.

        If ``ignore_isotopes`` is true, then the name check ignores numbers in the name of the spins.

        Args:
            ext_bath (BathArray): Array of the new spins.
            error_range (float): +- distance in Angstrom within which two positions are considered to be the same.
                Default is 0.2 A.
            ignore_isotopes (bool): True if ignore numbers in the name of the spins. Default True.
            inplace (bool): True if changes parameters of the array in place. If False, returns copy of the array.

        Returns:
            BathArray: updated BathArray instance.
        """
        bath = update_bath(self, ext_bath, error_range, ignore_isotopes, inplace)
        return bath

    def from_point_dipole(self, position, gyro_e=ELECTRON_GYRO, inplace=True):
        """
        Generate hyperfine couplings, assuming that bath spins interaction with central spin is the same as the
        one between two magnetic point dipoles.

        Args:
            position (ndarray with shape (3,)): position of the central spin
            gyro_e (obj:`float` or obj:`ndarray` with shape (3,3)):
                gyromagnetic ratio of the central spin *OR*
                tensor corresponding to interaction between magnetic field and central spin.
            inplace (bool): True if changes parameters of the array in place. If False, returns copy of the array.

        Returns:
            BathArray: updated BathArray instance with changed hyperfine couplings.
        """

        if inplace:
            array = self
        else:
            array = self.copy()

        identity = np.eye(3, dtype=np.float64)
        pos = array['xyz'] - position

        posxpos = np.einsum('ki,kj->kij', pos, pos)

        r = np.linalg.norm(pos, axis=1)[:, np.newaxis, np.newaxis]

        if isinstance(gyro_e, (np.floating, float, int)):
            pref = (gyro_e * array.gyro * HBAR)[:, np.newaxis, np.newaxis]

            array['A'] = -(3 * posxpos[np.newaxis, :] - identity[np.newaxis, :] * r ** 2) / (r ** 5) * pref

        else:
            pref = (gyro_e[np.newaxis, :, :] * array.gyro[:, np.newaxis, np.newaxis] * HBAR)
            postf = -(3 * posxpos[np.newaxis, :] - identity[np.newaxis, :] * r ** 2) / (r ** 5)
            np.matmul(pref, postf, out=array['A'])

        return array

    def from_cube(self, cube, gyro_e=ELECTRON_GYRO, inplace=True):
        """
        Generate hyperfine couplings, assuming that bath spins interaction with central spin can be approximated as
        a point dipole, interacting with given spin density distribution.

        Args:
            cube (Cube): An instance of `Cube` object, which contains spatial distribution of spin density.
                For details see documentation of `Cube` class.
            gyro_e (float): Gyromagnetic ratio of the central spin.
            inplace (bool): True if changes parameters of the array in place. If False, returns copy of the array.

        Returns:
            BathArray: Updated BathArray instance with changed hyperfine couplings.
        """
        if inplace:
            array = self
        else:
            array = self.copy()

        gyros = array.types[array].gyro
        array['A'] = cube.integrate(array['xyz'], gyros, gyro_e)
        return array

    def from_func(self, func, gyro_e=ELECTRON_GYRO, vectorized=False, inplace=True):
        """
        Generate hyperfine couplings from user-defined function.

        Args:

            func (func): Callable with signature
                ``func(coord, gyro, central_gyro)``, where ``coord`` is array of the bath spin coordinate,
                ``gyro`` is the gyromagnetic ratio of bath spin,
                ``central_gyro`` is the gyromagnetic ratio of the central bath spin.

            gyro_e (float): gyromagnetic ratio of the central spin to be used in the function.

            vectorized (bool): If True, assume that func takes arrays of all bath spin coordinates and array of
                gyromagnetic ratios as arguments.

            inplace (bool): True if changes parameters of the array in place. If False, returns copy of the array.

        Returns:
            BathArray: Updated BathArray instance with changed hyperfine couplings.

        """
        if inplace:
            array = self
        else:
            array = self.copy()

        if vectorized:
            array['A'] = func(array['xyz'], array.gyro, gyro_e)
        else:
            for a in array:
                a['A'] = func(a['xyz'], array.types[a].gyro, gyro_e)
        return array

    def from_efg(self, efg, inplace=True):
        """
        Generate quadrupole splittings from electric field gradient tensors for spins >= 1.

        Args:
            efg (array-like): Array of the electric field gradients for each bath spin. The data for spins-1/2
            should be included but can be any value.
            inplace (bool): True if changes parameters of the array in place. If False, returns copy of the array.

        Returns:
            BathArray: Updated BathArray instance with changed quadrupole tensors.

        """

        if inplace:
            array = self
        else:
            array = self.copy()

        efg = np.asarray(efg).reshape(-1, 3, 3)

        spins = array.s
        qmoments = array.q

        where = spins > 0.5
        pref = np.zeros(spins.shape, dtype=np.float64)
        pref[where] = qmoments[where] / (2 * spins[where] * (2 * spins[where] - 1))

        array['Q'] = pref[:, np.newaxis, np.newaxis] * efg
        return array

    def dist(self, pos=None):
        """
        Compute the distance of the bath spins from the given position.

        Args:
            pos (ndarray with shape (3,)): Cartesian coordinates of the position from which to compute the distance.
                Default is (0, 0, 0).
        Returns:
            ndarray  with shape (n,): Array of distances of each bath spin from the given position in angstrom.
        """
        if pos is None:
            pos = np.zeros(3)
        else:
            pos = np.asarray(pos)

        return np.linalg.norm(self['xyz'] - pos, axis=-1)

    def savetxt(self, filename, fmt='%18.8f', strip_isotopes=False, **kwargs):
        """
        Save name of the isotopes and their coordinates to the txt file of xyz format.

        Args:
            filename (str or file): Filename or file handle.
            fmt (str): Format of the coordinate entry,
            strip_isotopes (bool): True if remove numbers from the name of bath spins. Default False.
            **kwargs: Additional keywords of the ``numpy.savetxt`` function.
        """
        kwargs.setdefault('comments', '')
        ar = repack_fields(self[['N', 'xyz']]).view(np.ndarray).view(
            np.dtype([('N', 'U16'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8')]))

        if strip_isotopes:
            ar['N'] = np.core.defchararray.strip(ar['N'], '1234567890')

        header = f'{ar.size}\n'
        for n, c in zip(*np.unique(ar['N'], return_counts=True)):
            header += f'{n}{c} '

        kwargs.setdefault('header', header)
        np.savetxt(filename, ar, fmt=('%s', fmt, fmt, fmt), **kwargs)


def implements(numpy_function):
    """Register an __array_function__ implementation for BathArray objects."""

    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


@implements(np.concatenate)
def concatenate(arrays, axis=0, out=None):
    """
    Join a sequence of instances of ``BathArray`` along an existing axis. Overriders ``numpy.concatenate`` function.

    Args:
        arrays (list of BathArray): Arrays to concatenate.
            The bath must have the same shape, except in the dimension corresponding to axis (the first, by default).
        axis (int): ``axis`` argument of ``numpy.concatenate``. The axis along which the arrays will be joined.
            If axis is None, arrays are flattened before use. Default is 0.
        out (BathArray): ``out`` argument of ``numpy.concatenate``. If provided, the destination to place the result.

    Returns:
       BathArray: Concatenated array.
    """
    new_array = np.concatenate([x.view(np.ndarray) for x in arrays], axis=axis, out=out)
    new_array = new_array.view(BathArray)
    types = SpinDict()
    for x in arrays:
        types += x.types
    new_array.types = types
    return new_array


# @implements(np.broadcast_to)
# def broadcast_to(array, shape):
#     ...  # implementation of broadcast_to for MyArray objects


def same_bath_indexes(barray_1, barray_2, error_range=0.2, ignore_isotopes=True):
    """
    Find indexes of the same array elements in two ``BathArray`` instances.

    Args:
        barray_1 (BathArray): First array.
        barray_2 (BathArray): Second array.
        error_range (float): If distance between positions in two arrays is smaller than ``error_range``
            they are assumed to be the same.
        ignore_isotopes (bool): True if ignore numbers in the name of the spins. Default True.

    Returns:
        tuple: tuple containing:

            * **ndarray**: Indexes of the elements in the first array found in the second.
            * **ndarray**: Indexes of the elements in the second array found in the first.

    """
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


def update_bath(total_bath, added_bath, error_range=0.2, ignore_isotopes=True, inplace=True):
    """
    Update the ``BathArray`` with elements from the second array.

    Args:

        total_bath (BathArray): Array which contains elements that will be substituted.
        added_bath (BathArray): Array with elements that will be used to substitute elements in ``total_bath``.

        error_range (float): If distance between positions in two arrays is smaller than ``error_range``
            they are assumed to be the same.

        ignore_isotopes (bool): True if ignore numbers in the name of the spins. Default True.

        inplace (bool): True if changes parameters of the ``total_bath`` array in place.
            If False, returns copy of the array.

    Returns:
        BathArray: updated BathArray instance.
    """
    if not inplace:
        total_bath = total_bath.copy()

    indexes, ext_indexes = same_bath_indexes(total_bath, added_bath, error_range, ignore_isotopes)
    for n in added_bath.dtype.names:
        if ignore_isotopes and n == 'N':
            continue
        total_bath[n][indexes] = added_bath[n][ext_indexes]

    return total_bath


class SpinType:
    """
    Class which contains properties of each spin type in the bath.

    Args:
        name (str): Name of the bath spin.
        s (float): Total spin of the bath spin. Default 0.
        gyro (float): Gyromagnetic ratio in rad/(ms * G). Default 0.
        q (float): Quadrupole moment in millibarn (for s > 1/2). Default 0.
        detuning (float): Energy detuning from the zeeman splitting in rad/(ms). Default 0.

    Attributes:

        name (str): Name of the bath spin.
        s (float): Total spin of the bath spin.
        gyro (float): Gyromagnetic ratio in rad/(ms * G).
        q (float): Quadrupole moment in millibarn (for s > 1/2).
        detuning (float): Energy detuning from the zeeman splitting in rad/(ms).

    """

    def __init__(self, name, s=0., gyro=0., q=0., detuning=0.):

        self.name = name
        self.s = s
        self.gyro = gyro
        self.q = q
        self.detuning = detuning

    def __eq__(self, obj):
        if not isinstance(obj, SpinType):
            return False

        checks = (self.name == obj.name) & (self.s == obj.s) & (
                self.gyro == obj.gyro) & (self.q == obj.q) & (self.detuning == obj.detuning)

        return checks

    def __repr__(self):
        base_message = f'({self.name}, {self.s}, {self.gyro}'

        if np.asarray(self.q).any():
            base_message += f', {self.q}'

        if np.asarray(self.detuning).any():
            base_message += f', {self.detuning}'

        base_message += ')'

        return base_message


class SpinDict(UserDict):
    """
    Wrapper class for dictionary tailored for containing properties of the spin types.
    Can take ``np.void`` or ``BathArray`` instances as keys.
    Every entry is instance of the ``SpinType``.

    Each entry of the ``SpinDict`` can be initianlized as follows:

        * As a Tuple containing name (optional), spin, gyromagnetic ratio, quadrupole constant (optional)
          and detuning (optional).
        * As a ``SpinType`` instance.

    Examples:

        >>> types = SpinDict()
        >>> types['1H'] = ('1H', 1 / 2, 26.7519)
        >>> types['2H'] = 1, 4.1066, 0.00286
        >>> types['3H'] = SpinType('3H', 1 / 2, 28.535, 0)
        >>> print(types)
        SpinDict({'1H': (1H, 0.5, 26.7519, 0.0), '2H': (2H, 1, 4.1066, 0.00286), '3H': (3H, 0.5, 28.535, 0)})

    If ``SpinType`` of the given bath spin is not provided, when requested
    ``SpinDict`` will try to find information about the bath spins in the ``common_isotopes``.

    If found, adds an entry to the given ``SpinDict`` instance and
    returns it. Otherwise ``KeyError`` is raised.

    To initiallize several ``SpinType`` entries one can use ``add_types`` method.
    """

    def __delitem__(self, key):
        try:
            super().__delitem__(key)
        except TypeError:
            if key.shape:
                try:
                    names = key['N']
                except IndexError:
                    names = key
                for k in names:
                    super().__delitem__(k)
                return

            k = key[()]
            try:
                k = k['N']
            except TypeError:
                pass
            super().__delitem__(k)

    def __setitem__(self, key, value):
        try:
            value = _check_key_spintype(key, value)
            super().__setitem__(key, value)
        except (TypeError, ValueError):
            if key.shape:
                try:
                    names = key['N']
                except IndexError:
                    names = key
                for k, v in zip(names, value):
                    if not isinstance(v, SpinType):
                        if v[0] == k:
                            v = SpinType(*v)
                        else:
                            v = SpinType(k, *v)
                    super().__setitem__(k, v)
                    return

            k = key[()]
            try:
                k = k['N']
            except TypeError:
                pass

            if value[0] == k:
                value = SpinType(*value)
            else:
                value = SpinType(k, *value)

            super().__setitem__(k, value)

    def __getitem__(self, key):

        try:
            key = key[()]
            return self._super_get_item(key['N'])
            # self._super_get_item(key)
        except TypeError:
            try:
                return self._super_get_item(key)
            except TypeError:

                if key.dtype.names:
                    key = key['N']

                unique_names = np.unique(key)

                if unique_names.size == 1:
                    n = unique_names[0]
                    ones = np.ones(key.shape, dtype=np.float64)

                    spins = self._super_get_item(n).s * ones
                    gyros = self._super_get_item(n).gyro * ones
                    quads = self._super_get_item(n).q * ones
                    detus = self._super_get_item(n).detuning * ones

                else:
                    spins = np.empty(key.shape, dtype=np.float64)
                    gyros = np.empty(key.shape, dtype=np.float64)
                    quads = np.empty(key.shape, dtype=np.float64)
                    detus = np.empty(key.shape, dtype=np.float64)

                    for n in unique_names:

                        spins[key == n] = self._super_get_item(n).s
                        gyros[key == n] = self._super_get_item(n).gyro
                        quads[key == n] = self._super_get_item(n).q
                        detus[key == n] = self._super_get_item(n).detuning

                return SpinType(key, s=spins, gyro=gyros, q=quads, detuning=detus)

                # params = {}
                # unique_names = np.unique(key)
                # first_name = unique_names[0]
                # first_dict = vars(self._super_get_item(first_name))
                #
                # if unique_names.size == 1:
                #     for k in first_dict:
                #         params[k] = np.array([first_dict[k]]*key.size).reshape(key.shape)
                #     return SpinType(**params)
                #
                # for k in first_dict:
                #     params[k] = np.empty(key.shape, dtype=type(first_dict[k]))
                #     params[k][key == first_name] = getattr(self._super_get_item(first_name), k)
                #
                # for n in unique_names[1:]:
                #     for k in params:
                #         params[k][key == n] = getattr(self._super_get_item(n), k)
                # return SpinType(**params)
    # adding two objects
    def __add__(self, obj):
        new_obj = SpinDict()
        keys_1 = list(self.keys())
        keys_2 = list(obj.keys())

        for k in {*keys_1, *keys_2}:

            if (k in keys_1) and (k in keys_2):
                assert obj[k] == self[k], f'Error, type {k} has different properties in provided types'
                new_obj[k] = self[k]
            elif k in keys_1:
                new_obj[k] = self[k]
            else:
                new_obj[k] = obj[k]
        return new_obj

    def __repr__(self):
        message = f"{type(self).__name__}("
        for k in self.data:
            message += f"{k}: {self.data[k]}, "
            if len(message) > 150:
                message += '..., '
                break

        message = message[:-2] + ')'

        return message

    def _super_get_item(self, n):
        try:
            return super().__getitem__(n)
        except KeyError:
            if not n in common_isotopes:
                raise KeyError(_spin_not_found_message(n))
            super().__setitem__(n, common_isotopes[n])
            return super().__getitem__(n)

    def add_type(self, *args, **kwargs):
        """
        Add one or several spin types to the spin dictionary.

        Args:
            *args:
                any numbers of arguments which could initialize ``SpinType`` instances.
                Accepted arguments:

                    * Tuple containing name, spin, gyromagnetic ratio, quadrupole constant (optional)
                      and detuning (optional).
                    * ``SpinType`` instance.

                Can also initialize one instance of ``SpinType`` if each argument corresponds to
                each positional argument necessary to initiallize.

            **kwargs: any numbers of keyword arguments which could initialize ``SpinType`` instances.
                Usefull as an alternative for updating the dictionary. for each keyword argument adds an entry
                to the ``SpinDict`` with the same name as keyword.

        Examples:
            >>> types = SpinDict()
            >>> types.add_type('1H', 1 / 2, 26.7519)
            >>> types.add_type(('1H_det', 1 / 2, 26.7519, 10), ('2H', 1, 4.1066, 0.00286),
            >>>                 SpinType('3H', 1 / 2, 28.535, 0), e=(1 / 2, 6.7283, 0))
            >>> print(types)
            SpinDict(1H: (1H, 0.5, 26.7519), 1H_det: (1H_det, 0.5, 26.7519, 10),
            2H: (2H, 1, 4.1066, 0.00286), 3H: (3H, 0.5, 28.535), e: (e, 0.5, 6.7283))

        """
        try:
            keys = []
            for nuc in args:
                if isinstance(nuc, SpinType):
                    key = nuc.name
                    self[key] = nuc
                    keys.append(key)
                elif isinstance(nuc, Mapping):
                    self.update(nuc)
                    keys += list(nuc.keys())
                else:
                    key = nuc[0]
                    self[key] = SpinType(*nuc)
                    keys.append(key)
        except TypeError:
            for k in keys:
                self.pop(k)
            self[args[0]] = SpinType(*args)
        for nuc in kwargs:
            self[nuc] = kwargs[nuc]


def _check_key_spintype(k, v):
    if isinstance(v, SpinType):
        return v

    if v[0] == k:
        v = SpinType(*v)
    else:
        v = SpinType(k, *v)
    return v


_remove_digits = str.maketrans('', '', '+-^1234567890')
import re


def random_bath(name, size, number=1000, density=None, types=None,
                density_units='cm-3', center=None,
                seed=None):
    """
    Generate random bath containing spins with names provided with argument ``name`` in the box of size ``size``.
    By default generates coordinates in range (-size/2; +size/2) but this behavior can be changed by providing
    ``center`` keyword.

    Args:
        name (str or array-like with length n): Name of the bath spin or array with the names of the bath spins,
        size (float or ndarray with shape (3,)): Size of the box. If float is given,
            assumes 3D cube with the edge = ``size``. Otherwise the size specifies the dimensions of the box.
            Dimensionality is controlled by setting entries of the size array to 0.

        number (int or array-like with length n): Number of the bath spins in the box
            or array with the numbers of the bath spins. Has to have the same length as the ``name`` array.

        density (float or array-like with length n): Concentration of the bath spin
            or array with the concentrations. Has to have the same length as the ``name`` array.

        types (SpinDict): Dictionary with SpinTypes or input to create one

        density_units (str): If number of spins provided as density, defines units.
            Values are accepted in the format ``m``, or ``m^x`` or ``m-x`` where m is the length unit,
            x is dimensionality of the bath (e.g. x=1 for 1D, 2 for 2D etc).
            If only ``m`` is provided the dimensions are inferred from ``size`` argument.
            Accepted length units:

                * ``m`` meters;
                * ``cm`` centimeters;
                * ``a`` angstroms;

        center (ndarray with shape (3,)): Coordinates of the (0, 0, 0) point of the final coordinate system
            in the initial coordinates. Default is ``size / 2`` - center is in the middle of the box.

    Returns:
        BathArray with shape (np.prod(number)): Array of the bath spins with random positions.
    """
    size = np.asarray(size)
    unit_conversion = {'a': 1, 'cm': 1e-8, 'm': 1e-10}
    name = np.asarray(name)

    if size.size == 1:
        size = np.array([size, size, size])

    elif size.size > 3:
        raise RuntimeError('Wrong size format')

    if center is None:
        center = size / 2

    if density is not None:
        du = density_units.lower().translate(_remove_digits)
        power = re.findall(r'\d+', density_units.lower())

        sc = np.count_nonzero(size != 0)

        if power:
            powa = int(power[0])
            if sc != powa:
                warnings.warn(f'size dimensions {sc} do not agree with density units {density_units}',
                              stacklevel=2)
        else:
            powa = sc

        density = np.asarray(density) * unit_conversion[du] ** powa

        number = np.rint(density * np.prod(size[size != 0])).astype(np.int32)

    else:
        number = np.asarray(number, dtype=np.int32)

    total_number = np.sum(number)
    spins = BathArray((total_number,), types=types)

    if name.shape:
        counter = 0
        for n, no in zip(name, number):
            spins.N[counter:counter + no] = n
            counter += no

    else:
        spins.N = name

    # Generate the coordinates
    generator = np.random.default_rng(seed=seed)

    spins.xyz = generator.random(spins.xyz.shape) * size - center
    return spins


_spin_not_found_message = lambda x: 'Spin type for {} was not provided and was not found in common isotopes.'.format(x)

# Dictionary of the common isotopes. Placed in this file to avoid circular dependency
common_isotopes = SpinDict()
"""
SpinDict: An instance of the ``SpinDict`` dictionary, containing properties for the most of the common isotopes with 
nonzero spin.
The isotope is considered common if it is stable and has nonzero concentration in nature.  
"""
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

# mfield
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

# S
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

# Pd
common_isotopes['105Pd'] = SpinType('105Pd', 5 / 2, -1.2305, 0.66)

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
common_isotopes['138La'] = SpinType('138La', 5, 3.5575, 0.21)
common_isotopes['139La'] = SpinType('139La', 7 / 2, 3.8085, 0.200)

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
