import copy
import warnings
from collections import UserDict, Mapping

import numpy as np
from numpy.lib.recfunctions import repack_fields

from .map import InteractionMap
from ..constants import HBAR, ELECTRON_GYRO, HBAR_SI, NUCLEAR_MAGNETON, PI2

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

        imap (InteractionMap):
            Instance of InteractionMap containing user defined
            interaction tensors between bath spins stored in the array.

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
                types=None, imap=None,
                ca=None, sn=None, hf=None, q=None, efg=None):

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
        obj.imap = imap

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
        self.imap = getattr(obj, 'imap', None)

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

    def sort(self, axis=-1, kind=None, order=None):
        """
        Sort array in-place. Is implemented only when imap is None. Otherwise use ``np.sort``.
        """
        if self.imap is None:
            super().sort(axis=axis, kind=kind, order=order)
        else:
            raise NotImplementedError('Inplace sort is implemented only when .imap is None')

    @property
    def name(self):
        """
        ndarray: Array of the ``name`` attribute for each spin in the array from ``types`` dictionary.

        .. note::
            While the value of this attribute should be the same as the ``N`` field of the BathArray instance,
            ``.name`` *should not* be used for production as it creates a *new* array from ``types`` dictionary.
        """
        return self.types[self].name

    @name.setter
    def name(self, initial_value):
        _set_sd_attribute(self, 'name', initial_value)

    @property
    def s(self):
        """
        ndarray: Array of the ``spin`` (spin value) attribute for each spin in the array from ``types`` dictionary.
        """
        return self.types[self].s

    @s.setter
    def s(self, initial_value):
        _set_sd_attribute(self, 's', initial_value)

    @property
    def dim(self):
        """
        ndarray: Array of the ``dim`` (dimensions of the spin) attribute
            for each spin in the array from ``types`` dictionary.
        """
        return self.types[self].dim

    @property
    def gyro(self):
        """
        ndarray: Array of the ``gyro`` (gyromagnetic ratio)
            attribute for each spin in the array from ``types`` dictionary.
        """
        return self.types[self].gyro

    @gyro.setter
    def gyro(self, initial_value):
        _set_sd_attribute(self, 'gyro', initial_value)

    @property
    def q(self):
        """
        ndarray: Array of the ``q`` (quadrupole moment)
            attribute for each spin in the array from ``types`` dictionary.
        """
        return self.types[self].q

    @q.setter
    def q(self, initial_value):
        _set_sd_attribute(self, 'q', initial_value)

    @property
    def detuning(self):
        """
        ndarray: Array of the ``detuning``
            attribute for each spin in the array from ``types`` dictionary.
        """
        return self.types[self].detuning

    @detuning.setter
    def detuning(self, initial_value):
        _set_sd_attribute(self, 'detuning', initial_value)

    @property
    def x(self):
        """
        ndarray: Array of x coordinates for each spin in the array (``bath['xyz'][:, 0]``).
        """
        return self['xyz'][..., 0]

    @x.setter
    def x(self, val):
        self['xyz'][..., 0] = val

    @property
    def y(self):
        """
        ndarray: Array of y coordinates for each spin in the array (``bath['xyz'][:, 1]``).
        """
        return self['xyz'][..., 1]

    @y.setter
    def y(self, val):
        self['xyz'][..., 1] = val

    @property
    def z(self):
        """
        ndarray: Array of z coordinates for each spin in the array (``bath['xyz'][:, 2]``).
        """
        return self['xyz'][..., 2]

    @z.setter
    def z(self, val):
        self['xyz'][..., 2] = val

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
        if isinstance(item, (int, np.int64)):
            return super().__getitem__((Ellipsis, item))
        elif isinstance(item, (str, np.str_)):
            try:
                return self.view(np.ndarray).__getitem__(item)
            except ValueError:
                return self[self['N'] == item]
        else:

            obj = super().__getitem__(item)
            if self.imap is not None:
                if not isinstance(item, tuple):

                    if isinstance(item, slice):
                        item = np.arange(self.size)[item]
                    smap = self.imap.subspace(item)
                    if smap:
                        obj.imap = smap
                    else:
                        obj.imap = None
            return obj

    def __setitem__(self, key, val):
        np.ndarray.__setitem__(self, key, val)

        if isinstance(val, str):
            if val not in self.types.keys():
                try:
                    self.types[val] = copy.copy(common_isotopes[val])
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
                        self.types[n] = copy.copy(common_isotopes[n])
                    except KeyError:
                        warnings.warn(_spin_not_found_message(n), stacklevel=2)
                return

            for n in np.unique(val):
                if n not in self.types.keys():
                    try:
                        self.types[n] = copy.copy(common_isotopes[n])
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

    def add_interaction(self, i, j, tensor):
        """
        Add interactions tensor between bath spins with indexes ``i`` and ``j``.

        .. note::

            If called from the subarray this method **does not** change the tensors of the total BathArray.

        Args:
            i (int or ndarray (n,) ):
                Index of the first spin in the pair or array of the indexes of the first spins in n pairs.
            j (int or ndarray with shape (n,)):
                Index of the second spin in the pair or array of the indexes of the second spins in n pairs.
            tensor (ndarray with shape (3,3) or (n, 3,3)):
                Interaction tensor between the spins i and j or array of tensors.

        """
        if self.imap is None:
            self.imap = InteractionMap(i, j, tensor)
            # self.imap[i, j] = tensor
        else:
            self.imap[i, j] = tensor

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

    def transform(self, center=None, cell=None, rotation_matrix=None, style='col', inplace=True):
        """
        Coordinate transformation of BathArray.

        Args:
            center (ndarray with shape (3,)): (0, 0, 0) position of new coordinates in the initial frame.

            cell (ndarray with shape (3, 3)): Cell vectors in cartesian coordinates
                if initial coordinates of the ``atoms`` are in crystallographic reference frame.

            rotation_matrix (ndarray with shape (3, 3)): Rotation matrix R of the **coordinate system**.

                E.g. ``R @ [0, 0, 1] = [a, b, c]`` where ``[a, b, c]`` are
                coordinates of the z axis of the new coordinate
                system in the old coordinate system.

                Note, that rotation is applied after transition from cell coordinates to the cartesian coordinates,
                in which cell vectors are stored.

            style (str): Can have two values: 'col' or 'row'.
                Shows how ``cell`` and ``rotation_matrix`` matrices are given:

                    * if 'col', each column of the matrix is a vector in previous coordinates;
                    * if 'row' - each row is a new vector.

                Default is 'col'.

            inplace (bool): If true, makes inplace changes to the provided array.

        Returns:
            BathArray: Transformed array with bath spins.
        """

        bath = transform(self, center=center, cell=cell, rotation_matrix=rotation_matrix,
                         style=style, inplace=inplace)
        return bath

    def from_point_dipole(self, position, gyro_e=ELECTRON_GYRO, inplace=True):
        """
        Generate hyperfine couplings, assuming that bath spins interaction with central spin is the same as the
        one between two magnetic point dipoles.

        Args:
            position (ndarray with shape (3,)): position of the central spin

            gyro_e (float or ndarray with shape (3,3)):
                gyromagnetic ratio of the central spin

                **OR**

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
        try:
            posxpos = np.einsum('ki,kj->kij', pos, pos)
        except ValueError:
            posxpos = np.tensordot(pos, pos, axes=0)

        r = np.linalg.norm(pos, axis=-1)[..., np.newaxis, np.newaxis]

        if isinstance(gyro_e, (np.floating, float, int)):
            pref = np.asarray(gyro_e * array.gyro * HBAR / PI2)[..., np.newaxis, np.newaxis]

            array['A'] = -(3 * posxpos[np.newaxis, ...] - identity[np.newaxis, ...] * r ** 2) / (r ** 5) * pref

        else:
            pref = (gyro_e[np.newaxis, :, :] * np.asarray(array.gyro)[..., np.newaxis, np.newaxis] * HBAR / PI2)
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

            func (func):
                Callable with signature::

                    func(coord, gyro, central_gyro)

                where ``coord`` is array of the bath spin coordinate,
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

        efg = np.asarray(efg).reshape((-1, 3, 3))

        spins = np.asarray(array.s)
        qmoments = np.asarray(array.q)

        where = spins > 0.5
        if not where.shape:
            if where:
                pref = qmoments / (2 * spins * (2 * spins - 1))
            else:
                pref = 0
        else:
            pref = np.zeros(efg.shape[0], dtype=np.float64)
            pref[where] = qmoments[where] / (2 * spins[where] * (2 * spins[where] - 1))
            pref = pref[..., np.newaxis, np.newaxis]

        array['Q'] = pref * efg
        return array

    def dist(self, position=None):
        """
        Compute the distance of the bath spins from the given position.

        Args:
            position (ndarray with shape (3,)):
                Cartesian coordinates of the position from which to compute the distance. Default is (0, 0, 0).

        Returns:
            ndarray  with shape (n,): Array of distances of each bath spin from the given position in angstrom.
        """
        if position is None:
            position = np.zeros(3)
        else:
            position = np.asarray(position)

        return np.linalg.norm(self['xyz'] - position, axis=-1)

    def savetxt(self, filename, fmt='%18.8f', strip_isotopes=False, **kwargs):
        """
        Save name of the isotopes and their coordinates to the txt file of xyz format.

        Args:
            filename (str or file): Filename or file handle.
            fmt (str): Format of the coordinate entry.
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


@implements(np.sort)
def sort(a, axis=-1, kind=None, order=None):
    """
    Return a sorted copy of an array. Overrides numpy.sort function.
    """
    indexes = np.argsort(a, axis=axis, kind=kind, order=order)

    return a[indexes]


@implements(np.argsort)
def sort(a, *args, **kwargs):
    """
    Return a indexes of an sorted array. Overrides ``numpy.argsort`` function.
    """
    return np.argsort(a, *args, **kwargs).view(np.ndarray)


@implements(np.concatenate)
def concatenate(arrays, axis=0, out=None):
    """
    Join a sequence of instances of ``BathArray`` along an existing axis. Overrides ``numpy.concatenate`` function.

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
    imap = InteractionMap()

    offset = 0
    for x in arrays:
        types += x.types
        if x.imap:
            imap += x.imap.shift(offset, inplace=False)

        offset += x.size

    new_array.types = types

    if imap:
        new_array.imap = imap

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


def transform(atoms, center=None, cell=None, rotation_matrix=None, style='col', inplace=True):
    """
    Coordinate transformation of BathArray.

    Args:
        atoms (BathArray): Array to be transformed.

        center (ndarray with shape (3,)): (0, 0, 0) position of new coordinates in the initial frame.

        cell (ndarray with shape (3, 3)): Cell vectors in cartesian coordinates
            if initial coordinates of the ``atoms`` are in crystallographic reference frame.

        rotation_matrix (ndarray with shape (3, 3)): Rotation matrix R of the **coordinate system**.

            E.g. ``R @ [0, 0, 1] = [a, b, c]`` where ``[a, b, c]`` are coordinates of the z axis of the new coordinate
            system in the old coordinate system.

            Note, that rotaton is applied after transition from cell coordinates to the cartesian coordinates,
            in which cell vectors are stored.

        style (str): Can have two values: 'col' or 'row'.
            Shows how ``cell`` and ``rotation_matrix`` matrices are given:

                * if 'col', each column of the matrix is a vector in previous coordinates;
                * if 'row' - each row is a new vector.

            Default 'col'.

        inplace (bool): If true, makes inplace changes to the provided array.

    Returns:
        BathArray: Transformed array with bath spins.
    """

    styles = ['col', 'row']
    if style not in styles:
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

    if style.lower() == 'row':
        cell = cell.T
        rotation_matrix = rotation_matrix.T

    if not atoms.dtype.names:
        atoms -= np.asarray(center)
        atoms = np.einsum('jk,ik->ij', cell, atoms)
        atoms = np.einsum('jk,ik->ij', np.linalg.inv(rotation_matrix), atoms)

        return atoms

    atoms['xyz'] -= np.asarray(center)

    atoms['xyz'] = np.einsum('jk,ik->ij', cell, atoms['xyz'])
    atoms['xyz'] = np.einsum('jk,ik->ij', np.linalg.inv(rotation_matrix), atoms['xyz'])

    if 'A' in atoms.dtype.names:
        atoms['A'] = np.matmul(atoms['A'], rotation_matrix)
        atoms['A'] = np.matmul(np.linalg.inv(rotation_matrix), atoms['A'])

    if 'Q' in atoms.dtype.names:
        atoms['Q'] = np.matmul(atoms['Q'], rotation_matrix)
        atoms['Q'] = np.matmul(np.linalg.inv(rotation_matrix), atoms['Q'])

    return atoms


def _set_sd_attribute(array, attribute_name, initial_value):
    value = np.asarray(initial_value)

    if array.shape and value.shape == array.shape:
        keys, ind = np.unique(array['N'], return_index=True)
        values = value[ind]

        for k, v in zip(keys, values):
            _inner_set_attr(array.types, k, attribute_name, v)

        return

    if not array.shape:
        _inner_set_attr(array.types, array, attribute_name, initial_value)

        return

    for k in np.unique(array['N']):
        _inner_set_attr(array.types, k, attribute_name, initial_value)

        return


def _inner_set_attr(types, key, attr, value):
    try:
        setattr(types[key], attr, value)
    except KeyError:
        types[key] = (0, 0, 0)
        setattr(types[key], attr, value)
    return


class SpinType:
    r"""
    Class which contains properties of each spin type in the bath.

    Args:
        name (str): Name of the bath spin.
        s (float): Total spin of the bath spin.

            Default 0.

        gyro (float): Gyromagnetic ratio in rad * kHz / G.

            Default 0.

        q (float): Quadrupole moment in barn (for s > 1/2).

            Default 0.

        detuning (float): Energy detuning from the zeeman splitting in kHz,
            included as an extra :math:`+\omega \hat S_z` term in the Hamiltonian,
            where :math:`\omega` is the detuning.

            Default 0.


    Attributes:

        name (str): Name of the bath spin.
        s (float): Total spin of the bath spin.
        dim (int): Spin dimensionality = 2s + 1.

        gyro (float): Gyromagnetic ratio in rad/(ms * G).
        q (float): Quadrupole moment in barn (for s > 1/2).
        detuning (float): Energy detuning from the zeeman splitting in kHz.

    """

    def __init__(self, name, s=0., gyro=0., q=0., detuning=0.):

        self.name = name
        self.s = s

        try:
            self.dim = np.int(2 * s + 1 + 1e-8)

        except TypeError:
            self.dim = (2 * s + 1 + 1e-8).astype(np.int32)

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
        try:
            base_message = f'{self.name}: ({self.s:.1f}, {self.gyro:.4f}'
        except TypeError:
            base_message = f'{self.name}: ({self.s}, {self.gyro}'

        if np.asarray(self.q).any():
            try:
                m = f', {self.q:.4f}'
            except TypeError:
                m = f', {self.q}'
            base_message += m

        if np.asarray(self.detuning).any():
            try:
                m = f', {self.detuning:.4f}'
            except TypeError:
                m = f', {self.detuning}'
            base_message += m

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

    Args:
        *args: Any numbers of arguments which could initialize ``SpinType`` instances.
        **kwargs: Any numbers of keyword arguments which could initialize ``SpinType`` instances.
            For details see ``SpinDict.add_type`` method.

    """

    def __init__(self, *args, **kwargs):
        super(SpinDict, self).__init__()
        self.add_type(*args, **kwargs)

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
            key = np.asarray(key)
            if key.shape:
                try:
                    names = key['N']
                except IndexError:
                    names = key

                for k, v in zip(names, value):
                    v = _check_key_spintype(k, v)
                    super().__setitem__(k, v)
                return

            k = key[()]

            try:
                k = k['N']
            except TypeError:
                pass

            value = _check_key_spintype(k, value)
            super().__setitem__(k, value)

    def __getitem__(self, key):

        try:
            key = key[()]
            return self._super_get_item(key['N'])
            # self._super_get_item(key)

        except (TypeError, IndexError):
            try:
                return self._super_get_item(key)
            except TypeError:

                if key.dtype.names:
                    key = key['N']

                unique_names = np.unique(key)

                if unique_names.size == 1:
                    n = unique_names[0]
                    # ones = np.ones(key.shape, dtype=np.float64)

                    spins = self._super_get_item(n).s  # * ones
                    gyros = self._super_get_item(n).gyro  # * ones
                    quads = self._super_get_item(n).q  # * ones
                    detus = self._super_get_item(n).detuning  # * ones

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
            message += f"{self.data[k]}, "
            if len(message) > 75:
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
            super().__setitem__(n, copy.copy(common_isotopes[n]))
            return super().__getitem__(n)

    def add_type(self, *args, **kwargs):
        """
        Add one or several spin types to the spin dictionary.

        Args:
            *args:
                Any numbers of arguments which could initialize ``SpinType`` instances.
                Accepted arguments:

                    * Tuple containing name, spin, gyromagnetic ratio, quadrupole constant (optional)
                      and detuning (optional).
                    * ``SpinType`` instance.

                Can also initialize one instance of ``SpinType`` if each argument corresponds to
                each positional argument necessary to initiallize.

            **kwargs: Any numbers of keyword arguments which could initialize ``SpinType`` instances.
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
        keys = []
        try:
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


_spin_not_found_message = lambda x: 'Spin type for {} was not provided and was not found in common isotopes.'.format(x)

import pandas as pd

# try:
#     url = 'https://raw.githubusercontent.com/StollLab/EasySpin/main/easyspin/private/isotopedata.txt'
#     all_spins = pd.read_csv(url, delim_whitespace=True, header=None, comment='%',
#                             names=['protons', 'nucleons', 'radioactive', 'symbol', 'name', 'spin', 'g', 'conc', 'q'])
# except:
import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
filepath = os.path.join(__location__, 'isotopes.txt')
all_spins = pd.read_csv(filepath, delim_whitespace=True, header=None, comment='%',
                        names=['protons', 'nucleons', 'radioactive', 'symbol', 'name', 'spin', 'g', 'conc', 'q'])

stable_spins = all_spins[(all_spins['spin'] > 0) & (all_spins['conc'] > 0)]

_names = stable_spins['nucleons'].astype(str) + stable_spins['symbol']
_gyros = stable_spins['g'] / HBAR_SI * NUCLEAR_MAGNETON / 1e7
_quads = stable_spins['q']
_spins = stable_spins['spin']

_mi = pd.MultiIndex.from_arrays([stable_spins['symbol'], _names])
_ser = pd.Series((stable_spins['conc'] / 100).values, index=_mi)

common_concentrations = {level: _ser.xs(level).to_dict() for level in _ser.index.levels[0]}
"""
dict: Nested dict containing natural concentrations of the stable nuclear isotopes.  
"""

# Dictionary of the common isotopes. Placed in this file to avoid circular dependency
common_isotopes = SpinDict(*zip(_names, _spins, _gyros, _quads))
"""
SpinDict: An instance of the ``SpinDict`` dictionary, containing properties for the most of the common isotopes with 
nonzero spin.
The isotope is considered common if it is stable and has nonzero concentration in nature.  
"""

# electron spin
common_isotopes['e'] = SpinType('e', 1 / 2, ELECTRON_GYRO, 0)
