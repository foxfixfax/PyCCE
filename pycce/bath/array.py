import copy
import warnings
from collections import UserDict
from collections.abc import Mapping

import numpy as np
from numpy.lib.recfunctions import repack_fields
from pycce.bath.map import InteractionMap
from pycce.bath.state import BathState
from pycce.constants import HBAR_MU0_O4PI, ELECTRON_GYRO, HBAR_SI, NUCLEAR_MAGNETON, PI2
from pycce.utilities import gen_state_list, vector_from_s, rotate_coordinates, rotate_tensor, _add_args

HANDLED_FUNCTIONS = {}

_set_str_kinds = {'U', 'S'}

_stevens_str_doc = r"""
            dict: Dictionary with additional spin Hamiltonian parameters.
            Key denotes the product of spin operators as:
            
            Either a string containing ``x, y, z, +, -``  where each symbol is a corresponding spin operator:
            
                - ``x`` == :math:`S_x`
                - ``y`` == :math:`S_y`
                - ``z`` == :math:`S_z`
                - ``p`` == :math:`S_+`
                - ``m`` == :math:`S_-`
            
            Several symbols is a product of those spin operators.
            
            Or a tuple with indexes (k, q) for Stevens operators
            (see https://www.easyspin.org/documentation/stevensoperators.html).
            
            The item is the coupling parameter in float.
            
            Examples:
            
                - ``d['pm'] = 2000`` corresponds to the Hamiltonian term
                  :math:`\hat H_{add} = A \hat S_+ \hat S_-` with :math:`A = 2` MHz.
            
                - ``d[2, 0] = 1.5e6`` corresponds to Stevens operator
                  :math:`B^q_k \hat O^q_k = 3 \hat S_z - s(s+1) \hat I`
                  with :math:`k = 2`, :math:`q = 0`, and :math:`B^q_k = 1.5` GHz. """


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
        >>> ca = np.random.random((2, 3))
        >>> sn = ['1H', '2H']
        >>> hf = np.random.random((2, 3, 3))
        >>> ba = BathArray(ca=ca, hf=hf, sn=sn)
        >>> print(ba.N, ba.types)
        ['1H' '2H'] SpinDict(1H: (1H, 0.5, 26.7519), 2H: (2H, 1, 4.1066, 0.00286))

    .. warning::
        Due to how structured arrays work, if one uses a boolean array to access an subarray,
        and then access the name field, the initial array *will not change*.

        Example:

            >>> ha = BathArray((10,), sn='1H')
            >>> print(ha.N)
            ['1H' '1H' '1H' '1H' '1H' '1H' '1H' '1H' '1H' '1H']
            >>> bool_mask = np.arange(10) % 2 == 0
            >>> ha[bool_mask]['N'] = 'e'
            >>> print(ha.N)
            ['1H' '1H' '1H' '1H' '1H' '1H' '1H' '1H' '1H' '1H']

            To achieve the desired result, one should first access the name field and only then apply the boolean mask:

            >>> ha['N'][bool_mask] = 'e'
            >>> print(ha.N)
            ['e' '1H' 'e' '1H' 'e' '1H' 'e' '1H' 'e' '1H']

    Each bath spin can initiallized in some specific state accessing the ``.state`` attribute. It takes
    both state vectors and density matrices as values. See ``.state`` attribute documentation for details.

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

    _dtype_names = ('N', 'xyz', 'A', 'Q')

    def __new__(subtype, shape=None, array=None,
                names=None, hyperfines=None, quadrupoles=None,
                types=None, imap=None,
                ca=None, sn=None, hf=None, q=None, efg=None, state=None,
                center=1):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments. This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to BathArray.__array_finalize__
        if center == 1:
            atupl = (3, 3)
        else:
            atupl = (center, 3, 3)

        _dtype_bath = np.dtype([('N', np.unicode_, 16),
                                ('xyz', np.float64, (3,)),
                                ('A', np.float64, atupl),
                                ('Q', np.float64, (3, 3)),
                                # ('state', object),
                                # ('proj', np.float64),
                                ])

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
            obj = super(BathArray, subtype).__new__(subtype, shape, dtype=_dtype_bath)
        else:

            for a in (array, hyperfines, quadrupoles):
                if a is not None:
                    try:
                        obj = super(BathArray, subtype).__new__(subtype, (np.asarray(a).shape[0],),
                                                                dtype=_dtype_bath)
                    except IndexError:  # Empty tuple
                        obj = super(BathArray, subtype).__new__(subtype, np.asarray(a).shape,
                                                                dtype=_dtype_bath)
                    break
            else:
                raise ValueError('No shape provided')

        obj.types = SpinDict()
        obj.imap = imap

        obj._state = BathState(obj.size)

        if state is not None:
            obj.state = state

        # obj.__projected_state = np.zeros(obj.shape, dtype=np.float64)

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
            obj['A'] = np.asarray(hyperfines).reshape(-1, *atupl)
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
        # if obj.dtype.names != self._dtype_names:
        #     warnings.warn('Trying to view array with unknown dtype as BathArray. '
        #                   'This can lead to unexpected results.',
        #                   RuntimeWarning, stacklevel=2)

        self.types = getattr(obj, 'types', None)
        self.imap = getattr(obj, 'imap', None)
        self._state = getattr(obj, '_state', None)
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
        if not any(issubclass(t, BathArray) for t in types):
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

    @_add_args(_stevens_str_doc)
    @property
    def h(self):
        return _get_sd_attribute(self, 'h')

    @property
    def so(self):
        return _get_sd_attribute(self, 'so')

    @property
    def name(self):
        """
        ndarray: Array of the ``name`` attribute for each spin in the array from ``types`` dictionary.

        .. note::
            While the value of this attribute should be the same as the ``N`` field of the BathArray instance,
            ``.name`` *should not* be used for production as it creates a *new* array from ``types`` dictionary.
        """
        return _get_sd_attribute(self, 'name')

    @name.setter
    def name(self, initial_value):
        _set_sd_attribute(self, 'name', initial_value)

    @property
    def s(self):
        """
        ndarray: Array of the ``spin`` (spin value) attribute for each spin in the array from ``types`` dictionary.
        """
        return _get_sd_attribute(self, 's')

    @s.setter
    def s(self, initial_value):
        _set_sd_attribute(self, 's', initial_value)

    @property
    def dim(self):
        """
        ndarray: Array of the ``dim`` (dimensions of the spin) attribute
            for each spin in the array from ``types`` dictionary.
        """
        return _get_sd_attribute(self, 'dim')

    @property
    def gyro(self):
        """
        ndarray: Array of the ``gyro`` (gyromagnetic ratio)
            attribute for each spin in the array from ``types`` dictionary.
        """
        return _get_sd_attribute(self, 'gyro')

    @gyro.setter
    def gyro(self, initial_value):
        _set_sd_attribute(self, 'gyro', initial_value)

    @property
    def q(self):
        """
        ndarray: Array of the ``q`` (quadrupole moment)
            attribute for each spin in the array from ``types`` dictionary.
        """
        return _get_sd_attribute(self, 'q')

    @q.setter
    def q(self, initial_value):
        _set_sd_attribute(self, 'q', initial_value)

    @property
    def detuning(self):
        """
        ndarray: Array of the ``detuning``
            attribute for each spin in the array from ``types`` dictionary.
        """
        return _get_sd_attribute(self, 'detuning')

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

    @property
    def nc(self):
        """int: Number of central spins."""
        selfdim = len(self.shape)
        return self.A.shape[selfdim] if len(self.A.shape) == selfdim + 3 else 1

    @property
    def state(self):
        """
        BathState: Array of the bath spin states.

        Can have three types of entries:

            - **None**. If entry is **None**, assumes fully random density matrix. Default value.
            - **ndarray with shape (s,)**. If entry is vector, corresponds to the pure state of the spin.
            - **ndarray with shape (s, s)**. If entry is a matrix, corresponds to the density matrix of the spin.

        Examples:

            >>> print(ba.state)
            [None None]
            >>> ba[0].state = np.array([0, 1])
            >>> print(ba.state)
            [array([0, 1]) None]
        """
        return self._state

    @state.setter
    def state(self, rho):
        self._state[...] = rho

    @property
    def proj(self):
        """
        ndarray: Array of :math:`S_z` projections of the bath spin states.
        """
        return self.state.proj

    @proj.setter
    def proj(self, rho):
        # rho can be either (on the example of s = 1)
        # projection of pure state:  rho = 1 for ms = 1
        # pure state: rho = [0,0,1] for ms = - 1
        # density matrix rho: = [[0,0,0],[0,1,0],[0,0,0]] for ms = 0
        rho = np.asarray(rho)

        if not rho.shape:
            # assume s is int or float showing the spin projection in the pure state

            d = self.dim
            if not self.shape:
                rho = vector_from_s(rho, d)
            else:
                if (d == d[0]).all():
                    d = d[0]
                    rho = np.broadcast_to(vector_from_s(rho, d), self.shape + (d,))
                else:
                    rho = [vector_from_s(rho, d_i) for d_i in d]
        else:
            rho = gen_state_list(rho, np.broadcast_to(self.dim, self.shape))
        self._state[...] = rho

    @property
    def has_state(self):
        """
        ndarray: Bool array. True if given spin was initialized with a state, False otherwise.
        """
        return self.state.has_state

    def __getitem__(self, item):
        if isinstance(item, (int, np.int32, np.int64)):
            item = (Ellipsis, item)
            obj = np.ndarray.__getitem__(self, item)
            # obj._state = self._state[item]
            obj._state = self._state._get_state(item)

            return obj

        elif isinstance(item, tuple) and not item:
            return np.ndarray.__getitem__(self, item)
        #     else:
        #         item = (Ellipsis,) + item
        #         obj = np.ndarray.__getitem__(self, item)
        #         # obj._state = self._state[item]
        #         return obj

        # if string then return ndarray view of the field
        elif isinstance(item, (str, np.str_)):
            try:
                value = self.view(np.ndarray).__getitem__(item)

                if not value.shape:
                    return value[()]

                return value

            except ValueError:
                return self[self['N'] == item]

        else:

            obj = np.ndarray.__getitem__(self, item)

            try:
                obj._state = self._state._get_state(item)

                if self.imap is not None:
                    if not isinstance(item, tuple):

                        if isinstance(item, slice):
                            item = np.arange(self.size)[item]
                        smap = self.imap.subspace(item)
                        if smap:
                            obj.imap = smap

            except AttributeError:
                pass

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

            xyzs = (self['xyz'] == other['xyz']).all(axis=-1)
            hfs = (self['A'] == other['A']).reshape(*self.shape, -1).all(axis=-1)
            qds = (self['Q'] == other['Q']).reshape(*self.shape, -1).all(axis=-1)

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

    def add_single_jump(self, operator, rate=1, units='rad', square_root=False, which=None):
        """
        Add single-spin jump operator for the given type of spins to be used in the Lindbladian master equation CCE.

        Args:
            operator (str or ndarray with shape (dim, dim)): Definition of the operator. Can be either of the following:
                * Pair of integers defining the Sven operator.
                * String where each symbol corresponds to the spin matrix or operation between them.
                  Allowed symbols: ``xyz+``. If there is nothing between symbols,
                  assume multiplication of the operators.
                  If there is a ``+`` symbol, assume summation between terms. For example, ``xx+z`` would correspond to
                  the operator :math:`\hat S_x \hat S_x + \hat S_z`.
                * String equal to ``A``. Then assumes that the correct matrix form of the operator has been provided
                  by the user.

            rate (float): Rate associated with the given jump operator. By default, is given in rad ms^-1.
            units (str): Units of the rate, can be either rad (for radial frequency units) or deg
                (for rotational frequency).
            square_root (bool): True if the rate is given as a square root of the rate (to match how one sets up
                collapse operators in Qutip). Default False.

            which (str): For which type of the spins add the jump operator. Default is None - if
                 there is only one spin type in the array then the jump operator is added,
                 otherwise the exception is raised.
        """
        if not square_root:
            rate = np.sqrt(rate)
        if 'rad' in units:
            rate = rate / np.sqrt(PI2)

        if which is None:
            superoperators = self.so
        else:
            superoperators = self[which].so

        if isinstance(operator, str):
            superoperators[operator] = rate
        else:
            operator = np.asarray(operator, dtype=np.complex128)
            dimensions = self.dim

            if self.size > 1:
                dimensions = self.dim[0]

            assert operator.shape == (dimensions, dimensions), f'Operator should have shape {dimensions, dimensions}'
            superoperators['A'] = operator * rate

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

    def from_center(self, center, inplace=True, cube=None, which=0, **kwarg):
        """
        Generate hyperfine couplings using either the point dipole approximation or spin density in the .cube format,
        with the information from the CenterArray instance.

        Args:
            center (CenterArray): Array, containing properties of the central spin

            inplace (bool): True if changes parameters of the array in place. If False, returns copy of the array.

            cube (Cube or iterable of Cubes): An instance of ``Cube`` object,
                which contains spatial distribution of spin density of central spins.
                For details see documentation of ``Cube`` class.

            which (int): If ``cube`` is a single Cube instance,
                this is an index of the central spin it corresponds to.

            **kwarg: Additional arguments for .from_cube method.

        Returns:
            BathArray: Updated BathArray instance.

        """
        if inplace:
            array = self
        else:
            array = self.copy()

        if array.nc != center.size:
            array = array.expand(center.size)
            if inplace:
                warnings.warn("Cannot change array inplace, using a copy instead.")

        if center.size == 1:
            if cube is None:
                array.from_point_dipole(center[0].xyz, center[0].gyro, inplace=True)
            else:
                array.from_cube(cube, center[0].gyro, inplace=True, **kwarg)
        else:
            if cube is None:
                array.from_point_dipole(center.xyz, center.gyro, inplace=True)
            else:
                try:
                    for index, (cen, cub) in enumerate(zip(center, cube)):
                        array.from_cube(cub, cen.gyro, inplace=True, which=index, **kwarg)
                except TypeError:
                    array.from_point_dipole(center.xyz, center.gyro, inplace=True)
                    array.from_cube(cube, center[which].gyro, inplace=True, which=which, **kwarg)

        return array

    def from_point_dipole(self, position, gyro_center=ELECTRON_GYRO, inplace=True):
        """
        Generate hyperfine couplings, assuming that bath spins interaction with central spin is the same as the
        one between two magnetic point dipoles.

        Args:
            position (ndarray with shape (3,)): position of the central spin

            gyro_center (float or ndarray with shape (3,3)):
                gyromagnetic ratio of the central spin

                **OR**

                tensor corresponding to interaction between magnetic field and central spin.

            inplace (bool): True if changes parameters of the array in place. If False, returns copy of the array.

        Returns:
            BathArray: Updated BathArray instance with changed hyperfine couplings.
        """
        if inplace:
            array = self
        else:
            array = self.copy()

        position = np.asarray(position)

        if array.nc > 1:
            position = np.broadcast_to(position, (array.nc, 3))
            gyro_center = np.asarray(gyro_center)
            gyro_center = np.broadcast_to(gyro_center, (array.nc, *gyro_center.shape[1:]))

            for i in range(array.nc):
                pos = array.xyz - position[i]
                array.A[:, i] = point_dipole(pos, array.gyro, gyro_center[i])

            return array

        pos = array.xyz - position
        array.A = point_dipole(pos, array.gyro, gyro_center)

        return array

    def from_cube(self, cube, gyro_center=ELECTRON_GYRO, inplace=True, which=0, **kwargs):
        """
        Generate hyperfine couplings, assuming that bath spins interaction with central spin can be approximated as
        a point dipole, interacting with given spin density distribution.

        Args:
            cube (Cube): An instance of `Cube` object, which contains spatial distribution of spin density.
                For details see documentation of `Cube` class.

            gyro_center (float): Gyromagnetic ratio of the central spin.

            inplace (bool): True if changes parameters of the array in place. If False, returns copy of the array.

        Returns:
            BathArray: Updated BathArray instance with changed hyperfine couplings.
        """
        if inplace:
            array = self
        else:
            array = self.copy()

        gyros = array.gyro

        if array.nc > 1:
            array.A[:, which] = cube.integrate(array.xyz, gyros, gyro_center, **kwargs)

            return array

        array.A = cube.integrate(array.xyz, gyros, gyro_center, **kwargs)

        return array

    def from_func(self, func, *args, inplace=True, **kwargs):
        """
        Generate hyperfine couplings from user-defined function.

        Args:

            func (func):
                Callable with signature::

                    func(array, *args, **kwargs)

                where ``array`` is array of the bath spins,
            *args: Positional arguments of the ``func``.
            **kwargs: Keyword arguments of the ``func``.
            inplace (bool): True if changes parameters of the array in place. If False, returns copy of the array.

        Returns:
            BathArray: Updated BathArray instance with changed hyperfine couplings.

        """
        if inplace:
            array = self
        else:
            array = self.copy()

        func(array, *args, **kwargs)

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

        array.Q = pref * efg
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
        if isinstance(position, BathArray):
            return np.linalg.norm(self.xyz - position.xyz, axis=-1)

        if position is None:
            position = np.zeros(3)
        else:
            position = np.asarray(position)

        return np.linalg.norm(self.xyz - position, axis=-1)

    def savetxt(self, filename, fmt='%18.8f', strip_isotopes=False, **kwargs):
        r"""
        Save name of the isotopes and their coordinates to the txt file of xyz format.

        Args:
            filename (str or file): Filename or file handle.
            fmt (str): Format of the coordinate entry.
            strip_isotopes (bool): True if remove numbers from the name of bath spins. Default False.
            **kwargs: Additional keywords of the ``numpy.savetxt`` function.
        """
        kwargs.setdefault('comments', '')
        ar = repack_fields(self.view(np.ndarray)[['N', 'xyz']]).view(
            np.dtype([('N', 'U16'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8')]))

        if strip_isotopes:
            ar['N'] = np.core.defchararray.strip(ar['N'], '1234567890')

        header = f'{ar.size}\n'
        for n, c in zip(*np.unique(ar['N'], return_counts=True)):
            header += f'{n}{c} '

        kwargs.setdefault('header', header)
        np.savetxt(filename, ar, fmt=('%s', fmt, fmt, fmt), **kwargs)

    def expand(self, ncenters):

        array = BathArray(array=self.xyz, quadrupoles=self.Q, names=self.N, center=ncenters, imap=self.imap,
                          types=self.types, state=self.state)

        hyperfine = self.A
        if hyperfine.any():
            ocs = self.nc  # old central spin
            if ocs == ncenters:
                array.A = hyperfine

            elif ocs == 1 and ncenters > 1:
                array.A[..., 0, :, :] = hyperfine

            elif ocs > 1 and ncenters == 1:
                array.A[..., :, :] = hyperfine[..., 0, :, :]

            else:
                limit = ocs if ocs < ncenters else ncenters
                array.A[..., :limit + 1, :, :] = hyperfine[..., :limit + 1, :, :]

        return array


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
def argsort(a, *args, **kwargs):
    """
    Return ``a`` indexes of a sorted array. Overrides ``numpy.argsort`` function.
    """
    return np.argsort(a, *args, **kwargs).view(np.ndarray)


@implements(np.delete)
def delete(arr, obj, axis=None):
    newarr = np.delete(arr.view(np.ndarray), obj, axis=axis).view(BathArray)
    newarr._state = BathState(newarr.size)
    newarr.types = arr.types

    if arr.imap:
        newarr.imap = arr.imap.subspace(obj)

    newarr.state = np.delete(arr.state[...], obj, axis=axis)
    return newarr


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
    state = BathState(new_array.size)

    for x in arrays:
        types += x.types

        if x.imap:
            imap += x.imap.shift(offset, inplace=False)

        state[offset:offset + x.size] = x.state
        offset += x.size

    new_array.types = types
    new_array._state = state

    if imap:
        new_array.imap = imap

    return new_array


# @implements(np.broadcast_to)
# def broadcast_to(array, shape):
#     ...  # implementation of broadcast_to for MyArray objects

def check_gyro(gyro):
    """
    Check if gyro is matrix or scalar.

    Args:
        gyro (ndarray or float): Gyromagnetic ratio matrix or float.

    Returns:
        tuple: tuple containing:

            * **ndarray or float**: Gyromagnetic ratio.
            * **bool**: True if gyro is float, False otherwise.
    """
    try:
        gyro = float(gyro)
        check = True
    except TypeError:
        check = False

    if not check:
        gyro = np.asarray(gyro)
        if gyro.ndim == 1:  # Assume array
            check = True
        elif not gyro.shape or gyro.shape[0] == 1:
            check = True
            gyro = gyro.reshape(1)[0]
        else:
            test_gyros = gyro.copy()
            indexes = np.arange(gyro.shape[-1])
            test_gyros[..., indexes, indexes] = 0

            diag_check = np.isclose(test_gyros, 0).all()
            same_check = ((gyro[..., 0, 0] == gyro[..., 1, 1]) & (gyro[..., 1, 1] == gyro[..., 2, 2])).all()
            check = diag_check & same_check
            if check:
                gyro = gyro[..., 0, 0][()]

    return gyro, check


def point_dipole(pos, gyro_array, gyro_center):
    """
    Generate an array hyperfine couplings, assuming point dipole approximation.

    Args:
        pos (ndarray with shape (n, 3)): Relative position of the bath spins.
        gyro_array (ndarray with shape (n,)): Array of the gyromagnetic ratios of the bath spins.

        gyro_center (float or ndarray with shape (3, 3)):
            gyromagnetic ratio of the central spin

            **OR**

            tensor corresponding to interaction between magnetic field and central spin.

    Returns:
        ndarray with shape (n, 3, 3): Array of hyperfine tensors.
    """
    identity = np.eye(3, dtype=np.float64)

    try:
        posxpos = np.einsum('...i,...j->...ij', pos, pos)
    except ValueError:
        posxpos = np.outer(pos, pos)

    r = np.linalg.norm(pos, axis=-1)[..., np.newaxis, np.newaxis]
    gyro_center, check = check_gyro(gyro_center)
    gyro_array, check_array = check_gyro(gyro_array)

    gyro_center = np.asarray(gyro_center)
    gyro_array = np.asarray(gyro_array)

    if check and check_array:

        if gyro_center.shape and gyro_array.shape:
            pref = gyro_center[:, np.newaxis] * gyro_array[np.newaxis, :]

        else:
            pref = gyro_center * gyro_array
        out = -(3 * posxpos - identity * r ** 2) / (r ** 5)
        out = out * pref.reshape(*pref.shape, 1, 1) * HBAR_MU0_O4PI / PI2
        return out

    out = -(3 * posxpos - identity * r ** 2) / (r ** 5) * HBAR_MU0_O4PI / PI2

    if not check_array:
        out = np.matmul(out, gyro_array)
    else:
        out *= gyro_array.reshape(*gyro_array.shape, 1, 1)
    if not check:
        out = np.matmul(gyro_center, out)
    else:
        out *= gyro_center.reshape(*gyro_center.shape, 1, 1)
    return out


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
    dist_matrix = np.linalg.norm(barray_1.xyz[:, np.newaxis, :] - barray_2.xyz[np.newaxis, :, :], axis=-1)

    if ignore_isotopes:
        tb_names = np.core.defchararray.strip(barray_1.N, '1234567890')
        ab_names = np.core.defchararray.strip(barray_2.N, '1234567890')

        same_names = tb_names[:, np.newaxis] == ab_names[np.newaxis, :]

    else:
        same_names = barray_1.N[:, np.newaxis] == barray_2.N[np.newaxis, :]

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
    neq = total_bath.nc != added_bath.nc
    if neq and added_bath.nc != 1:
        raise ValueError('Arrays correspond to different number of central spins.')

    for n in added_bath.dtype.names:
        if ignore_isotopes and n == 'N':
            continue
        if n == 'A' and neq:
            total_bath[n][indexes, 0] = added_bath[n][ext_indexes]
        else:
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
    if style.lower() not in styles:
        raise ValueError('Unsupported style of matrices. Available styles are: ' + ', '.join(*styles))

    if not inplace:
        atoms = atoms.copy()

    if center is None:
        center = np.zeros(3)

    if not atoms.dtype.names:
        atoms -= np.asarray(center)
        atoms = rotate_coordinates(atoms, rotation=rotation_matrix, cell=cell, style=style)

        return atoms

    atoms['xyz'] -= np.asarray(center)

    atoms['xyz'] = rotate_coordinates(atoms['xyz'], rotation=rotation_matrix, cell=cell, style=style)

    if 'A' in atoms.dtype.names:
        atoms['A'] = rotate_tensor(atoms['A'], rotation=rotation_matrix, style=style)

    if 'Q' in atoms.dtype.names:
        atoms['Q'] = rotate_tensor(atoms['Q'], rotation=rotation_matrix, style=style)

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


def _get_sd_attribute(array, attribute_name):
    if not array.shape:
        return getattr(array.types[array], attribute_name)

    if (attribute_name == 'h') or (attribute_name == 'so'):
        if array.size == 1:
            return getattr(array.types[array[0]], attribute_name)

        unique_names = np.unique(array.N)

        if unique_names.size == 1:
            return getattr(array.types[array[0]], attribute_name)

        raise RuntimeError('Hamiltonian and Jump operator terms can be modified only for single spin type at a time')

    if array.size == 1:
        newarr = np.array([getattr(array.types[array.N[0]], attribute_name)])

        if attribute_name == 'dim':
            newarr = newarr.astype(int)

        return newarr

    unique_names = np.unique(array.N)

    if unique_names.size == 1:
        n = unique_names[0]
        v = getattr(array.types[n], attribute_name)
        if attribute_name == 'gyro' and isinstance(v, np.ndarray):
            values = np.tile(v, reps=(array.size, 1, 1))
        else:
            if attribute_name == 'dim':
                ones = np.ones(array.shape, dtype=int)
            else:
                ones = np.ones(array.shape, dtype=np.float64)
            values = v * ones

    else:

        temp_values = []
        check = True
        for n in unique_names:
            v = getattr(array.types[n], attribute_name)
            if attribute_name == 'gyro' and isinstance(v, np.ndarray):
                check = False
            temp_values.append(v)

        if check:
            if attribute_name == 'dim':
                values = np.empty(array.shape, dtype=int)
            else:
                values = np.empty(array.shape, dtype=np.float64)

            for n, v in zip(unique_names, temp_values):
                values[array.N == n] = v
        else:
            values = np.empty((array.size, 3, 3), dtype=np.float64)
            for i, n in enumerate(unique_names):
                v = temp_values[i]
                if not isinstance(v, np.ndarray):
                    temp_values[i] = np.eye(3, dtype=np.float64) * v
                values[array.N == n] = v
    return values


def broadcast_array(array, root=0):
    """
    Using mpi4py broadcast ``BathArray`` or ``CenterArray`` to all processes.
    Args:
        array (BathArray or CenterArray): Array to broadcast.
        root (int): Rank of the process to broadcast from.

    Returns:
        BathArray or CenterArray: Broadcasted array.
    """
    import mpi4py
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == root:
        parameters = vars(array)
    else:
        array = None
        parameters = None

    nbath = comm.bcast(array, root)
    nparam = comm.bcast(parameters, root)

    for k in nparam:
        setattr(nbath, k, nparam[k])

    return nbath


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

        self.gyro = gyro
        self.q = q
        self.detuning = detuning
        self._h = None  # custom hamiltonian
        self._so = None  # superoperators

    def __eq__(self, obj):
        if not isinstance(obj, SpinType):
            return False

        checks = (self.name == obj.name) & (self.s == obj.s) & (
                self.gyro == obj.gyro) & (self.q == obj.q) & (self.detuning == obj.detuning)

        return checks

    @_add_args(_stevens_str_doc)
    @property
    def h(self):
        if self._h is None:
            self._h = {}
        return self._h

    @property
    def so(self):

        if self._so is None:
            self._so = {}
        # thinking about making a key with coma, before coma - left redfield operator, after - right redfield operator
        return self._so


    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def s(self):
        return self._s

    @s.setter
    def s(self, s):

        try:
            if isinstance(s, np.ndarray) and s.shape:
                raise TypeError

            self._s = float(s)
            self._dim = int(2 * s + 1 + 1e-8)

        except TypeError:
            self._dim = np.asarray(2 * s + 1 + 1e-8).astype(np.int32)
            self._s = np.asarray(s).astype(np.float64)

    @property
    def gyro(self):
        return self._gyro

    @gyro.setter
    def gyro(self, gyro):
        gyro, check = check_gyro(gyro)
        self._gyro = gyro

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q):
        self._q = q

    @property
    def detuning(self):
        return self._detuning

    @detuning.setter
    def detuning(self, d):
        try:
            self._detuning = float(d)
        except TypeError:
            self._detuning = np.asarray(d).astype(np.float64)

    @property
    def dim(self):
        return self._dim

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

                key = np.asarray(key)
                if key.size == 1:
                    n = key[0]
                    spins = np.array([self._super_get_item(n).s])
                    gyros = np.array([self._super_get_item(n).gyro])
                    quads = np.array([self._super_get_item(n).q])
                    detus = np.array([self._super_get_item(n).detuning])
                    return SpinType(key, s=spins, gyro=gyros, q=quads, detuning=detus)
                else:
                    raise TypeError('Unsupported key.')


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
            st = SpinType(*args, **kwargs)
            self[st.name] = st
            return
        except TypeError:
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


def _process_key_operator(key, rate, sm):
    r"""
    Process key of the .so or .h dictionaries of the SpinType
    Args:
        key (str or int): key of the dictionary. Can be either of the following:

            * Pair of integers defining the Sven operator.
            * String where each symbol corresponds to the spin matrix or operation between them.
              Allowed symbols: ``xyz+``. If there is nothing between symbols, assume multiplication of the operators.
              If there is a ``+`` symbol, assume summation between terms. For example, ``xx+z`` would correspond to
              the operator :math:`\hat S_x \hat S_x + \hat S_z`.
            * String equal to ``A``. Then assumes that the correct matrix form of the operator has been provided
              by the user.


        rate (float or ndarray with shape (n,n): value stored in the dictionary.
        sm (SpinMatrix): Object containing spin matrices of the given spin.

    Returns:
        ndarray with shape (n,n): Resulting operator.
    """
    if isinstance(key, str):
        if key.lower() == 'a':
            return rate

        separated = key.split('+')
        operator = 0
        for k in separated:
            current = None
            for sym in k:
                current = getattr(sm, sym) if current is None else np.matmul(current, getattr(sm, sym))
            operator = operator + current

        operator = operator * rate
    else:
        operator = sm.stev(*key) * rate

    return operator

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

# only stable isotopes with nonzero spins
spins = all_spins[(all_spins['spin'] > 0) & (all_spins['conc'] > 0)]

_names = spins['nucleons'].astype(str) + spins['symbol']
_gyros = spins['g'] / HBAR_SI * NUCLEAR_MAGNETON / 1e7
_quads = spins['q']
_spins = spins['spin']

_mi = pd.MultiIndex.from_arrays([spins['symbol'], _names])
_ser = pd.Series((spins['conc'] / 100).values, index=_mi)

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
# allias for common_isotopes
ci = common_isotopes
