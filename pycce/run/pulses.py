import copy
import warnings
from collections import UserList, UserDict

import numpy as np
import scipy.linalg


class BasePulse:
    """
    Base class for Pulse.

    Args:
        x (float): Rotation angle about x-axis in radians.
        y (float): Rotation angle about y-axis in radians.
        z (float): Rotation angle about z-axis in radians.
    """
    axes = 'xyz'
    indices = np.arange(3)
    indices.flags.writeable = False

    def __init__(self, x=None, y=None, z=None):

        self.__angles = np.zeros(3, dtype=np.float64)
        self.has_angle = np.zeros(3, dtype=np.float64)
        self._naxes = None
        self._flip = False

        self.x = x
        self.y = y
        self.z = z

    def set_angle(self, axis, angle):
        """
        Set rotation angle ``angle`` about axis ``axis``.

        Args:
            axis (str): Axis of the rotation.
            angle (float): Rotation angle in radians.

        Returns:

        """
        if axis not in self.axes:
            raise ValueError(f'Wrong axis format: {axis}')
        ind = _rot[axis]
        angle = _check_angle(angle)

        self._angles[ind] = angle
        self.has_angle = self.__angles.astype(bool)
        self._flip = None
        self._naxes = None

    def check_flip(self):
        """
        Check if the rotation is about single cartesian axis by an angle :math:`\pi`.
        """
        self._flip = (self.naxes == 1) and (np.isclose(self._angles[self.has_angle], np.pi))

    @property
    def naxes(self):
        """
        int: Number of axes the rotation is defined for.
        """
        if self._naxes is None:
            self._naxes = self.has_angle.sum()
        return self._naxes

    @property
    def flip(self):
        """bool: True if the angle == pi."""
        if self._flip is None:
            self.check_flip()
        return self._flip

    @property
    def _angles(self):
        return self.__angles

    @_angles.setter
    def _angles(self, angles):
        self.__angles[:] = [_check_angle(a) for a in angles]
        self.has_angle = self.__angles.astype(bool)
        self._flip = None
        self._naxes = None

    @property
    def x(self):
        """float: Angle of rotation of the spin about x axis in rad."""
        return self._angles[0]

    @x.setter
    def x(self, x):
        self.set_angle('x', x)

    @property
    def y(self):
        """float: Angle of rotation of the spin about y axis in rad."""
        return self._angles[1]

    @y.setter
    def y(self, y):
        self.set_angle('y', y)

    @property
    def z(self):
        """float: Angle of rotation of the spin about z axis in rad."""
        return self._angles[2]

    @z.setter
    def z(self, z):
        self.set_angle('z', z)

    def __repr__(self):
        inner_message = '(' + ', '.join(f'{x}: {getattr(self, x):.2f}' for x in 'xyz') + ')'
        return inner_message

    def generate_rotation(self, spinvec, spin_half=False):
        """
        Generate rotation matrix given spin vector.

        Args:
            spinvec (ndarray with shape (3, n, n)): Spin vector.
            spin_half (bool): True if spin vector is for a spin-1/2. Default is False.

        Returns:
            ndarray with shape (n, n): Rotation operator.
        """
        if spin_half and self.flip:
            return -1j * 2 * spinvec[self.indices[self.has_angle][0]]
        if self.naxes == 1:
            ind = self.indices[self.has_angle][0]
            return scipy.linalg.expm(-1j * spinvec[ind] * self._angles[ind])

        na = np.newaxis
        return scipy.linalg.expm(-1j * (spinvec[self.has_angle] * self._angles[self.has_angle][:, na, na]).sum(axis=0))


class Pulse(BasePulse, UserDict):
    """
    Class containing properties of each control pulse, applied to the system.

    The properties of the pulse, applied on the central spin(s) can be accessed as attributes, while bath spin
    pulses can be acessed as elements of the ``Pulse`` instance.

    Args:

        axis (str): Axis of rotation of the central spin. Can be 'x', 'y', or 'z'. Default is None.

        angle (float or str): Angle of rotation of central spin. Can be provided in rad, or as a string, containing
            fraction of pi: ``'pi'``, ``'pi/2'``, ``'2*pi'`` etc. Default is None.

        delay (float or ndarray): Delay before the pulse or array of delays with the same shape as time points.
            Default is None.

        which (array-like): Indexes of the central spins to be rotated by the pulse. Default is all.
            Separated indexes are supported only if qubit states are provided separately for all
            center spins.

        bath_names (str or array-like of str): Name or array of names of bath spin types, impacted by the bath pulse.
            Default is None.

        bath_axes (str or array-like of str): Axis of rotation or array of axes of the bath spins.
            Default is None. If ``bath_names`` is provided, but ``bath_axes`` and ``bath_angles`` are not,
            assumes the same axis and angle as the one of the central spin

        bath_angles (float or str or array-like): Angle of rotation or array of axes of rotations of the bath spins.

        x (float): Rotation angle of the central spin about x-axis in radians.

        y (float): Rotation angle of the central spin about y-axis in radians.

        z (float): Rotation angle of the central spin about z-axis in radians.

    Examples:

        >>> Pulse('x', 'pi')
        Pulse((x: 3.14, y: 0.00, z: 0.00))
        >>> Pulse('x', 'pi', bath_names=['13C', '14C'])
        Pulse((x: 3.14, y: 0.00, z: 0.00), {13C: (x: 3.14, y: 0.00, z: 0.00), 14C: (x: 3.14, y: 0.00, z: 0.00)})
        >>> import numpy as np
        >>> p = Pulse('x', 'pi', delay=np.linspace(0, 1, 5), bath_names=['13C', '14C'],
        >>>           bath_axes='x', bath_angles='pi/2')
        >>> print(p)
        Pulse((x: 3.14, y: 0.00, z: 0.00), {13C: (x: 1.57, y: 0.00, z: 0.00), 14C: (x: 1.57, y: 0.00, z: 0.00)},
        t = [0.   0.25 0.5  0.75 1.  ])
        >>> print(p['13C'])
        (x: 1.57, y: 0.00, z: 0.00)
    """

    def __init__(self, axis=None, angle=None, delay=None, which=None,
                 bath_names=None, bath_axes=None, bath_angles=None, **kwargs):
        super().__init__(**kwargs)
        self.data = {}

        if angle is not None:
            angle = _check_angle(angle)

        if axis is not None:
            self.set_angle(axis, angle)

        elif angle is not None:
            angle = np.asarray(angle)
            if angle.size == 3:
                self._angles = angle

        self._has_delay = False
        self.delay = delay
        """ndarray or float: Delay or array of delays before the pulse."""
        self.which = None
        """iterable: Indexes of the central spins to be rotated by the pulse."""

        if which is not None:
            self.which = np.array(which).reshape(-1)

        if bath_names is not None:
            bath_names = np.array(bath_names).reshape(-1)

            use_axes = False
            if bath_axes is not None:
                bath_axes = np.array(bath_axes).reshape(-1)
                use_axes = True

            if bath_angles is not None:
                bath_angles = np.array(bath_angles)

                if (bath_axes is None) and (bath_angles.ndim > 1 or (bath_angles.size == 3)):
                    if bath_names.size == 3 and bath_angles.size == 3:
                        warnings.warn('Ill defined bath angles format. Assuming same vector of angles for all spins.')
                    try:
                        bath_angles = bath_angles.reshape(-1, 3)
                    except ValueError:
                        raise ValueError('Wrong bath angles format')
                elif axis is not None:
                    use_axes = True
                    bath_axes = np.array(axis).reshape(-1)
                else:
                    use_axes = True
                    bath_angles = bath_angles.reshape(-1)

            else:
                bath_angles = np.array(angle).reshape(-1)

                if bath_angles.size == 3:
                    use_axes = False
                    bath_angles = bath_angles.reshape(-1, 3)
                else:
                    use_axes = True
                    bath_axes = axis
            if use_axes:
                for n, x, a in np.broadcast(bath_names, bath_axes, bath_angles):
                    self[n] = BasePulse()
                    if x is not None:
                        self[n].set_angle(x, a)
            else:
                for n, a in zip(bath_names, np.tile(bath_angles, [bath_names.size, 1])):
                    self[n] = BasePulse()
                    self[n]._angles = a

        self.bath_names = bath_names
        """ndarray: Array of names of bath spin types, impacted by the bath pulse."""

        self.bath_axes = bath_axes
        """ndarray: Array of axes of rotation of the bath spins."""

        self.bath_angles = bath_angles
        """ndarray: Array of angles of rotation of the bath spins."""

        self.rotation = None
        """ndarray: Matrix representation of the pulse for the given cluster. Generated by ``Run`` object."""

    @property
    def delay(self):
        """float or ndarray:  Delay before the pulse or array of delays with the same shape as time points."""
        return self._delay

    @delay.setter
    def delay(self, value):
        self._delay = value
        if value is not None:
            self._delay = np.asarray(value)
            self._has_delay = True
        else:
            self._has_delay = False

    def __repr__(self):

        w = f'Pulse('
        inner_message = ''

        if self.has_angle.any() is not None:
            if self.which is not None:
                inner_message += f'{self.which}: '

            x = super().__repr__()
            inner_message += x

        if self.bath_names is not None:

            bm = ''
            for k in self:
                if bm:
                    bm += ', '
                bm += f'{k}: ' + self[k].__repr__()
            if inner_message:
                inner_message += ', '
            inner_message += '{' + bm + '}'

        if self.delay is not None:
            if inner_message:
                inner_message += ', '
            inner_message += f't = {self.delay}'

        w += inner_message + ')'
        return w


_rot = {'x': 0, 'y': 1, 'z': 2}


def _check_angle(angle):
    pi = np.pi
    if angle is None:
        angle = 0
    elif isinstance(angle, str):
        angle = eval(angle)

    if np.isclose(angle, 0):
        angle = 0

    return angle


def _get_pulse(value):
    if isinstance(value, Pulse):
        value = copy.deepcopy(value)
    else:
        try:
            value = Pulse(**value)
        except TypeError:
            value = Pulse(*value)
    return value


class Sequence(UserList):
    """
    List-like object, which contains the sequence of the pulses.

    Each element is a ``Pulse`` instance, which can be generated from either the tuple with positional arguments
    or from the dictionary, or set manually.

    If delay is not provided in **all** pulses in the sequence, assume equispaced pulse sequence:

        t - pulse - 2t - pulse - 2t - ... - pulse - t

    If only **some** delays are provided, assumes 0 delay in the pulses without delay provided.


    Examples:
        >>> import numpy as np
        >>> Sequence([('x', np.pi, 0),
        >>>           {'axis': 'y', 'angle': 'pi', 'delay': np.linspace(0, 1, 3), 'bath_names': '13C'},
        >>>           Pulse('x', 'pi', 1)])
        [Pulse((x, 3.14), t = 0), Pulse((y, 3.14), {13C: (y, 3.14)}, t = [0.  0.5 1. ]), Pulse((x, 3.14), t = 1)]
    """

    def __init__(self, t=None):
        if t is not None:
            t = [_get_pulse(x) for x in t]

        super().__init__(t)

        # self.delays = None
        # """list or None: List with delays before each pulse or None if equispaced.
        # Generated by ``.generate_pulses`` method."""
        # self.rotations = None
        # """list: List with matrix representations of the rotation from each pulse.
        # Generated by ``.generate_pulses`` method."""

    def __setitem__(self, key, value):
        self.data[key] = _get_pulse(value)

    def append(self, item):
        self.data.append(_get_pulse(item))
