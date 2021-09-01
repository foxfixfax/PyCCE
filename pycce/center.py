from collections.abc import MutableSequence

import numpy as np

from pycce.bath.array import check_gyro
from pycce.constants import ELECTRON_GYRO
from pycce.utilities import zfs_tensor


def attr_arr_setter(self, attr, value, dtype=np.float64):
    if getattr(self, attr) is not None:
        obj = getattr(self, attr)
        obj[...] = np.asarray(value, dtype=dtype)
    else:
        setattr(self, attr, np.asarray(value, dtype=dtype))


class Center:
    def __init__(self, position=None,
                 spin=0, D=0, E=0,
                 gyro=ELECTRON_GYRO):
        if position is None:
            position = [0, 0, 0]

        self._zfs = None
        self._gyro = None
        self._xyz = None
        self._s = None

        self.xyz = position
        self.s = spin
        self.set_zfs(D, E)
        self.set_gyro(gyro)

    @property
    def xyz(self):
        return self._xyz

    @xyz.setter
    def xyz(self, position):
        attr_arr_setter(self, '_xyz', position)

    @property
    def gyro(self):
        return self._gyro

    @gyro.setter
    def gyro(self, gyro):
        attr_arr_setter(self, '_gyro', gyro)

    @property
    def zfs(self):
        return self._zfs

    @zfs.setter
    def zfs(self, zfs):
        attr_arr_setter(self, '_zfs', zfs)

    @property
    def s(self):
        return self._s[()]

    @s.setter
    def s(self, spin):
        attr_arr_setter(self, '_s', spin)

    def set_zfs(self, D=0, E=0):
        """
         Set Zero Field Splitting of the central spin from longitudinal ZFS *D* and transverse ZFS *E*.

        Args:
            D (float or ndarray with shape (3, 3)): D (longitudinal splitting) parameter of central spin
                in ZFS tensor of central spin in kHz.

                **OR**

                Total ZFS tensor. Default 0.

            E (float): E (transverse splitting) parameter of central spin in ZFS tensor of central spin in kHz.
                 Default 0. Ignored if ``D`` is None or tensor.
        """
        self.zfs = zfs_tensor(D, E)
        #
        # else:
        #     for i, d, e in enumerate(zip(np.broadcast_to(D, self.shape), np.broadcast_to(E, self.shape))):
        #         self.zfs[i] = zfs_tensor(D, E)

    def set_gyro(self, gyro):
        """
        Set gyromagnetic ratio of the central spin.

        Args:
            gyro ():

        Returns:

        """
        check = not np.asarray(gyro).shape == (3, 3)
        if check:
            gyro, check = check_gyro(gyro)

        if check:
            self.gyro = np.eye(3) * gyro
        else:
            self.gyro = gyro

        # else:
        #     for i, g in enumerate(np.broadcast_to(gyro, self.shape)):
        #
        #         g, check = check_gyro(g)
        #
        #         if check:
        #             self.gyro[i] = np.eye(3) * g
        #         else:
        #             self.gyro[i] = g

    @property
    def alpha(self):
        r"""
        ndarray or int: :math:`\ket{0}` qubit state of the central spin in Sz basis
            **OR** index of the energy state to be considered as one.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, alpha_state):
        try:
            if len(alpha_state) > 1:
                self._alpha = np.asarray(alpha_state, dtype=np.complex128)
            else:
                self._alpha = np.asarray(alpha_state, dtype=np.int32)
        except TypeError:
            self._alpha = np.asarray(alpha_state, dtype=np.int32)

    @property
    def beta(self):
        r"""
        ndarray or int: :math:`\ket{1}` qubit state of the central spin in Sz basis
            **OR** index of the energy state to be considered as one.
        """
        return self._beta

    @beta.setter
    def beta(self, beta_state):

        try:
            if len(beta_state) > 1:
                self._beta = np.asarray(beta_state, dtype=np.complex128)
            else:
                self._beta = np.int(beta_state)

        except TypeError:
            self._beta = np.int(beta_state)


class CenterList(Center, MutableSequence):
    def __init__(self, size, position=None,
                 spin=0, D=0, E=0,
                 gyro=ELECTRON_GYRO, imap=None):
        self.size = size
        if position is None:
            position = [[0, 0, 0]] * self.size

        spin = np.asarray(spin).reshape(-1)
        if spin.size != self.size:
            spin = np.array(np.broadcast_to(spin, self.size))

        self._alpha = None
        self._beta = None
        self._state = None

        super().__init__(position=position, spin=spin, D=D, E=E, gyro=gyro)
        self._list = [Center(position=p, spin=s[..., 0], D=zfs, gyro=g) for p, s, zfs, g in
                      zip(self.xyz, self.s[:, np.newaxis], self.zfs, self.gyro)]

        self.imap = imap

    @property
    def state(self):
        if self._state is not None:
            return self._state
        else:
            return

    @state.setter
    def state(self, state):

        raise NotImplementedError

    def __getitem__(self, item):
        return self._list.__getitem__(item)

    def __setitem__(self, key, val):
        if not isinstance(val, Center):
            raise ValueError
        self._list.__setitem__(key, val)

    def __delitem__(self, key):
        raise NotImplementedError

    def __len__(self):
        return self.size

    def insert(self, index: int, value) -> None:
        raise NotImplementedError

    def set_zfs(self, D=0, E=0):
        """
         Set Zero Field Splitting of the central spin from longitudinal ZFS *D* and transverse ZFS *E*.

        Args:
            D (float or ndarray with shape (3, 3)): D (longitudinal splitting) parameter of central spin
                in ZFS tensor of central spin in kHz.

                **OR**

                Total ZFS tensor. Default 0.

            E (float): E (transverse splitting) parameter of central spin in ZFS tensor of central spin in kHz.
                 Default 0. Ignored if ``D`` is None or tensor.
        """
        if self.zfs is None:
            self.zfs = np.zeros((self.size, 3, 3), dtype=np.float64)
        for i, (d, e) in enumerate(zip(np.broadcast_to(D, self.size), np.broadcast_to(E, self.size))):
            self.zfs[i] = zfs_tensor(d, e)

    def set_gyro(self, gyro):
        """
        Set gyromagnetic ratio of the central spin.

        Args:
            gyro ():

        Returns:

        """
        if self.gyro is None:
            self.gyro = np.zeros((self.size, 3, 3), dtype=np.float64)

        for i, g in enumerate(np.broadcast_to(gyro, self.size)):

            g, check = check_gyro(g)

            if check:
                self.gyro[i] = np.eye(3) * g
            else:
                self.gyro[i] = g
#
# class CenterArray:
#     _dtype_center = np.dtype([('N', np.unicode_, 16),
#                               ('s', np.float64),
#                               ('xyz', np.float64, (3,)),
#                               ('D', np.float64, (3, 3)),
#                               ('gyro', np.float64, (3, 3))])
#
#     def __init__(self, shape=None, position=None,
#                  spin=None, D=None, E=0,
#                  gyro=ELECTRON_GYRO, imap=None):
#
#         if shape is None:
#             raise ValueError('No shape provided')
#
#         self.shape = shape
#         self.size = shape
#
#         self.indexes = np.arange(shape)
#         self.xyz = np.zeros((self.size, 3))
#         self.s = np.zeros(self.size)
#         self.gyro = np.zeros((self.size, 3, 3))
#         self.zfs = np.zeros((self.size, 3, 3))
#
#         self.xyz = position
#         self.spin = spin
#         self.set_gyro(gyro)
#         self.set_zfs(D, E)
#
#         self.imap = imap
#
#         self._state = None
#         self._alpha = None
#         self._beta = None
#
#         self._alpha_list = None
#         self._beta_list = None
#
