from collections.abc import Sequence

import numpy as np

from pycce.bath.array import check_gyro, BathArray
from pycce.constants import ELECTRON_GYRO
from pycce.h.total import total_hamiltonian
from pycce.utilities import zfs_tensor, generate_projections


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

        self.projections_alpha = None
        self.projections_beta = None

        # You cannot initialize these from center, only from CenterArray
        self.projections_alpha_all = None
        self.projections_beta_all = None

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
                gyro = np.eye(3) * gyro

        self.gyro = gyro

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

    @property
    def dim(self):
        return (self._s * 2 + 1 + 1e-8).astype(int)[()]


class CenterArray(Center, Sequence):
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

        self.energies = None
        self.eigenvectors = None

        self.hamiltonian = None

        self.energy_alpha = None
        self.energy_beta = None
        self.energies = None

        self.run_alpha = None
        self.run_beta = None

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

        self.zfs[key] = val.zfs
        self.gyro[key] = val.gyro
        self.s[key] = val.s
        self.xyz[key] = val.xyz

        center = self._list.__getitem__(key)

        center.alpha = val.alpha
        center.beta = val.beta

    def __len__(self):
        return self.size

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

    def generate_hamiltonian(self, magnetic_field=None, bath=None, projected_bath_state=None):

        self.hamiltonian = total_hamiltonian(BathArray((0,)), magnetic_field, central_spin=self, others=bath,
                                             other_states=projected_bath_state, )

        self.energies, self.eigenvectors = np.linalg.eigh(self.hamiltonian)

        alpha = self.alpha
        beta = self.beta

        if (not alpha.shape) or (not beta.shape):

            alpha = self.eigenvectors[:, alpha]
            beta = self.eigenvectors[:, beta]

            state = (alpha + beta) / np.linalg.norm(alpha + beta)

            self.run_alpha = alpha
            self.run_beta = beta

        else:
            state = self.state
        self.run_alpha = alpha
        self.run_beta = beta

        self.state = state

    #
    def generate_projections(self, second_order=False, level_confidence=0.95):
        if second_order:
            ai = _close_state_index(self.alpha, self.eigenvectors, level_confidence=level_confidence)
            bi = _close_state_index(self.beta, self.eigenvectors, level_confidence=level_confidence)

            alpha = self.eigenvectors[:, ai]
            beta = self.eigenvectors[:, bi]

            self.energy_alpha = self.energies[ai]
            self.energy_beta = self.energies[bi]

            self.energies = self.energies

            gp = generate_projections
            self.projections_alpha_all = np.array([gp(alpha, s, spins=self.s) for s in self.eigenvectors.T])
            self.projections_beta_all = np.array([gp(beta, s, spins=self.s) for s in self.eigenvectors.T])

            for i, center in enumerate(self):
                center.projections_alpha_all = self.projections_alpha_all[:, i]
                center.projections_beta_all = self.projections_beta_all[:, i]

        else:

            self.energy_alpha = None
            self.energy_beta = None
            self.energies = None

            self.projections_alpha_all = None
            self.projections_beta_all = None

        self.projections_alpha = np.array(generate_projections(self.alpha, spins=self.s))
        self.projections_beta = np.array(generate_projections(self.beta, spins=self.s))

        for i, center in enumerate(self):
            center.projections_alpha = self.projections_alpha[i]
            center.projections_beta = self.projections_beta[i]


def _close_state_index(state, eiv, level_confidence=0.95):
    """
    Get index of the eigenstate stored in eiv,
    which has fidelity higher than ``level_confidence`` with the provided ``state``.

    Args:
        state (ndarray with shape (2s+1,)): State for which to find the analogous eigen state.
        eiv (ndarray with shape (2s+1, 2s+1)): Matrix of eigenvectors as columns.
        level_confidence (float): Threshold fidelity. Default 0.95.

    Returns:
        int: Index of the eigenstate.
    """
    indexes = np.argwhere((eiv.T @ state) ** 2 > level_confidence).flatten()

    if not indexes.size:
        raise ValueError(f"Initial qubit state is below F = {level_confidence} "
                         f"to the eigenstate of central spin Hamiltonian.\n"
                         f"Qubit level:\n{repr(state)}"
                         f"Eigenstates (rows):\n{repr(eiv.T)}")
    return indexes[0]

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
