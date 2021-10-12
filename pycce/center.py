from collections.abc import Sequence

import numpy as np
from pycce.bath.array import check_gyro, point_dipole, BathArray
from pycce.bath.map import InteractionMap
from pycce.constants import ELECTRON_GYRO
from pycce.h.total import central_hamiltonian
from pycce.utilities import zfs_tensor, generate_projections, expand


def attr_arr_setter(self, attr, value, dtype=np.float64):
    if getattr(self, attr) is not None:
        obj = getattr(self, attr)
        obj[...] = np.asarray(value, dtype=dtype)
    else:
        setattr(self, attr, np.asarray(value, dtype=dtype))


class Center:

    r"""

        energy_alpha (

        energy_beta (float): Energy of the beta state

        energies (ndarray with shape (2s-1,)): Array of energies of all states of the central spin.



    """
    def __init__(self, position=None,
                 spin=0, D=0, E=0,
                 gyro=ELECTRON_GYRO, alpha=None, beta=None):
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
        r"""
            ndarray with shape (2s-1, 3):
                Array of vectors of the central spin matrix elements of form:
    
                .. math::
    
                    [\bra{\alpha}\hat{S}_x\ket{j}, \bra{\alpha}\hat{S}_y\ket{j}, \bra{\alpha}\hat{S}_z\ket{j}],
    
                where :math:`\ket{\alpha}` is the alpha qubit state, and :math:`\ket{\j}` are all states.
        """

        self.projections_beta_all = None
        r"""
            ndarray with shape (2s-1, 3):
                Array of vectors of the central spin matrix elements of form:
    
                .. math::
    
                    [\bra{\beta}\hat{S}_x\ket{j}, \bra{\beta}\hat{S}_y\ket{j}, \bra{\beta}\hat{S}_z\ket{j}],
    
                where :math:`\ket{\beta}` is the beta qubit state, and :math:`\ket{\j}` are all states.
        """

        self.energies = None
        self.eigenvectors = None
        self.hamiltonian = None

        self._alpha = None
        self._beta = None

        self.alpha_index = None
        self.beta_index = None

        self.alpha = alpha
        self.beta = beta

        self.sigma = None

    @property
    def xyz(self):
        """ndarray with shape (3, ): Position of the central spin in Cartesian coordinates."""
        return self._xyz

    @xyz.setter
    def xyz(self, position):
        attr_arr_setter(self, '_xyz', position)

    @property
    def gyro(self):
        """
        ndarray with shape (3,3) or (n,3,3): Tensor describing central spin interactions
            with the magnetic field or array of spins.

            Default -17608.597050 rad / ms / G - gyromagnetic ratio of the free electron spin."""

        return self._gyro

    @gyro.setter
    def gyro(self, gyro):
        attr_arr_setter(self, '_gyro', gyro)

    @property
    def zfs(self):
        """ndarray with shape (3,3) or (n,3,3): Zero field splitting tensor of the central spin or array of spins."""
        return self._zfs

    @zfs.setter
    def zfs(self, zfs):
        attr_arr_setter(self, '_zfs', zfs)

    @property
    def s(self):
        """float or ndarray with shape (n,): Total spin of the central spin or array of spins."""
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
        return self._get_state('alpha')

    @alpha.setter
    def alpha(self, state):
        self._set_state('alpha', state)

    @property
    def beta(self):
        r"""
        ndarray or int: :math:`\ket{1}` qubit state of the central spin in Sz basis
            **OR** index of the energy state to be considered as one.
        """
        return self._get_state('beta')

    @beta.setter
    def beta(self, state):
        self._set_state('beta', state)

    @property
    def dim(self):
        return (self._s * 2 + 1 + 1e-8).astype(int)[()]

    def generate_sigma(self):
        r"""
        Set Pauli matrices of the central spin.
        """
        assert np.isclose(np.inner(self.alpha, self.beta.conj()), 0), \
            f"Pauli matrix can be generated only for orthogonal states, " \
            f"{self.alpha} and {self.beta} are not orthogonal"

        alpha_x_alpha = np.outer(self.alpha, self.alpha)
        beta_x_beta = np.outer(self.beta, self.beta)
        alpha_x_beta = np.outer(self.alpha, self.beta)
        beta_x_alpha = np.outer(self.beta, self.alpha)

        self.sigma = {'x': alpha_x_beta + beta_x_alpha,
                      'y': -1j * alpha_x_beta + 1j * beta_x_alpha,
                      'z': alpha_x_alpha - beta_x_beta}

    def _set_state(self, name, state):
        if state is not None:
            state = np.asarray(state)
            if state.size == 1:
                setattr(self, name + '_index', int(state))
            else:
                assert state.size == np.prod(self.dim), f"Incorrect format of {name}: {state}"
                setattr(self, '_' + name, state.astype(np.complex128))
                # remove index if manually set alpha state
                setattr(self, name + '_index', None)
        else:
            setattr(self, '_' + name, state)

    def _get_state(self, name):
        state = getattr(self, '_' + name)
        if state is not None:
            return state

        state = getattr(self, name + '_index')
        if state is not None:
            return state

    def generate_states(self, magnetic_field=None, bath=None, projected_bath_state=None):
        """
        Compute eigenstates of the central spin Hamiltonian.

        Args:
            magnetic_field (ndarray with shape (3,)): Array containing external magnetic field as (Sx, By, Bz).
            bath (BathArray with shape (m,) or ndarray with shape (m,3,3):
                Array of all bath spins or array of hyperfine tensors.
            projected_bath_state (ndarray with shape (m,) or (m, 3)):
                Array of Iz projections for each bath spin.
        """

        if magnetic_field is None:
            magnetic_field = [0, 0, 0]
        if isinstance(bath, BathArray):
            bath = bath['A']

        self.hamiltonian = central_hamiltonian(self, magnetic_field, hyperfine=bath,
                                               bath_state=projected_bath_state)

        self.energies, self.eigenvectors = np.linalg.eigh(self.hamiltonian)

        if self.alpha_index is not None:
            self._alpha = self.eigenvectors[:, self.alpha_index]
        if self.beta_index is not None:
            self._beta = self.eigenvectors[:, self.beta_index]


class CenterArray(Center, Sequence):
    def __init__(self, size=None, position=None,
                 spin=None, D=0, E=0,
                 gyro=ELECTRON_GYRO, imap=None,
                 alpha=None,
                 beta=None):

        if size is None:
            if spin is not None:
                spin = np.asarray(spin)
                size = spin.size

            elif position is not None:
                position = np.asarray(position)
                size = position.size // 3

            else:
                raise ValueError('Size of the array is not provided')

        self.size = size

        if position is None:
            position = [[0, 0, 0]] * self.size
        if spin is None:
            spin = 0

        spin = np.asarray(spin).reshape(-1)
        if spin.size != self.size:
            spin = np.array(np.broadcast_to(spin, self.size))

        self._state = None

        super().__init__(position=position, spin=spin, D=D, E=E, gyro=gyro, alpha=alpha, beta=beta)

        self._array = np.array([Center(position=p, spin=s[..., 0], D=zfs, gyro=g) for p, s, zfs, g in
                                zip(self.xyz, self.s[:, np.newaxis], self.zfs, self.gyro)], dtype=object)

        if imap is not None and not isinstance(imap, InteractionMap):

            if self.size < 2:
                raise ValueError(f'Cannot assign interaction map for array of size {self.size}')

            imap = np.broadcast_to(imap, (self.size * (self.size - 1) // 2, 3, 3))
            imap = InteractionMap(rows=np.arange(self.size), columns=np.arange(1, self.size), tensors=imap)

        self._imap = imap

        self.energy_alpha = None
        """float: Energy of the alpha state"""
        self.energy_beta = None
        self.energies = None

    @property
    def imap(self):
        if self._imap is None:
            self._imap = InteractionMap()

        return self._imap

    @property
    def state(self):
        r"""
        ndarray: Innitial state of the qubit in gCCE simulations.
            Assumed to be :math:`1/\sqrt{2}(\ket{0} + \ket{1}` unless provided."""

        if self._state is not None:
            return self._state
        else:
            self._check_states()
            a = self.alpha
            b = self.beta
            return (a + b) / np.linalg.norm(a + b)

    @state.setter
    def state(self, state):
        if state is not None:
            state = np.asarray(state, dtype=np.complex128)
            assert state.size == np.prod(self.dim), 'Wrong state format.'
        self._state = state

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._array.__getitem__(item)

        else:
            newarray = self._array.__getitem__(item)
            xyz = self.xyz[item]
            gyro = self.gyro[item]
            s = self.s[item]
            zfs = self.zfs[item]
            ca = CenterArray(len(newarray), position=xyz, gyro=gyro, spin=s, D=zfs)
            ca._array = newarray
            if self._imap is not None:
                ca._imap = self.imap.subspace(np.arange(self.size)[item])
            return ca

    def __setitem__(self, key, val):
        if not isinstance(val, Center):
            raise ValueError

        self.zfs[key] = val.zfs
        self.gyro[key] = val.gyro
        self.s[key] = val.s
        self.xyz[key] = val.xyz

        center = self._array.__getitem__(key)

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
        darr = np.asarray(D)

        if darr.shape == (self.size, 3, 3):
            self.zfs = darr

        else:
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
        garr = np.asarray(gyro)

        if garr.shape == (self.size, 3, 3):
            self.gyro = garr
        else:
            if self.gyro is None:
                self.gyro = np.zeros((self.size, 3, 3), dtype=np.float64)

            for i, g in enumerate(np.broadcast_to(gyro, self.size)):

                g, check = check_gyro(g)

                if check:
                    self.gyro[i] = np.eye(3) * g
                else:
                    self.gyro[i] = g

    def point_dipole(self):
        """
        Using point-dipole approximation, generate interaction tensors between central spins.
        """
        for i in range(self.size):
            for j in range(i + 1, self.size):
                c1 = self[i]
                c2 = self[j]
                self.imap[i, j] = point_dipole(c1.xyz - c2.xyz, c1.gyro, c2.gyro)

    def generate_states(self, magnetic_field=None, bath=None, projected_bath_state=None):

        if isinstance(bath, BathArray):
            bath = bath['A']

        for i, c in enumerate(self):
            if bath is None:
                hf = None
            elif len(self) == 1:
                hf = bath
            else:
                hf = bath[..., i, :, :]

            c.generate_states(magnetic_field=magnetic_field,
                              bath=hf, projected_bath_state=projected_bath_state)

        super(CenterArray, self).generate_states(magnetic_field=magnetic_field,
                                                 bath=bath, projected_bath_state=projected_bath_state)

    def generate_projections(self, second_order=False, level_confidence=0.95):
        self._check_states()
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

            self.projections_alpha_all = None
            self.projections_beta_all = None

        self.projections_alpha = np.array(generate_projections(self.alpha, spins=self.s))
        self.projections_beta = np.array(generate_projections(self.beta, spins=self.s))

        for i, center in enumerate(self):
            center.projections_alpha = self.projections_alpha[i]
            center.projections_beta = self.projections_beta[i]

    def generate_sigma(self):
        self._check_states()
        super(CenterArray, self).generate_sigma()
        for i, c in enumerate(self):
            if c.alpha is not None and c.beta is not None:
                try:
                    c.generate_sigma()
                    for x in c.sigma:
                        c.sigma[x] = expand(c.sigma[x], i, self.dim)
                except AssertionError:
                    pass

    def _check_states(self):
        for n in ['alpha', 'beta']:
            s = getattr(self, n)
            if s is None or isinstance(s, int):
                raise ValueError(f'Wrong {n} format: {s}')

    def _get_state(self, name):
        state = getattr(self, '_' + name)
        if state is not None:
            return state

        state = getattr(self, name + '_index')
        if state is not None:
            return state

        state = 1
        for c in self:
            s = getattr(c, name)
            assert s is not None, f"{name} is not provided for array and is not provided for all separate spins."
            state = np.kron(state, s)

        return state


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
