import numpy as np
from numba import jit
from numba.typed import List
from pycce.sm import _smc
from pycce.utilities import gen_state_list, vector_from_s


class BathState:
    r"""
    Class for storing the state of the bath spins. Usually is not generated directly, but accessed as
    an ``BathArray.state`` attribute.

    Args:
        size (int): Number of bath states to be stored.
    """
    _dtype_state = np.dtype([
        ('proj', np.float64),
        ('state', object),
        ('hs', bool),
        ('hps', bool),
        ('pure', bool)
    ])

    def __init__(self, size):

        self._data = np.zeros(size, dtype=self._dtype_state)
        self._data['state'] = None

        self.__up_to_date = None

    def __getitem__(self, item):

        if isinstance(item, tuple):
            if Ellipsis in item:
                # self.__in_need_of_checking[item] = True
                return self.state.__getitem__(item)

            try:
                key = item[0]
                within = item[1:]
                # self.__in_need_of_checking[key] = True
                return self.state.__getitem__(key)[within]

            except IndexError:
                pass

        # self.__in_need_of_checking[item] = True
        return self.state.__getitem__(item)

    def __setitem__(self, key, value):
        # Only allowed values are density matrices, vectors, or None
        self._up_to_date = False
        self._data['hps'][key] = False

        if value is None:
            self.state[key] = None
            self.has_state[key] = False
            self.pure[key] = False

            return

        if isinstance(key, tuple):
            within = key[1:]

            if within:
                self.state[key[0]][within] = value
                self.has_state[key[0]] = True

                n = len(np.asarray(key[0]).shape)

                if n:
                    self.pure[key[0]] = self.state[key[0]][0].shape < 2
                else:
                    self.pure[key[0]] = self.state[key[0]].shape < 2

                return

            else:
                try:
                    key = key[0]
                except IndexError:
                    pass

        if isinstance(key, int):

            self.state[key] = value
            self.has_state[key] = np.bool_(value).any()
            self.pure[key] = len(self.state[key].shape) < 2

        else:

            if isinstance(value, BathState):
                self.state[key] = [rho for rho in value]
                self.has_state[key] = value.has_state
                self.pure[key] = value.pure

                return

            if not isinstance(value, np.ndarray):
                try:
                    value = np.asarray(value, dtype=np.complex128)

                except (ValueError, TypeError):
                    value = np.asarray(value, dtype=np.object_)

            if not value.dtype == np.object_:

                inshape = np.asarray(self.state[key]).shape

                if not inshape:

                    self.state[key][...].reshape(-1)[0] = value
                    self.has_state[key] = True
                    self.pure = len(value.shape) < 2

                else:
                    if len(value.shape) == 2:

                        v1 = value.shape[0]
                        v2 = value.shape[1]
                        if v1 == v2:
                            # Matrix
                            if v1 == inshape[0]:
                                if np.isclose(np.linalg.norm(value, axis=1), 1).all():
                                    # Array of vectors
                                    fshape = value.shape

                                elif np.isclose(np.trace(value), 1):
                                    # Density matrix b/c trace is 1
                                    # warnings.warn("Cannot decide whether a stack of vectors or density matrix. "
                                    #               "Assuming density matrix", stacklevel=2)
                                    fshape = inshape + value.shape
                                else:
                                    raise ValueError('Something is really wrong with this world.')
                                    # Else array of vectors

                            else:
                                # Still matrix
                                fshape = inshape + value.shape

                            self.pure[key] = False
                        else:
                            # Array of vectors
                            self.pure[key] = True
                            fshape = value.shape

                    elif len(value.shape) == 1:
                        # Vector
                        self.pure[key] = True
                        fshape = inshape + value.shape
                    else:
                        # Matrix
                        fshape = value.shape
                        self.pure[key] = False

                    broadcasted = np.array(np.broadcast_to(value, fshape))
                    self.state[key] = objarr(broadcasted)
                    self.has_state[key] = True  # [rho is not None for rho in broadcasted] Cannot be None

            else:
                self.state[key] = value
                self.has_state[key] = [rho is not None for rho in value]
                self.pure[key] = [(len(rho.shape) == 1) if rho is not None else False for rho in value]

    def gen_pure(self, rho, dim):
        """
        Generate pure states from the :math:`S_z` projections to be stored in the given ``BathState`` object.

        Args:
            rho (ndarray with shape (n, )): Array of the desired projections.
            dim (ndarray with shape (n,) ): Array of the dimensions of the spins.

        Returns:
            BathState: View of the ``BathState`` object.
        """
        rho = np.asarray(rho)

        if not rho.shape:
            # assume s is int or float showing the spin projection in the pure state
            d = dim
            if not self.shape:
                rho = vector_from_s(rho, d)
            else:
                if (d == d[0]).all():
                    d = d[0]
                    rho = np.broadcast_to(vector_from_s(rho, d), self.shape + (d,))
                else:
                    rho = [vector_from_s(rho, d_i) for d_i in d]
        else:
            rho = gen_state_list(rho, np.broadcast_to(dim, self.shape))
        self[...] = rho

        return self

    @property
    def state(self):
        """
        ndarray: Return an underlying object array.
        """
        return self._data['state']

    @property
    def pure(self):
        """
        ndarray: Bool property. True if given entry is a pure state, False otherwise.
        """
        return self._data['pure']

    @pure.setter
    def pure(self, pure):

        self._data['pure'] = pure

    @property
    def _up_to_date(self):
        """
        bool: True if projections are up to date, False otherwise.
        """
        if self.__up_to_date is None:
            self.__up_to_date = np.asarray(False)

        return self.__up_to_date

    @_up_to_date.setter
    def _up_to_date(self, item):

        if self.__up_to_date is None:
            self.__up_to_date = np.asarray(False)

        self.__up_to_date[...] = item

    @property
    def proj(self):
        """
        ndarray: Projections of bath states on :math:`S_z`.
        """
        if not self._up_to_date:
            self._project()

        return self._data['proj']

    @property
    def has_state(self):
        """
        ndarray: Bool property. True if given element was initialized as a state, False otherwise.
        """
        # if self._in_need_of_checking.any():
        #     self._has_state[self._in_need_of_checking] = False
        return self._data['hs']

    def project(self, rotation=None):
        """
        Generate projections of bath states on :math:`S_z`.

        Args:
            rotation (optional, ndarray with shape (3, 3)):
                Matrix used to transform :math:`S_z` matrix as :math:`S_z' = R^{\dagger} S_z R`.

        Returns:
            ndarray with shape (n, ): Array with projections of the state.
        """
        if rotation is None:
            self._data['hps'] = False
            self._project()
            return self._data['proj'].copy()

        projections = np.zeros(self.shape)

        if self.has_state.any():
            projections[self.has_state] = project_bath_states(self.state[self.has_state],
                                                              (not bool(self.state[self.has_state].shape)),
                                                              rotation)
        return projections

    def _project(self):

        which = self.has_state & (~self._data['hps'])
        if which.any():
            projected = project_bath_states(self.state[which], not bool(self.state[which].shape))
            self._data['proj'][which] = projected
            self._data['hps'][which] = True

        self._up_to_date = True

    def _get_state(self, item):

        obj = self.__new__(self.__class__)
        obj._data = self._data[item]
        obj.__up_to_date = self._up_to_date

        return obj

    @property
    def shape(self):
        """
        tuple: Shape of the BathState underlying array.
        """
        return self._data.shape

    @property
    def size(self):
        """
        int: Size of the BathState underlying array.
        """
        return self._data.size

    def any(self, *args, **kawrgs):
        """
        Returns the output of ``.has_state.any`` method.
        Args:
            *args: Positional arguments of ``.has_state.any`` method.
            **kawrgs: Keyword arguments of ``.has_state.any`` method.

        Returns:
            bool: True if any entry is initialized. Otherwise False.
        """
        return self.has_state.any(*args, **kawrgs)

    def __repr__(self):
        return f"{self.__class__.__name__}(" + self.state.__str__() + ")"

    def __str__(self):
        return self.state.__str__()


def objarr(array):
    """
    Make an array with object entries from iterable.

    Args:
        array (iterable): Iterable containing elements of the future array.

    Returns:
        ndarray: Object array.
    """
    nar = len(array)
    obj = np.empty(nar, dtype=object)

    for i in range(nar):
        obj[i] = np.ascontiguousarray(array[i])
    return obj


def project_bath_states(states, single=False, rotation=None):
    r"""
    Generate projections of bath states on :math:`S_z` axis from any type of states input.

    Args:
        states (array-like): Array of bath spin states.
        single (bool): True if a single bath spin. Default False.
        rotation (optional, ndarray with shape (3, 3)):
            Matrix used to transform :math:`S_z` matrix as :math:`S_z' = R^{\dagger} S_z R`.

    Returns:
        ndarray: Array of :math:`S_z` projections of the bath states.
    """
    # Ask for single b/c check against shape cannot distinguish 2x2 dm and 2 vectors of 2
    # Other checks are kinda expensive

    ndstates = np.asarray(states)

    if not ndstates.shape and ndstates.dtype == object:
        ndstates = ndstates[()]
        single = True

    projected_bath_state = None

    if ndstates.dtype == object:

        try:
            ndstates = np.stack(ndstates)

        except ValueError:
            if rotation is None:
                projected_bath_state = _loop_trace(List(states))
            else:
                projected_bath_state = _loop_trace_rotate(List(states))

    if projected_bath_state is None:
        spin = (ndstates.shape[1] - 1) / 2
        sz = _smc[spin].z

        if rotation is not None:
            sz = rotation.conj().T @ sz @ rotation

        if len(ndstates.shape) == 2 + (not single):
            # projected_bath_state = np.empty((ndstates.shape[0], 3))

            # projected_bath_state[:, 0] = np.trace(np.matmul(ndstates, _smc[spin].x), axis1=-2, axis2=-1)
            # projected_bath_state[:, 1] = np.trace(np.matmul(ndstates, _smc[spin].y), axis1=-2, axis2=-1)
            projected_bath_state = np.trace(np.matmul(ndstates, sz), axis1=-2, axis2=-1).real  # [:, 2]

        else:
            # Assume vectors
            z_psi = np.einsum('ij,...j->...i', sz, ndstates)
            projected_bath_state = np.einsum('...j,...j->...', ndstates.conj(), z_psi).real
            # projected_bath_state = ndstates

    # if len(projected_bath_state.shape) > 1 and not np.any(projected_bath_state[:, :2]):
    #     projected_bath_state = projected_bath_state[:, 2]

    return projected_bath_state


@jit(cache=True, nopython=True)
def _loop_trace(states):
    proj_states = np.empty((len(states),), dtype=np.float64)  # (len(states), 3)
    dims = List()

    # sx = List()
    # sy = List()
    sz = List()

    for j, dm in enumerate(states):
        dm = dm.astype(np.complex128)
        dim = dm.shape[0]
        try:
            ind = dims.index(dim)
        except:
            # sxnew, synew, sznew = _gen_sm(dim)
            sznew = _gen_sz(dim)
            # sx.append(sxnew)
            # sy.append(synew)
            sz.append(sznew)
            dims.append(dim)

            ind = -1
        if len(dm.shape) == 2:
            # xproj = np.trace(dm @ sx[ind])
            # yproj = np.trace(dm @ sy[ind])
            zproj = np.diag(dm @ sz[ind]).sum().real
        else:
            # xproj = dm.conj() @ sx[ind] @ dm
            # yproj = dm.conj() @ sy[ind] @ dm
            zproj = (dm.conj() @ sz[ind] @ dm).real

        # proj_states[j, 0] = xproj
        # proj_states[j, 1] = yproj
        proj_states[j] = zproj  # [j, 2]

    return proj_states


@jit(cache=True, nopython=True)
def _loop_trace_rotate(states, rotation):
    proj_states = np.empty((len(states),), dtype=np.float64)  # (len(states), 3)
    dims = List()
    sz = List()

    for j, dm in enumerate(states):
        dm = dm.astype(np.complex128)
        dim = dm.shape[0]
        try:
            ind = dims.index(dim)

        except:

            sznew = rotation.conj().T @ _gen_sz(dim) @ rotation
            sz.append(sznew)
            dims.append(dim)
            ind = -1

        if len(dm.shape) == 2:

            dm = dm
            zproj = np.diag(dm @ sz[ind]).sum().real

        else:
            zproj = (dm.conj() @ sz[ind] @ dm).real

        proj_states[j] = zproj

    return proj_states


@jit(nopython=True)
def _gen_sz(dim):
    s = (dim - 1) / 2
    projections = np.linspace(-s, s, dim).astype(np.complex128)
    return np.diag(projections[::-1])
