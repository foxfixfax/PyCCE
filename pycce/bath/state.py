import warnings

import numpy as np
from pycce.utilities import project_bath_states


class BathState:
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
        # Only allowed values are density matrices or None
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

            try:
                value = np.asarray(value, dtype=np.complex128)
            except ValueError:
                value = np.asarray(value, dtype=np.object_)

            if not value.dtype == np.object_:

                inshape = np.asarray(self.state[key]).shape

                if not inshape:

                    self.state[key][...] = value
                    self.has_state[key] = True
                    self.pure = len(value.shape) < 2

                else:
                    if len(value.shape) == 2:

                        v1 = value.shape[0]
                        v2 = value.shape[1]
                        if v1 == v2:
                            # Matrix
                            if v1 == inshape[0]:
                                if np.isclose(np.trace(value), 1):
                                    # Density matrix b/c trace is 1
                                    # warnings.warn("Cannot decide whether a stack of vectors or density matrix. "
                                    #               "Assuming density matrix", stacklevel=2)
                                    fshape = inshape + value.shape
                                else:
                                    # Else array of vectors
                                    fshape = value.shape
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

                    broadcasted = np.broadcast_to(value, fshape)
                    self.state[key] = [rho for rho in broadcasted]
                    self.has_state[key] = True  # [rho is not None for rho in broadcasted] Cannot be None

            else:
                self.state[key] = [rho for rho in value]
                self.has_state[key] = [rho is not None for rho in value]
                self.pure[key] = [(len(rho.shape) == 1) if rho is not None else False for rho in value]

    @property
    def state(self):
        return self._data['state']

    @property
    def pure(self):
        return self._data['pure']

    @pure.setter
    def pure(self, pure):
        self._data['pure'] = pure

    @property
    def _up_to_date(self):

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
        if not self._up_to_date:
            self._project()

        return self._data['proj']

    @property
    def has_state(self):

        # if self._in_need_of_checking.any():
        #     self._has_state[self._in_need_of_checking] = False
        return self._data['hs']

    def project(self):
        self._data['hps'] = False
        self._project()

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
        return self._data.shape

    @property
    def size(self):
        return self._data.size

    def any(self, *args, **kawrgs):

        return self.has_state.any(*args, **kawrgs)

    def __repr__(self):
        return f"{self.__class__.__name__}(" + self.state.__str__() + ")"

    def __str__(self):
        return self.state.__str__()


class State:
    def __init__(self):
        self.pure = None
        self.dm = None
        self.is_pure = None
