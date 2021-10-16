import numpy as np

from ..utilities import project_bath_states


class BathState:
    _dtype_state = np.dtype([
        ('proj', np.float64),
        ('state', object),
        ('hs', bool),
        ('hps', bool)
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
            return

        if isinstance(key, tuple):
            within = key[1:]

            if within:
                self.state[key[0]][within] = value
                self.has_state[key[0]] = True
                return

            else:
                try:
                    key = key[0]
                except IndexError:
                    pass

        if isinstance(key, int):

            self.state[key] = value
            self.has_state[key] = np.bool_(value).any()

        else:
            try:
                self.state[key] = value
                self.has_state[key] = np.bool_(value).reshape(*np.asarray(self.state[key]).shape, -1).any(axis=-1)

            except ValueError:
                value = np.asarray(value)
                if len(value.shape) > 1:
                    inshape = np.asarray(self.state[key]).shape
                    if not inshape:
                        self.state[key][...] = value
                        self.has_state[key] = True
                    else:
                        broadcasted = np.broadcast_to(value, inshape + value.shape[-2:])
                        self.state[key] = [rho for rho in broadcasted]
                        self.has_state[key] = [rho is not None for rho in broadcasted]
                else:
                    self.state[key] = [rho for rho in value]
                    self.has_state[key] = [rho is not None for rho in value]

    @property
    def state(self):
        return self._data['state']

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

        which = self.has_state & ~self._data['hps']
        if which.any():
            projected = project_bath_states(self.state[which])
            self._data['proj'][which] = projected
            self._data['hps'][which] = True

        self._up_to_date = True

    def _get_state(self, item):

        obj = self.__new__(self.__class__)
        obj._data = self._data[item]
        obj.__up_to_date = self._up_to_date

        return obj

    def any(self, *args, **kawrgs):

        return self.has_state.any(*args, **kawrgs)

    def __repr__(self):
        return f"{self.__class__.__name__}(" + self.state.__str__() + ")"

    def __str__(self):
        return self.state.__str__()
