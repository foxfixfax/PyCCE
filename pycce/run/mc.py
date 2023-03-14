import functools

import numpy as np
from numpy import ma as ma
from pycce.utilities import gen_state_list
from numba.typed import List


def generate_bath_state(bath, nbstates, seed=None, parallel=False):
    r"""
    Generator of the random *pure* :math:`\hat{I}_z` bath eigenstates.

    Args:
        bath (BathArray): Array of bath spins.
        nbstates (int): Number of random bath states to generate.
        seed (int): Optional. Seed for RNG.
        parallel (bool): True if run in parallel mode. Default False.

    Yields:
        List: list of the pure bath spin state vectors.
    """
    rgen = np.random.default_rng(seed)
    rank = 0
    comm = None

    if parallel:
        try:
            import mpi4py
            comm = mpi4py.MPI.COMM_WORLD
            rank = comm.Get_rank()

        except ImportError:
            print('Parallel failed: mpi4py is not found. Running serial')
            parallel = False

    dimensions = np.empty(bath.shape, dtype=np.int32)

    for _ in range(nbstates):
        bath_state = np.empty(bath.shape, dtype=np.float64)

        if rank == 0:

            for n in np.unique(bath.N):
                s = bath.types[n].s
                snumber = np.int32(round(2 * s + 1))
                mask = bath.N == n
                bath_state[mask] = rgen.integers(snumber, size=np.count_nonzero(mask)) - s

                dimensions[mask] = snumber

        if parallel:
            comm.Bcast(bath_state, root=0)
            comm.Bcast(dimensions, root=0)

        bath_state = gen_state_list(bath_state, dimensions)

        yield bath_state


def monte_carlo_method_decorator(func):
    """
    Decorator to sample over random bath states given function.
    """

    @functools.wraps(func)
    def inner_method(self, *args,
                     **kwargs):
        seed = self.seed
        if self.parallel_states:
            try:
                from mpi4py import MPI
            except ImportError:
                print('Parallel failed: mpi4py is not found. Running serial')
                self.parallel_states = False

        if isinstance(self.masked, bool) and not self.masked:
            self.masked = None

        if self.masked is not None:
            divider = 0
        else:
            root_divider = self.nbstates

        if self.parallel_states:
            comm = MPI.COMM_WORLD

            size = comm.Get_size()
            rank = comm.Get_rank()

            remainder = self.nbstates % size
            add = int(rank < remainder)
            nbstates = self.nbstates // size + add

            if seed:
                seed = seed + rank * 19653252
        else:
            rank = 0
            nbstates = self.nbstates

        total = 0j

        his = self.bath.state.has_state.copy()  # have_initial_state
        for bath_state in generate_bath_state(self.bath[~his], nbstates, seed=seed, parallel=self.parallel):

            self.bath.state[~his] = bath_state
            result = func(self, *args, **kwargs)

            if self.masked is not None:
                if isinstance(self.masked, bool):
                    self.masked = np.abs(result[0] * 1.01)

                proper = np.abs(result) <= self.masked
                divider += proper.astype(np.int32)
                result[~proper] = 0.

            total += result

        self.bath.state[~his] = None

        if self.parallel_states:
            if rank == 0:
                result_shape = total.shape
            else:
                result_shape = None

            result_shape = comm.bcast(result_shape, root=0)
            total = np.asarray(total)
            if not total.shape:
                total = np.zeros(result_shape,

                                 dtype=np.complex128)
            root_result = ma.array(np.zeros(result_shape), dtype=np.complex128)
            comm.Allreduce(total, root_result, MPI.SUM)

            if self.masked is not None:
                if rank == 0:
                    divider_shape = divider.shape
                else:
                    divider_shape = None

                divider_shape = comm.bcast(divider_shape, root=0)
                if np.array(divider).shape != divider_shape:
                    divider = np.zeros(divider_shape, dtype=np.int32)
                root_divider = np.zeros(divider_shape, dtype=np.int32)
                comm.Allreduce(divider, root_divider, MPI.SUM)

        else:
            root_result = total
            if self.masked is not None:
                root_divider = divider

        root_result = ma.array(root_result, fill_value=0, dtype=np.complex128)

        if self.masked is not None:
            root_result[root_divider == 0] = ma.masked
            # root_divider = root_divider.reshape(root_result.shape[0], *[1] * len(root_result.shape[1:]))

        root_result /= root_divider

        return root_result

    return inner_method
