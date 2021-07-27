import functools

import numpy as np
from numpy import ma as ma


def generate_bath_state(bath, nbstates, seed=None, fixstates=None, parallel=False):
    r"""
    Generator of the random *pure* :math:`\hat{I}_z` bath eigenstates.

    Args:
        bath (BathArray): Array of bath spins.
        nbstates (int): Number of random bath states to generate.
        seed (int): Optional. Seed for RNG.
        fixstates (dict): Optional. dict of which bath states to fix. Each key is the index of bath spin,
            value - fixed :math:`\hat{I}_z` projection of the mixed state of nuclear spin.

    Yields:
        ndarray: Array of ``shape = len(bath)`` containing z-projections of the bath spins states.
    """
    rgen = np.random.default_rng(seed)
    rank = 0
    if parallel:
        try:
            import mpi4py
            comm = mpi4py.MPI.COMM_WORLD
            rank = comm.Get_rank()

        except ImportError:
            print('Parallel failed: mpi4py is not found. Running serial')
            parallel = False

    for _ in range(nbstates):
        bath_state = np.empty(bath.shape, dtype=np.float64)
        if rank == 0:
            for n in bath.types:
                s = bath.types[n].s
                snumber = int(round(2 * s + 1))
                mask = bath['N'] == n
                bath_state[mask] = rgen.integers(snumber, size=np.count_nonzero(mask)) - s

            if fixstates is not None:
                for fs in fixstates:
                    bath_state[fs] = fixstates[fs]

        if parallel:
            comm.Bcast(bath_state, root=0)

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

        if self.masked:
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

        for bath_state in generate_bath_state(self.bath, nbstates, seed=seed,
                                              fixstates=self.fixstates, parallel=self.parallel):
            self.bath_state = bath_state
            self.projected_bath_state = bath_state
            result = func(self, *args, **kwargs)

            if self.masked:
                reshaped_result = np.abs(result).reshape(result.shape[0], -1)
                proper = np.all(reshaped_result <= reshaped_result[0] * 1.01, axis=(-1))
                divider += proper.astype(np.int32)
                result[~proper] = 0.

            total += result

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

            if self.masked:
                if rank == 0:
                    divider_shape = divider.shape
                else:
                    divider_shape = None

                divider_shape = comm.bcast(divider_shape, root=0)

                root_divider = np.zeros(divider_shape, dtype=np.int32)
                comm.Allreduce(divider, root_divider, MPI.SUM)

        else:
            root_result = total
            if self.masked:
                root_divider = divider

        root_result = ma.array(root_result, fill_value=0, dtype=np.complex128)

        if self.masked:
            root_result[root_divider == 0] = ma.masked
            root_divider = root_divider.reshape(root_result.shape[0], *[1] * len(root_result.shape[1:]))

        root_result /= root_divider

        return root_result

    return inner_method
