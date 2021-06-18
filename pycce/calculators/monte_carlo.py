from numpy import ma as ma
import numpy as np
import functools

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


def monte_carlo_decorator(func):
    """
    Decorator to sample over random bath states given function.


    **Additional parameters**:

        * **bath** (*BathArray*) -- Array of all bath spins.
        * **nbstates** (*int*) -- Number of random states to sample over.

        * **seed** (*int*) -- Seed for RNG. Default None.

        * **parallel** (*bool*) -- True if parallelize calculation of cluster contributions
          over different mpi threads using mpi4py. Default False.

        * **parallel_states** (*bool*) -- True if to use MPI to parallelize the calculations of
          density matrix equally over present mpi processes. Compared to ``parallel`` keyword,
          when this argument is True each process is given a fraction of random bath states.
          This makes the implementation faster. Works best when the
          number of bath states is divisible by the number of processes, ``nbstates % size == 0``.
          Default False

        * fixstates (*dict*) -- Shows which bath states to fix in random bath states.
          Each key is the index of bath spin,
          value - fixed Sz projection of the mixed state of nuclear spin.

        * masked (*bool*) -- True if mask numerically unstable points (with coherence > 1)
          in the averaging over bath states. Default True. It is up to user to check whether the
          instability is due to numerical error or unphysical system.

    """

    @functools.wraps(func)
    def inner_function(bath, *args,
                       nbstates=100, seed=None, masked=True,
                       parallel_states=False,
                       fixstates=None, parallel=False, **kwargs):

        if parallel_states:
            try:
                from mpi4py import MPI
            except ImportError:
                print('Parallel failed: mpi4py is not found. Running serial')
                parallel_states = False

        if masked:
            divider = 0
        else:
            root_divider = nbstates

        if parallel_states:
            comm = MPI.COMM_WORLD

            size = comm.Get_size()
            rank = comm.Get_rank()

            remainder = nbstates % size
            add = int(rank < remainder)
            nbstates = nbstates // size + add

            if seed:
                seed = seed + rank * 19653252
        else:
            rank = 0

        total = 0j

        for bath_state in generate_bath_state(bath, nbstates, seed=seed, fixstates=fixstates, parallel=parallel):

            result = func(bath, *args, **kwargs,
                          parallel=parallel,
                          bath_state=bath_state)

            if masked:
                reshaped_result = np.abs(result).reshape(result.shape[0], -1)
                proper = np.all(reshaped_result <= reshaped_result[0] + 1e-6, axis=(-1))
                divider += proper.astype(np.int32)
                result[~proper] = 0.

            total += result

        if parallel_states:
            if rank == 0:
                divider_shape = divider.shape
                result_shape = total.shape
            else:
                result_shape = None
                divider_shape = None

            result_shape = comm.bcast(result_shape, root=0)
            divider_shape = comm.bcast(divider_shape, root=0)

            root_result = ma.array(np.zeros(result_shape), dtype=np.complex128)
            comm.Allreduce(total, root_result, MPI.SUM)

            if masked:
                root_divider = np.zeros(divider_shape, dtype=np.int32)
                comm.Allreduce(divider, root_divider, MPI.SUM)

        else:
            root_result = total
            if masked:
                root_divider = divider

        root_result = ma.array(root_result, fill_value=0j, dtype=np.complex128)

        if masked:
            root_result[root_divider == 0] = ma.masked
            root_divider = root_divider.reshape(root_result.shape[0], *[1] * len(root_result.shape[1:]))

        root_result /= root_divider

        return root_result

    return inner_function