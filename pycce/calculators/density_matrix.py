import numpy as np
import scipy.linalg
from numpy import ma as ma

from pycce.bath.array import BathArray
from pycce.cluster_expansion import cluster_expansion_decorator
from pycce.constants import ELECTRON_GYRO, PI2
from pycce.hamiltonian import total_hamiltonian, expand, eta_hamiltonian, total_hamiltonian


def propagator(timespace, hamiltonian, dimensions=None,
               pulse_sequence=None, alpha=None, beta=None, as_delay=False):
    """
    Function to compute time propagator U.

    Args:
        timespace (ndarray):
            Time points at which to compute propagators.
        hamiltonian (ndarray):
            Matrix representation of the cluster Hamiltonian.
        dimensions (ndarray):
            ndarray of bath spin dimensions. Last entry - electron spin dimensions.
        pulse_sequence (list):
            Pulse_sequence should have format of list with tuples,
            each tuple contains two or three entries:

                1. axis the rotation is about;
                2. angle of rotation. E.g. for Hahn-Echo ``[('x', np.pi/2)]``.
                3. (Optional). Fraction of time before the pulse. E.g. for Hahn-Echo ``[('x', np.pi/2, 0.5)]``.

        alpha (ndarray with shape (2s+1,)):
            Vector representation of the alpha qubit state in :math:`\hat{S}_z` basis.
        beta (ndarray with shape (2s+1,)):
            Vector representation of the beta qubit state in :math:`\hat{S}_z` basis.
        as_delay (bool):
            True if time points are delay between pulses, False if time points are total time.

    Returns:
        ndarray: array of propagators, evaluated at each time point in timespace.
    """
    evalues, evec = np.linalg.eigh(hamiltonian * PI2)

    if not pulse_sequence:

        eigexp = np.exp(-1j * np.tensordot(timespace, evalues, axes=0),
                        dtype=np.complex128)

        u = np.matmul(np.einsum('ij,kj->kij', evec, eigexp, dtype=np.complex128),
                      evec.conj().T)

        return u

    else:

        alpha_x_alpha = np.tensordot(alpha, alpha, axes=0)
        beta_x_beta = np.tensordot(beta, beta, axes=0)
        alpha_x_beta = np.tensordot(alpha, beta, axes=0)
        beta_x_alpha = np.tensordot(beta, alpha, axes=0)

        sigmax = alpha_x_beta + beta_x_alpha
        sigmay = -1j * alpha_x_beta + 1j * beta_x_alpha
        sigmaz = alpha_x_alpha - beta_x_beta

        sigma = {'x': expand(sigmax, len(dimensions) - 1, dimensions),
                 'y': expand(sigmay, len(dimensions) - 1, dimensions),
                 'z': expand(sigmaz, len(dimensions) - 1, dimensions)}

        equispaced = False
        try:
            pulse_sequence[0][2]
        except IndexError:
            equispaced = True

        if equispaced:
            if not as_delay:
                N = len(pulse_sequence)
                timespace = timespace / (2 * N)

            eigexp = np.exp(-1j * np.tensordot(timespace, evalues, axes=0),
                            dtype=np.complex128)

            u = np.matmul(np.einsum('ij,kj->kij', evec, eigexp, dtype=np.complex128),
                          evec.conj().T)
            U = np.eye(u.shape[1])

            for pulse in pulse_sequence:
                ax = pulse[0]
                angle = pulse[1]
                rotation = scipy.linalg.expm(-1j * sigma[ax] * angle / 2)
                U = np.matmul(u, U)
                U = np.matmul(rotation, U)
                U = np.matmul(u, U)

            return U

        U = None
        total_fraction = 0
        for pulse in pulse_sequence:

            ax = pulse[0]
            angle = pulse[1]
            fraction = pulse[2]

            eigexp = np.exp(-1j * np.tensordot(timespace * fraction, evalues, axes=0),
                            dtype=np.complex128)

            u = np.matmul(np.einsum('ij,kj->kij', evec, eigexp, dtype=np.complex128),
                          evec.conj().T)

            rotation = scipy.linalg.expm(-1j * sigma[ax] * angle / 2)

            if U is None:
                U = np.matmul(rotation, u)

            else:
                U = np.matmul(u, U)
                U = np.matmul(rotation, U)

            total_fraction += fraction

        if total_fraction < 1:
            eigexp = np.exp(-1j * np.tensordot(timespace * (1 - total_fraction), evalues, axes=0),
                            dtype=np.complex128)

            u = np.matmul(np.einsum('ij,kj->kij', evec, eigexp, dtype=np.complex128),
                          evec.conj().T)

            U = np.matmul(u, U)

    return U


def compute_dm(dm0, H, alpha, beta, timespace, pulse_sequence=None, as_delay=False, states=None):
    """
    Function to compute density matrix of the central spin, given Hamiltonian H.

    Args:
        dm0 (ndarray): Initial density matrix of central spin.
        H (ndarray): Cluster Hamiltonian.
        alpha (ndarray with shape (2s+1,)):
            Vector representation of the alpha qubit state in :math:`\hat{S}_z` basis.
        beta (ndarray with shape (2s+1,)):
            Vector representation of the beta qubit state in :math:`\hat{S}_z` basis.
        timespace (ndarray): Time points at which to compute density matrix.
        pulse_sequence (list):
            Pulse_sequence should have format of list with tuples,
            each tuple contains two or three entries:

                1. axis the rotation is about;
                2. angle of rotation. E.g. for Hahn-Echo ``[('x', np.pi/2)]``.
                3. (Optional). Fraction of time before the pulse. E.g. for Hahn-Echo ``[('x', np.pi/2, 0.5)]``.

        as_delay (bool):
            True if time points are delay between pulses, False if time points are total time.
        states (ndarray): ndarray of bath states in any accepted format.

    Returns:
        ndarray: Array of density matrices evaluated at all time points in timespace.
    """

    dm0 = generate_dm0(dm0, H.dimensions, states)
    dm = full_dm(dm0, H, alpha, beta, timespace, pulse_sequence=pulse_sequence, as_delay=as_delay)

    initial_shape = dm.shape
    dm.shape = (initial_shape[0], *H.dimensions, *H.dimensions)
    for d in range(len(H.dimensions) + 1, 2, -1):  # The last one is el spin
        dm = np.trace(dm, axis1=1, axis2=d)
    return dm


def full_dm(dm0, H, alpha, beta, timespace, pulse_sequence=None, as_delay=False):
    """
    A function to compute density matrix of the cluster, using hamiltonian H
    from the initial density matrix of the cluster.

    Args:
        dm0 (ndarray):
            Initial density matrix of the cluster
        H (ndarray):
            Cluster Hamiltonian
        alpha (ndarray with shape (2s+1,)):
            Vector representation of the alpha qubit state in :math:`\hat{S}_z` basis.
        beta (ndarray with shape (2s+1,)):
            Vector representation of the beta qubit state in :math:`\hat{S}_z` basis.
        timespace (ndarray): Time points at which to compute coherence function.
        pulse_sequence (list):
            Pulse_sequence should have format of list with tuples,
            each tuple contains two or three entries:

                1. axis the rotation is about;
                2. angle of rotation. E.g. for Hahn-Echo ``[('x', np.pi/2)]``.
                3. (Optional). Fraction of time before the pulse. E.g. for Hahn-Echo ``[('x', np.pi/2, 0.5)]``.

        as_delay (bool):
            True if time points are delay between pulses, False if time points are total time.

    Returns:
        ndarray: Array of density matrices of the cluster, evaluated at the time points from timespace.
    """

    U = propagator(timespace, H.data, H.dimensions, pulse_sequence, alpha, beta, as_delay=as_delay)

    dmUdagger = np.matmul(dm0, np.transpose(U.conj(), axes=(0, 2, 1)))
    dm = np.matmul(U, dmUdagger)
    # einsum does the same as the following
    # dm = np.einsum('zli,ij,zkj->zlk', U, dm0, U.conj())
    return dm


def generate_dm0(dm0, dimensions, states=None):
    """
    A function to generate initial density matrix of the cluster.
    Args:
        dm0 (ndarray):
            Initial density matrix of the central spin.
        dimensions (ndarray):
            ndarray of bath spin dimensions. Last entry - electron spin dimensions.
        states (ndarray):
            ndarray of bath states in any accepted format.

    Returns:
        ndarray: Initial density matrix of the cluster.
    """

    if states is None:
        dmtotal0 = expand(dm0, len(dimensions) - 1, dimensions) / np.prod(dimensions[:-1])
    else:
        dmtotal0 = gen_density_matrix(states, dimensions[:-1])
        dmtotal0 = np.kron(dmtotal0, dm0)

    return dmtotal0


def gen_density_matrix(states=None, dimensions=None):
    """
    Generate density matrix from the ndarray of states.

    Args:
        states (ndarray):
            Array of bath spin states. If None, assume completely random state.
            Can have the following forms:

                - array of the :math:`\hat{I}_z` projections for each spin.
                  Assumes that each bath spin is in the pure eigenstate of :math:`\hat{I}_z`.

                - array of the diagonal elements of the density matrix for each spin.
                  Assumes mixed state and the density matrix for each bath spin
                  is diagonal in :math:`\hat{I}_z` basis.

                - array of the density matrices of the bath spins.

        dimensions (ndarray):
            array of bath spin dimensions. Last entry - electron spin dimensions.

    Returns:
        ndarray: Density matrix of the system.
    """
    if states is None:
        tdim = np.prod(dimensions)
        dmtotal0 = np.eye(tdim) / tdim

        return dmtotal0

    dmtotal0 = np.eye(1, dtype=np.complex128)

    for i, s in enumerate(states):

        if not hasattr(s, "__len__"):
            # assume s is int or float showing the spin projection in the pure state
            d = dimensions[i]
            dm_nucleus = np.zeros((d, d), dtype=np.complex128)
            state_number = int(round((d - 1) / 2 - s))
            dm_nucleus[state_number, state_number] = 1

        else:
            if s.shape.__len__() == 1:
                d = dimensions[i]
                dm_nucleus = np.zeros((d, d), dtype=np.complex128)
                np.fill_diagonal(dm_nucleus, s)

            else:
                dm_nucleus = s

        dmtotal0 = np.kron(dmtotal0, dm_nucleus)
    return dmtotal0


@cluster_expansion_decorator
def decorated_density_matrix(cluster, allspin, dm0, alpha, beta, magnetic_field, zfs, timespace, pulse_sequence,
                             gyro_e=ELECTRON_GYRO, as_delay=False, zeroth_cluster=None,
                             bath_state=None, projected_bath_state=None):
    """
    Decorated function to compute electron density matrix with gCCE with mean field.

    Args:
        cluster (dict): Clusters included in different CCE orders of structure ``{int order: ndarray([[i,j],[i,j]])}``.

        allspin (BathArray): Array of all bath spins.

        dm0 (ndarray): Initial density matrix of the central spin.

        alpha (ndarray with shape (2s+1,)): Vector representation of the alpha qubit state in :math:`\hat{S}_z` basis.

        beta (ndarray with shape (2s+1,)): Vector representation of the beta qubit state in :math:`\hat{S}_z` basis.

        magnetic_field (ndarray): Magnetic field of type ``mfield = np.array([Bx, By, Bz])``.

        zfs (ndarray with shape (3,3)): Zero Field Splitting tensor of the central spin.

        timespace (ndarray): Time points at which to compute coherence function.

        pulse_sequence (list):
            Pulse_sequence should have format of list with tuples,
            each tuple contains two or three entries:

                1. axis the rotation is about;
                2. angle of rotation. E.g. for Hahn-Echo ``[('x', np.pi/2)]``.
                3. (Optional). Fraction of time before the pulse. E.g. for Hahn-Echo ``[('x', np.pi/2, 0.5)]``.

        bath_state (ndarray): Array of bath states in any accepted format.

        gyro_e (float or ndarray with shape (3, 3)):
            Gyromagnetic ratio of the central spin

            **OR**

            tensor corresponding to interaction between magnetic field and
            central spin.

        as_delay (bool): True if time points are delay between pulses, False if time points are total time.

        zeroth_cluster (ndarray): Density matrix of isolated central spin at all time points.

        projected_bath_state (ndarray): Array of ``shape = len(allspin)``
            containing z-projections of the bath spins states.

    Returns:
        ndarray: array of central spin density matrices at each time point.
    """
    nspin = allspin[cluster]
    central_spin = (alpha.size - 1) / 2

    states, others, other_states = _check_projected_states(cluster, allspin, bath_state, projected_bath_state)

    # en, eiv = np.linalg.eigh(selfh)

    if zeroth_cluster is None:
        selfh = total_hamiltonian(BathArray(0), magnetic_field, zfs, others=allspin, other_states=projected_bath_state,
                                  central_gyro=gyro_e, central_spin=central_spin)
        zeroth_cluster = compute_dm(dm0, selfh, alpha, beta,
                                    timespace, pulse_sequence, as_delay=as_delay)
        #

        zeroth_cluster = ma.masked_array(zeroth_cluster, mask=(np.isclose(zeroth_cluster, 0)))

    totalh = total_hamiltonian(nspin, magnetic_field, zfs, others=others, other_states=other_states,
                               central_gyro=gyro_e, central_spin=central_spin)

    dms = compute_dm(dm0, totalh, alpha, beta, timespace, pulse_sequence,
                     as_delay=as_delay, states=states) / zeroth_cluster
    return dms


def compute_cce_dm(clusters, bath, dm0, alpha, beta, magnetic_field, zfs, timespace, pulse_sequence,
                   bath_state=None, gyro_e=ELECTRON_GYRO, as_delay=False,
                   projected_bath_state=None, parallel=False, direct=False):
    central_spin = (alpha.size - 1) / 2

    selfh = total_hamiltonian(BathArray(0), magnetic_field, zfs, others=bath, other_states=projected_bath_state,
                              central_gyro=gyro_e, central_spin=central_spin)

    #
    # en, eiv = np.linalg.eigh(selfh)
    # Rotate dms into eigenspace of self hamiltonian eiv.conj().T @ ... @ eiv
    dms = compute_dm(dm0, selfh, alpha, beta, timespace, pulse_sequence, as_delay=as_delay)
    dms = ma.masked_array(dms, mask=(np.isclose(dms, 0)), fill_value=0j, dtype=np.complex128)

    dms *= decorated_density_matrix(clusters, bath, dm0, alpha, beta,
                                    magnetic_field, zfs, timespace,
                                    pulse_sequence, bath_state=bath_state,
                                    projected_bath_state=projected_bath_state,
                                    gyro_e=gyro_e, as_delay=as_delay, zeroth_cluster=dms,
                                    parallel=parallel, direct=direct)
    #  eiv @ ... @ eiv.conj().T
    return dms.filled(0j)


def _check_projected_states(cluster, allspin, states=None, projected_states=None):
    others = None
    other_states = None

    if states is not None:
        states = states[cluster]

    if projected_states is not None:
        others_mask = np.ones(allspin.shape, dtype=bool)
        others_mask[cluster] = False
        others = allspin[others_mask]
        other_states = projected_states[others_mask]

    return states, others, other_states


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
        random_state (ndarray): Array of ``shape = len(bath)`` containing z-projections of the bath spins states.
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


def monte_carlo_dm(clusters, bath, dm0, alpha, beta, magnetic_field, zfs, timespace, pulse_sequence,
                   central_gyro=ELECTRON_GYRO, as_delay=False,
                   nbstates=100, seed=None, masked=True,
                   normalized=None, parallel_states=False,
                   fixstates=None, direct=False, parallel=False):
    """
    Compute density matrix of the central spin using generalized CCE with Monte-Carlo bath state sampling.

    Args:

        cluster (dict):
            Clusters included in different CCE orders of structure ``{int order: ndarray([[i,j],[i,j]])}``.

        bath (BathArray):
            Array of all bath spins.

        dm0 (ndarray):
            Initial density matrix of the central spin.

        alpha (ndarray with shape (2s+1,)):
            Vector representation of the alpha qubit state in :math:`\hat{S}_z` basis.

        beta (ndarray with shape (2s+1,)):
            Vector representation of the beta qubit state in :math:`\hat{S}_z` basis.

        magnetic_field (ndarray):
            Magnetic field of type ``mfield = np.array([Bx, By, Bz])``.

        zfs (ndarray with shape (3, 3)):
            Zero Field Splitting tensor of the central spin.

        timespace (ndarray): Time points at which to compute coherence function.

        pulse_sequence (list):
            Pulse_sequence should have format of list with tuples,
            each tuple contains two or three entries:

                1. axis the rotation is about;
                2. angle of rotation. E.g. for Hahn-Echo ``[('x', np.pi/2)]``.
                3. (Optional). Fraction of time before the pulse. E.g. for Hahn-Echo ``[('x', np.pi/2, 0.5)]``.

        central_gyro (float or ndarray with shape (3,3)):
            gyromagnetic ratio of the central spin

            **OR**

            tensor corresponding to interaction between magnetic field and
            central spin.

        as_delay (bool): True if time points are delay between pulses, False if time points are total time.

        nbstates (int): Number of random bath states to sample.

        seed (int): Seed for the RNG.

        masked (bool): True if mask numerically unstable points (with elements > 1) in the averaging over bath states
            False if not. Default True.

        normalized (ndarray of bool): which diagonal elements to renormalize,
            so the total sum of the diagonal elements is 1.

        parallel_states (bool): True if use MPI to parallelize the calculations of density matrix.
            for each random bath state.

        fixstates (dict): Dictionary with bath states to fix. Each key is the index of bath spin,
            value - fixed Sz projection of the mixed state of nuclear spin.

        direct (bool): True if use the direct approach in cluster expansion.

        parallel (bool): True if use MPI for parallel computing of the cluster contributions.

        **kwargs: Additional keyword arguments for the projected_hamiltonian.

    Returns:
        ndarray: array of central spin density matrices at each time point.

    """
    if parallel_states:
        try:
            from mpi4py import MPI
        except ImportError:
            print('Parallel failed: mpi4py is not found. Running serial')
            parallel_states = False

    if masked:
        divider = np.zeros(timespace.shape, dtype=np.int32)
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
            seed = seed + rank * 196532520
    else:
        rank = 0

    averaged_dms = ma.zeros((timespace.size, *dm0.shape), dtype=np.complex128)

    for bath_state in generate_bath_state(bath, nbstates, seed=seed, fixstates=fixstates, parallel=parallel):

        dms = compute_cce_dm(clusters, bath, dm0, alpha, beta, magnetic_field, zfs, timespace, pulse_sequence,
                             bath_state=bath_state, gyro_e=central_gyro, as_delay=as_delay,
                             projected_bath_state=bath_state, parallel=parallel, direct=direct)
        if masked:
            proper = np.all(np.abs(dms) <= 1, axis=(1, 2))
            divider += proper.astype(np.int32)
            dms[~proper] = 0.

        if normalized is not None:
            norm = np.asarray(normalized)
            ind = np.arange(dms.shape[1])
            diagonals = dms[:, ind, ind]

            sums = np.sum(diagonals[:, norm], keepdims=True, axis=-1)
            sums[sums == 0.] = 1

            expsum = 1 - np.sum(diagonals[:, ~norm], keepdims=True, axis=-1)

            diagonals[:, norm] = diagonals[:, norm] / sums * expsum
            dms[:, ind, ind] = diagonals

        averaged_dms += dms

    if parallel_states:
        root_dms = ma.array(np.zeros(averaged_dms.shape), dtype=np.complex128)
        comm.Allreduce(averaged_dms, root_dms, MPI.SUM)

        if masked:
            root_divider = np.zeros(divider.shape, dtype=np.int32)
            comm.Allreduce(divider, root_divider, MPI.SUM)

    else:
        root_dms = averaged_dms
        if masked:
            root_divider = divider

    root_dms = ma.array(root_dms, fill_value=0j, dtype=np.complex128)

    if masked:
        root_dms[root_divider == 0] = ma.masked
        root_divider = root_divider[:, np.newaxis, np.newaxis]

    root_dms /= root_divider

    return root_dms


#### Legacy code tbd
@cluster_expansion_decorator
def decorated_density_matrix_old(cluster, allspin, dm0, alpha, beta, magnetic_field, zfs, timespace, pulse_sequence,
                                 gyro_e=ELECTRON_GYRO, as_delay=False, zeroth_cluster=None,
                                 bath_state=None, eta=None):
    """
    Decorated function to compute electron density matrix with gCCE (without mean field).

    Args:
        cluster (dict): Clusters included in different CCE orders of structure ``{int order: ndarray([[i,j],[i,j]])}``.

        allspin (BathArray): Array of all bath spins.

        dm0 (ndarray): Initial density matrix of the central spin.

        alpha (ndarray with shape (2s+1,)): Vector representation of the alpha qubit state in :math:`\hat{S}_z` basis.

        beta (ndarray with shape (2s+1,)): Vector representation of the beta qubit state in :math:`\hat{S}_z` basis.

        magnetic_field (ndarray): Magnetic field of type ``mfield = np.array([Bx, By, Bz])``.

        zfs (ndarray with shape (3, 3)): Zero Field Splitting tensor of the central spin.

        timespace (ndarray): Time points at which to compute coherence function.

        pulse_sequence (list):
            Pulse_sequence should have format of list with tuples,
            each tuple contains two or three entries:

                1. axis the rotation is about;
                2. angle of rotation. E.g. for Hahn-Echo ``[('x', np.pi/2)]``.
                3. (Optional). Fraction of time before the pulse. E.g. for Hahn-Echo ``[('x', np.pi/2, 0.5)]``.

        gyro_e (float or ndarray with shape (3, 3)):
            Gyromagnetic ratio of the central spin

            **OR**

            tensor corresponding to interaction between magnetic field and central spin.

        as_delay (bool):
            True if time points are delay between pulses, False if time points are total time.

        zeroth_cluster (ndarray):
            Density matrix of isolated central spin at all time poins.

        bath_state (ndarray): Array of bath states in any accepted format.

        eta (float): Value of eta (see eta_hamiltonian)

    Returns:
        ndarray: Array of central spin density matricies at each time point.
    """

    nspin = allspin[cluster]

    # if imap is not None:
    #     imap = imap.subspace(cluster)

    central_spin = (alpha.size - 1) / 2
    if bath_state is not None:
        states = bath_state[cluster]
    else:
        states = None

    if zeroth_cluster is None:
        H = total_hamiltonian(BathArray(0), magnetic_field, zfs, central_spin=central_spin,
                              central_gyro=gyro_e)
        zeroth_cluster = compute_dm(dm0, H, alpha, beta, timespace, pulse_sequence, as_delay=as_delay)
        zeroth_cluster = ma.masked_array(zeroth_cluster, mask=(zeroth_cluster == 0))

    H = total_hamiltonian(nspin, magnetic_field, zfs, central_spin=central_spin,
                          central_gyro=gyro_e)
    if eta is not None:
        H += eta_hamiltonian(nspin, alpha, beta, eta)

    dms = compute_dm(dm0, H, alpha, beta, timespace, pulse_sequence, as_delay=as_delay, states=states) / zeroth_cluster

    return dms
