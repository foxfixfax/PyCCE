import numpy as np
import scipy.linalg
from numpy import ma as ma
from pycce.bath.array import BathArray
from pycce.cluster_expansion import cluster_expansion_decorator
from pycce.constants import ELECTRON_GYRO, PI2
from pycce.hamiltonian import total_hamiltonian, expand
from numba import jit
from pycce.monte_carlo import monte_carlo_decorator

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
                3. (Optional). Time before the pulse. Can be as fixed, as well as varied.
                   If varied, it should be provided as an array with the same
                   length as ``timespace``.

                   E.g. for Hahn-Echo ``[('x', np.pi/2, timespace/2)]``.

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

                if angle == np.pi:
                    rotation = -1j * sigma[ax]
                elif angle == 0:
                    rotation = np.eye(sigma[ax].shape)
                else:
                    rotation = scipy.linalg.expm(-1j * sigma[ax] * angle / 2)

                U = np.matmul(u, U)
                U = np.matmul(rotation, U)
                U = np.matmul(u, U)

            return U

        U = None
        times = 0
        for pulse in pulse_sequence:

            ax = pulse[0]
            angle = pulse[1]
            timesteps = pulse[2]

            eigexp = np.exp(-1j * np.tensordot(timesteps, evalues, axes=0),
                            dtype=np.complex128)

            u = np.matmul(np.einsum('...ij,...j->...ij', evec, eigexp, dtype=np.complex128),
                          evec.conj().T)
            times += timesteps

            if angle == np.pi:
                rotation = -1j * sigma[ax]
            else:
                rotation = scipy.linalg.expm(-1j * sigma[ax] * angle / 2)

            if U is None:
                U = np.matmul(rotation, u)

            else:
                U = np.matmul(u, U)
                U = np.matmul(rotation, U)

        if ((timespace - times) >= 0).all() and (timespace - times).any():
            eigexp = np.exp(-1j * np.tensordot(timespace - times, evalues, axes=0),
                            dtype=np.complex128)

            u = np.matmul(np.einsum('ij,kj->kij', evec, eigexp, dtype=np.complex128),
                          evec.conj().T)

            U = np.matmul(u, U)
        elif ((timespace - times) < 0).any():
            raise ValueError(f"Pulse sequence time steps add up to larger than total times"
                             f"{np.argwhere((timespace - times) < 0)} are longer than total time.")
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
    if len(dm0.shape) > 1:
        dmUdagger = np.matmul(dm0, np.transpose(U.conj(), axes=(0, 2, 1)))
        dm = np.matmul(U, dmUdagger)
    else:
        dm = U @ dm0
        dm = np.einsum('ki,kj->kij', dm, dm.conj())

    return dm


def generate_dm0(dm0, dimensions, states=None):
    """
    A function to generate initial density matrix or statevector of the cluster.
    Args:
        dm0 (ndarray):
            Initial density matrix of the central spin.
        dimensions (ndarray):
            ndarray of bath spin dimensions. Last entry - electron spin dimensions.
        states (ndarray):
            ndarray of bath states in any accepted format.

    Returns:
        ndarray:
            Initial density matrix of the cluster
            **OR** statevector if dm0 is vector and ``states`` are provided as list of pure states.
    """

    if states is None:
        dmtotal0 = expand(dm0, len(dimensions) - 1, dimensions) / np.prod(dimensions[:-1])
    elif len(dm0.shape) == 1:
        dmtotal0 = generate_pure_initial_state(dm0, dimensions, states)

    else:
        dmtotal0 = gen_density_matrix(states, dimensions[:-1])
        dmtotal0 = np.kron(dmtotal0, dm0)

    return dmtotal0


def generate_pure_initial_state(state0, dimensions, states):
    """
    A function to generate initial state vector of the cluster with central spin.

    Args:
        state0 (ndarray):
            Initial state of the central spin.
        dimensions (ndarray):
            ndarray of bath spin dimensions. Last entry - electron spin dimensions.
        states (ndarray):
            ndarray of bath states in any accepted format.

    Returns:
        ndarray: Initial state vector of the cluster.
    """

    cluster_state = 1

    for i, s in enumerate(states):
        d = dimensions[i]
        n = int(round((d - 1) / 2 - s))

        state = np.zeros(d)
        state[n] = 1
        cluster_state = np.kron(cluster_state, state)

    with_central_spin = np.kron(cluster_state, state0)

    return with_central_spin


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
def decorated_density_matrix(allspin, cluster, dm0, alpha, beta, magnetic_field, zfs, timespace, pulse_sequence,
                             gyro_e=ELECTRON_GYRO, as_delay=False, zeroth_cluster=None,
                             bath_state=None, projected_bath_state=None):
    r"""
    Inner decorated function to compute central spin coherence with gCCE with mean field.

    Args:
        allspin (BathArray): Array of all bath spins.

        cluster (dict): Clusters included in different CCE orders of structure ``{int order: ndarray([[i,j],[i,j]])}``.

        dm0 (ndarray): Initial density matrix of the central spin.

        alpha (ndarray with shape (2s+1,) or int):
            Vector representation of the alpha qubit state in :math:`\hat{S}_z` basis or
            index of the energy eigenstate to be considered as one.


        beta (ndarray with shape (2s+1,) or int):
            Vector representation of the beta qubit state in :math:`\hat{S}_z` basis or
            index of the energy eigenstate to be considered as one.

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
        ndarray: Array of central spin coherence at each time point without zero cluster contribution.
    """
    nspin = allspin[cluster]
    central_spin = (alpha.size - 1) / 2

    states, others, other_states = _check_projected_states(cluster, allspin, bath_state, projected_bath_state)

    # en, eiv = np.linalg.eigh(selfh)

    if zeroth_cluster is None:
        selfh = total_hamiltonian(BathArray((0,)), magnetic_field, zfs, others=allspin,
                                  other_states=projected_bath_state,
                                  central_gyro=gyro_e, central_spin=central_spin)

        res = full_dm(dm0, selfh, alpha, beta, timespace, pulse_sequence=pulse_sequence, as_delay=as_delay)

        if len(dm0.shape) > 1:
            zeroth_cluster = alpha.conj() @ res @ beta
            zeroth_cluster = (alpha.conj() @ zeroth_cluster @ beta)

        else:
            zeroth_cluster = np.inner(alpha.conj(), res) * np.inner(res.conj(), beta)

        zeroth_cluster = ma.masked_array(zeroth_cluster, mask=(np.isclose(zeroth_cluster, 0)))

    totalh = total_hamiltonian(nspin, magnetic_field, zfs, others=others, other_states=other_states,
                               central_gyro=gyro_e, central_spin=central_spin)

    result = compute_dm(dm0, totalh, alpha, beta, timespace, pulse_sequence,
                        as_delay=as_delay, states=states)

    result = alpha.conj() @ result @ beta / zeroth_cluster

    return result


def compute_cce_dm(bath, clusters, timespace, alpha, beta, magnetic_field, zfs, pulses, dm0, bath_state=None,
                   gyro_e=ELECTRON_GYRO, as_delay=False, projected_bath_state=None, parallel=False, direct=False,
                   central_spin=None):
    """
    Function to compute coherence of the central spin using gCCE.

    Args:

        bath (BathArray): Array of all bath spins.

        clusters (dict): Clusters included in different CCE orders of structure ``{int order: ndarray([[i,j],[i,j]])}``.

        alpha (ndarray with shape (2s+1,)): Vector representation of the alpha qubit state in :math:`\hat{S}_z` basis
            or index of the eigenstate to be considered one (0 being lowest energy).

        beta (ndarray with shape (2s+1,)): Vector representation of the beta qubit state in :math:`\hat{S}_z` basis.
            or index of the eigenstate to be considered one (0 being lowest energy).

        magnetic_field (ndarray): Magnetic field of type ``mfield = np.array([Bx, By, Bz])``.

        zfs (ndarray with shape (3,3)): Zero Field Splitting tensor of the central spin.

        timespace (ndarray): Time points at which to compute coherence function.

        pulses (list):
            Pulse sequence should have format of list with tuples,
            each tuple contains two or three entries:

                1. axis the rotation is about;
                2. angle of rotation. E.g. for Hahn-Echo ``[('x', np.pi/2)]``.
                3. (Optional). Fraction of time before the pulse. E.g. for Hahn-Echo ``[('x', np.pi/2, 0.5)]``.

        dm0 (ndarray): Initial density matrix of the central spin.

        bath_state (ndarray): Array of bath states in any accepted format.

        gyro_e (float or ndarray with shape (3, 3)):
            Gyromagnetic ratio of the central spin

            **OR**

            tensor corresponding to interaction between magnetic field and
            central spin.

        as_delay (bool): True if time points are delay between pulses, False if time points are total time.


        projected_bath_state (ndarray): Array of ``shape = len(allspin)``
            containing z-projections of the bath spins states.

       central_spin (float): Value of the central spin.

       direct (bool):
            True if use direct approach (requires way more memory but might be more numerically stable).
            False if use memory efficient approach. Default False.

       parallel (bool):
            True if parallelize calculation of cluster contributions over different mpi processes.
            Default False.

    Returns:
        ndarray: Array of central spin coherence at each time point.

    """

    alpha = np.asarray(alpha)
    beta = np.asarray(beta)
    if central_spin is None:
        try:
            central_spin = (dm0.shape[0] - 1) / 2
        except AttributeError:
            raise ValueError('Central spin is unknown')
    selfh = total_hamiltonian(BathArray((0,)), magnetic_field, zfs=zfs, others=bath, other_states=projected_bath_state,
                              central_gyro=gyro_e, central_spin=central_spin)

    # onlys = total_hamiltonian(BathArray(0), magnetic_field, zfs, central_gyro=gyro_e, central_spin=central_spin)
    if (not alpha.shape) or (not beta.shape):
        en, eiv = np.linalg.eigh(selfh)

        alpha = eiv[:, alpha]
        beta = eiv[:, beta]

        state = (alpha + beta) / np.linalg.norm(alpha + beta)

        check = False
        if projected_bath_state is not None:
            try:
                check = all(projected_bath_state == bath_state)
            except TypeError:
                check = False

        if check:
            dm0 = state
        else:
            dm0 = np.tensordot(state, state, axes=0)

    if dm0 is None:
        raise ValueError('Initial density matrix of the central spin is not provided')
    res = full_dm(dm0, selfh, alpha, beta, timespace, pulse_sequence=pulses, as_delay=as_delay)
    dms = alpha.conj() @ res @ beta

    if len(dm0.shape) > 1:
        normalization = (alpha.conj() @ dm0 @ beta)
    else:
        normalization = np.inner(alpha.conj(), dm0) * np.inner(dm0.conj(), beta)

    dms = ma.masked_array(dms, mask=(np.isclose(dms, 0)), fill_value=0j, dtype=np.complex128)
    dms *= decorated_density_matrix(bath, clusters, dm0, alpha, beta,
                                    magnetic_field, zfs, timespace,
                                    pulses, bath_state=bath_state,
                                    projected_bath_state=projected_bath_state,
                                    gyro_e=gyro_e, as_delay=as_delay, zeroth_cluster=dms,
                                    parallel=parallel, direct=direct, )

    #  eiv @ ... @ eiv.conj().T
    return dms.filled(0) / normalization


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


@monte_carlo_decorator
def monte_carlo_dm(bath, clusters, timespace, pulses, dm0, alpha, beta,
                   magnetic_field, zfs,
                   central_gyro=ELECTRON_GYRO, as_delay=False, bath_state=None,
                   direct=False, parallel=False, central_spin=None):
    """
    Compute coherence of the central spin using generalized CCE with Monte-Carlo bath state sampling.
    Note that because the function is decorated, the actual call differs from the one above by virtue of adding
    several additional keywords (see ``monte_carlo_decorator`` for details).

    Args:

        bath (BathArray):
            Array of all bath spins.

        clusters (dict):
            Clusters included in different CCE orders of structure ``{int order: ndarray([[i,j],[i,j]])}``.

        timespace (ndarray): Time points at which to compute coherence function.

        pulses (list):
            Pulse sequence should have format of list with tuples,
            each tuple contains two or three entries:

                1. axis the rotation is about;
                2. angle of rotation. E.g. for Hahn-Echo ``[('x', np.pi/2)]``.
                3. (Optional). Fraction of time before the pulse. E.g. for Hahn-Echo ``[('x', np.pi/2, 0.5)]``.

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

        parallel_states (bool): True if use MPI to parallelize the calculations of density matrix.
            for each random bath state.

        fixstates (dict): Dictionary with bath states to fix. Each key is the index of bath spin,
            value - fixed Sz projection of the mixed state of nuclear spin.

        direct (bool): True if use the direct approach in cluster expansion.

        parallel (bool): True if use MPI for parallel computing of the cluster contributions.

        central_spin (float): Value of the central spin.

    Returns:
        ndarray: Array of central spin coherence at each time point.

    """
    result = compute_cce_dm(bath, clusters, timespace, alpha, beta, magnetic_field, zfs, pulses, dm0,
                            bath_state=bath_state, gyro_e=central_gyro, as_delay=as_delay,
                            projected_bath_state=bath_state, parallel=parallel, direct=direct,
                            central_spin=central_spin)
    return result






