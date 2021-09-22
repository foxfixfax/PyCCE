from .base import Hamiltonian
from .functions import *


def projected_hamiltonian(bath, center, mfield,
                          others=None, other_states=None):
    r"""
    Compute projected hamiltonian on state and beta qubit states. Wrapped function so the actual call does not
    follow the one above!

    Args:
        bath (BathArray):
            array of all bath spins in the cluster.
        projections_alpha (ndarray with shape (3,)): Projections of the central spin level alpha
            :math:`[\braket{\hat{S}_x}, \braket{\hat{S}_y}, \braket{\hat{S}_z}]`.

        projections_beta (ndarray with shape (3,)): Projections of the central spin level beta.
            :math:`[\braket{\hat{S}_x}, \braket{\hat{S}_y}, \braket{\hat{S}_z}]`

        mfield (ndarray with shape (3,)):
            Magnetic field of type ``mfield = np.array([Bx, By, Bz])``.

        others (BathArray with shape (m,)):
            array of all bath spins outside the cluster

        other_states (ndarray with shape (m,) or (m, 3)):
            Array of Iz projections for each bath spin outside of the given cluster.

        energy_alpha (float): Energy of the alpha state

        energy_beta (float): Energy of the beta state

        energies (ndarray with shape (2s-1,)): Array of energies of all states of the central spin.

        projections_alpha_all (ndarray with shape (2s-1, 3)):
            Array of vectors of the central spin matrix elements of form:

            .. math::

                [\bra{\alpha}\hat{S}_x\ket{j}, \bra{\alpha}\hat{S}_y\ket{j}, \bra{\alpha}\hat{S}_z\ket{j}],

            where :math:`\ket{\alpha}` is the alpha qubit state, and :math:`\ket{\j}` are all states.

        projections_beta_all (ndarray with shape (2s-1, 3)):
            Array of vectors of the central spin matrix elements of form:

            .. math::

                [\bra{\beta}\hat{S}_x\ket{j}, \bra{\beta}\hat{S}_y\ket{j}, \bra{\beta}\hat{S}_z\ket{j}],

            where :math:`\ket{\beta}` is the beta qubit state, and :math:`\ket{\j}` are all states.

    Returns:
        tuple: *tuple* containing:

            * **Hamiltonian**: Hamiltonian of the given cluster, conditioned on the alpha qubit state.
            * **Hamiltonian**: Hamiltonian of the given cluster, conditioned on the beta qubit state.
    """
    dims, vectors = dimensions_spinvectors(bath, central_spin=None)
    clusterint = bath_interactions(bath, vectors)

    halpha = 0 + clusterint
    hbeta = 0 + clusterint

    for ivec, n in zip(vectors, bath):
        hsingle = expanded_single(ivec, n.gyro, mfield, n['Q'], n.detuning)

        if others is not None and other_states is not None:
            hsingle += overhauser_bath(ivec, n['xyz'], n.gyro, others.gyro,
                                       others['xyz'], other_states)
        hf_alpha = 0
        hf_beta = 0
        for i, c in enumerate(center):
            if center.size > 1:
                hf = n['A'][i]
            else:
                hf = n['A']
            hf_alpha += conditional_hyperfine(hf, ivec, c.projections_alpha)
            hf_beta += conditional_hyperfine(hf, ivec, c.projections_beta)

        halpha += hsingle + hf_alpha
        hbeta += hsingle + hf_beta

    if center.energy_alpha is not None:
        for i, c in enumerate(center):
            if center.size > 1:
                hf = bath.A[:, i]
            else:
                hf = bath.A

            halpha += bath_mediated(hf, vectors, center.energy_alpha,
                                    center.energies, c.projections_alpha_all)

            hbeta += bath_mediated(hf, vectors, center.energy_beta,
                                   center.energies, c.projections_beta_all)

    halpha = Hamiltonian(dims, vectors, data=halpha)
    hbeta = Hamiltonian(dims, vectors, data=hbeta)

    return halpha, hbeta


def total_hamiltonian(bath, center, mfield, others=None, other_states=None):
    """
    Compute total Hamiltonian for the given cluster including mean field effect of all bath spins.
    Wrapped function so the actual call does not follow the one above!

    Args:
        bath (BathArray): Array of bath spins.
        center(CenterArray): Array of central spins.
        mfield (ndarray with shape (3,)):
            Magnetic field of type ``mfield = np.array([Bx, By, Bz])``.
        others (BathArray with shape (m,)):
            array of all bath spins outside the cluster
        other_states (ndarray with shape (m,) or (m, 3)):
            Array of Iz projections for each bath spin outside of the given cluster.
        zfs (ndarray with shape (3,3)):
            Zero Field Splitting tensor of the central spin.

    Returns:
        Hamiltonian: hamiltonian of the given cluster, including central spin.

    """
    dims, vectors = dimensions_spinvectors(bath, central_spin=center)

    totalh = bath_interactions(bath, vectors)

    for i, c in enumerate(center):
        totalh += self_central(vectors[bath.size + i], mfield, c.zfs, c.gyro)

        if others is not None and other_states is not None:
            if center.size == 1:
                hf = others['A']
            else:
                hf = others['A'][:, i]
            totalh += overhauser_central(vectors[bath.size + i], hf, other_states)

    totalh += center_interactions(center, vectors[bath.size:])

    for j, n in enumerate(bath):
        ivec = vectors[j]

        hsingle = expanded_single(ivec, n.gyro, mfield, n['Q'], n.detuning)

        if others is not None and other_states is not None:
            hsingle += overhauser_bath(ivec, n['xyz'], n.gyro, others.gyro, others['xyz'], other_states)

        hhyperfine = 0

        for i in range(center.size):
            if center.size == 1:
                hf = n['A']
            else:
                hf = n['A'][i]

            hhyperfine += hyperfine(hf, vectors[bath.size + i], ivec)

        totalh += hsingle + hhyperfine

    return Hamiltonian(dims, vectors, data=totalh)


def central_hamiltonian(center, magnetic_field, bath=None, bath_state=None):
    dims, vectors = dimensions_spinvectors(central_spin=center)
    totalh = 0

    for i, c in enumerate(center):
        totalh += self_central(vectors[i], magnetic_field, c.zfs, c.gyro)

        if bath is not None and bath_state is not None:
            if center.size == 1:
                hf = bath['A']
            else:
                hf = bath['A'][:, i]

            totalh += overhauser_central(vectors[i], hf, bath_state)

    totalh += center_interactions(center, vectors)

    return Hamiltonian(dims, vectors, data=totalh)

# def hamiltonian_wrapper(_func=None, *, projected=False):
#     """
#     Wrapper with the general structure for the cluster Hamiltonian.
#     Adds several additional arguments to the wrapped function.
#
#     **Additional parameters**:
#
#         * **central_spin** (*float*) -- value of the central spin.
#
#     Args:
#         _func (func): Function to be wrapped.
#         projected (bool): True if return two projected Hamiltonians, False if remove single not projected one.
#
#     Returns:
#         func: Wrapped function.
#     """
#
#     def inner_hamiltonian_wrapper(function):
#
#         def base_hamiltonian(bath, *arg,
#                              central_spin=None,
#                              **kwargs):
#             if projected:
#                 dim, spinvectors = dimensions_spinvectors(bath, central_spin=None)
#             else:
#                 dim, spinvectors = dimensions_spinvectors(bath, central_spin=central_spin)
#
#             clusterint = bath_interactions(bath, spinvectors)
#
#             if projected:
#                 halpha, hbeta = Hamiltonian(dim, vectors=spinvectors), Hamiltonian(dim, vectors=spinvectors)
#
#                 data1, data2 = function(bath, spinvectors, central_spin, *arg, **kwargs)
#
#                 halpha.data += data1 + clusterint
#                 hbeta.data += data2 + clusterint
#
#                 return halpha, hbeta
#
#             totalh = Hamiltonian(dim)
#             data = function(bath, spinvectors, central_spin, *arg, **kwargs)
#             totalh.data += data + clusterint
#
#             return totalh
#
#         return base_hamiltonian
#
#     if _func is None:
#         return inner_hamiltonian_wrapper
#     else:
#         return inner_hamiltonian_wrapper(_func)

