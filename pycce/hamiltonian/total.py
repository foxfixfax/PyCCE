import functools

from pycce.utilities import *
from .base import Hamiltonian
from .functions import *


def hamiltonian_wrapper(_func=None, *, projected=False):
    """
    Wrapper with the general structure for the cluster Hamiltonian.
    Adds several additional arguments to the wrapped function.

    **Additional parameters**:

        * **central_spin** (*float*) -- value of the central spin.

    Args:
        _func (func): Function to be wrapped.
        projected (bool): True if return two projected Hamiltonians, False if remove single not projected one.

    Returns:
        func: Wrapped function.
    """

    def inner_hamiltonian_wrapper(function):

        @functools.wraps(function)
        def base_hamiltonian(bath, *arg,
                             central_spin=None,
                             **kwargs):

            dim, spinvectors = dimensions_spinvectors(bath, central_spin=central_spin)

            clusterint = bath_interactions(bath, spinvectors)

            if projected:
                halpha, hbeta = Hamiltonian(dim, vectors=spinvectors), Hamiltonian(dim, vectors=spinvectors)

                data1, data2 = function(bath, spinvectors, *arg, **kwargs)

                halpha.data += data1 + clusterint
                hbeta.data += data2 + clusterint

                return halpha, hbeta

            totalh = Hamiltonian(dim)
            data = function(bath, spinvectors, *arg, **kwargs)
            totalh.data += data + clusterint

            return totalh

        return base_hamiltonian

    if _func is None:
        return inner_hamiltonian_wrapper
    else:
        return inner_hamiltonian_wrapper(_func)


@hamiltonian_wrapper(projected=True)
def projected_hamiltonian(bath, vectors, projections_alpha, projections_beta, mfield,
                          others=None, other_states=None,
                          energy_alpha=None, energy_beta=None,
                          energies=None, projections_alpha_all=None, projections_beta_all=None):
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
    halpha = 0
    hbeta = 0

    for ivec, n in zip(vectors, bath):
        hsingle = expanded_single(ivec, n.gyro, mfield, n['Q'], n.detuning)

        if others is not None and other_states is not None:
            hsingle += overhauser_bath(ivec, n['xyz'], n.gyro, others.gyro,
                                       others['xyz'], other_states)

        hf_alpha = conditional_hyperfine(n['A'], ivec, projections_alpha)
        hf_beta = conditional_hyperfine(n['A'], ivec, projections_beta)

        halpha += hsingle + hf_alpha
        hbeta += hsingle + hf_beta

    if energies is not None:
        halpha += bath_mediated(bath, vectors, energy_alpha,
                                energies, projections_alpha_all)

        hbeta += bath_mediated(bath, vectors, energy_beta,
                               energies, projections_beta_all)

    return halpha, hbeta


@hamiltonian_wrapper
def total_hamiltonian(bath, vectors, mfield, zfs=None, others=None, other_states=None, central_gyro=ELECTRON_GYRO):
    """
    Compute total Hamiltonian for the given cluster including mean field effect of all bath spins.
    Wrapped function so the actual call does not follow the one above!

    Args:
        bath (BathArray):
            array of all bath spins.
        mfield (ndarray with shape (3,)):
            Magnetic field of type ``mfield = np.array([Bx, By, Bz])``.
        others (BathArray with shape (m,)):
            array of all bath spins outside the cluster
        other_states (ndarray with shape (m,) or (m, 3)):
            Array of Iz projections for each bath spin outside of the given cluster.
        zfs (ndarray with shape (3,3)):
            Zero Field Splitting tensor of the central spin.
        central_gyro(float or ndarray with shape (3,3)):
            gyromagnetic ratio of the central spin OR tensor corresponding to interaction between magnetic field and
            central spin.
        central_spin (float): value of the central spin.

    Returns:
        Hamiltonian: hamiltonian of the given cluster, including central spin.

    """

    totalh = self_central(vectors[-1], mfield, zfs, central_gyro)

    if others is not None and other_states is not None:
        totalh += overhauser_central(vectors[-1], others['A'], other_states)

    for j, n in enumerate(bath):
        ivec = vectors[j]

        hsingle = expanded_single(ivec, n.gyro, mfield, n['Q'], n.detuning)

        if others is not None and other_states is not None:
            hsingle += overhauser_bath(ivec, n['xyz'], n.gyro, others.gyro, others['xyz'], other_states)

        hhyperfine = hyperfine(n['A'], vectors[-1], ivec)

        totalh += hsingle + hhyperfine

    return totalh
