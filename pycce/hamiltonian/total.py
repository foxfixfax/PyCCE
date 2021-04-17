import functools

from pycce.utilities import *
from .base import Hamiltonian
from .functions import *


def hamiltonian_wrapper(_func=None, *, projected=False):
    def inner_hamiltonian_wrapper(function):

        @functools.wraps(function)
        def base_hamiltonian(bath, *arg,
                             central_spin=None,
                             imap=None, map_error=None,
                             **kwargs):
            """

            :param bath: BathArray with shape (n,)
                ndarray of bath spins
            :param arg:
                additional arguments, defined by inner function
            :param central_spin: float, optional
                if provided, gives total spin of the central spin
            :param imap: InteractionMap
                optional. dictionary-like object containing tensors for all bath spin pairs
            :param map_error: bool
                optional. If true and imap is not None, raises error when cannot find pair of nuclear spins in imap.
                Default False
            :param kwargs:
            :return:
            """
            dim, spinvectors = dimensions_spinvectors(bath, central_spin=central_spin)
            clusterint = bath_interactions(bath, spinvectors, imap=imap, raise_error=map_error)

            if projected:
                halpha, hbeta = Hamiltonian(dim), Hamiltonian(dim)

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
    """
    Compute projected hamiltonian on state and beta qubit states
    :param projections_alpha: np.ndarray with shape (3,)
        projections of the central spin state level [<Sx>, <Sy>, <Sz>]
    :param projections_beta: np.ndarray with shape (3,)
        projections of the central spin beta level [<Sx>, <Sy>, <Sz>]
    :param mfield: ndarray with shape (3,)
        magnetic field of format (Bx, By, Bz)

    :return: H_alpha, H_beta
    """
    ntype = bath.types

    halpha = 0
    hbeta = 0

    for ivec, n in zip(vectors, bath):
        hsingle = expanded_single(ivec, ntype[n].gyro, mfield, n['Q'])

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
def total_hamiltonian(bath, vectors, mfield, zfs, central_gyro=ELECTRON_GYRO):
    """
    Total hamiltonian for cluster including central spin
    :param nspin: ndarray with shape (n,)
        ndarray of bath spins in the given cluster with size n
    :param central_spin: float
        total spin of the central spin
    :param mfield: ndarray with shape (3,)
        magnetic field of format (Bx, By, Bz)
    :param zfs: ndarray with shape (3,3)
        ZFS tensor
    :param central_gyro: float
        gyromagnetic ratio (in rad/(ms*Gauss)) of the central spin
    :return: H, dimensions
        H: ndarray with shape (prod(dimensions), prod(dimensions))
    """
    ntype = bath.types

    H = self_central(vectors[-1], mfield, zfs, central_gyro)

    for j, n in enumerate(bath):
        ivec = vectors[j]

        H_single = expanded_single(ivec, ntype[n].gyro, mfield, n['Q'])

        H_HF = hyperfine(n['A'], vectors[-1], ivec)

        H += H_single + H_HF

    return H


@hamiltonian_wrapper
def mean_field_hamiltonian(bath, vectors, mfield, others, others_state, D=None, central_gyro=ELECTRON_GYRO):
    """
    compute total Hamiltonian for the given cluster including mean field effect of all nuclei
    outside of the given cluster
    :param bath: ndarray with shape (n,)
        ndarray of bath spins in the given cluster with size n
    :param mfield: ndarray with shape (3,)
        magnetic field of format (Bx, By, Bz)
    :param central_spin: float
        total spin of the central spin
    :param others: ndarray of shape (n_bath - n_cluser,)
        ndarray of all bath spins not included in the cluster
    :param others_state: ndarray of shape (n_bath - n_cluser,)
        Sz projections of the state of all others nuclear spins not included in the given cluster
    :param D: float or ndarray with shape (3,3)
        D parameter in central spin ZFS OR total ZFS tensor
    :param E: float
        E parameter in central spin ZFS
    :param central_gyro: float
        gyromagnetic ratio (in rad/(msec*Gauss)) of the central spin
    :return: H
        H: ndarray with shape (prod(dimensions), prod(dimensions)) hamiltonian
    """
    ntype = bath.types

    totalh = self_central(vectors[-1], mfield, D, central_gyro) + overhauser_central(vectors[-1], others['A'],
                                                                                     others_state)

    for j, n in enumerate(bath):
        ivec = vectors[j]

        mfbath = overhauser_bath(ivec, n['xyz'], ntype[n].gyro, ntype[others].gyro, others['xyz'], others_state)
        hsingle = expanded_single(ivec, ntype[n].gyro, mfield, n['Q']) + mfbath
        hhyperfine = hyperfine(n['A'], vectors[-1], ivec)

        totalh += hsingle + hhyperfine

    return totalh
