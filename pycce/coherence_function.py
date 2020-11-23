import numpy as np

from .cluster_expansion import cluster_expansion_decorator
from .hamiltonian import projected_hamiltonian


def propagators(timespace, H0, H1, N):
    """
    Function to compute propagators U0 and U1 in conventional CCE
    @param timespace: ndarray
        Time delay values at which to compute propagators
    @param H0: ndarray
        Hamiltonian projected on alpha qubit state
    @param H1: ndarray
        Hamiltonian projected on beta qubit state
    @param N: int
        number of pulses in CPMG
    @return: U0, U1
    """
    eval0, evec0 = np.linalg.eigh(H0)
    eval1, evec1 = np.linalg.eigh(H1)

    eigen_exp0 = np.exp(-1j * np.tensordot(timespace,
                                           eval0, axes=0), dtype=np.complex128)
    eigen_exp1 = np.exp(-1j * np.tensordot(timespace,
                                           eval1, axes=0), dtype=np.complex128)

    v0 = np.matmul(np.einsum('ij,kj->kij', evec0, eigen_exp0,
                             dtype=np.complex128),
                   evec0.conj().T, dtype=np.complex128)

    v1 = np.matmul(np.einsum('ij,kj->kij', evec1, eigen_exp1,
                             dtype=np.complex128),
                   evec1.conj().T, dtype=np.complex128)

    if N == 0:
        return v0, v1

    V0_HE = np.matmul(v0, v1, dtype=np.complex128)
    V1_HE = np.matmul(v1, v0, dtype=np.complex128)

    if N == 1:
        return V0_HE, V1_HE

    V0 = np.matmul(V0_HE, V1_HE, dtype=np.complex128)  # v0 @ v1 @ v1 @ v0
    V1 = np.matmul(V1_HE, V0_HE, dtype=np.complex128)  # v1 @ v0 @ v0 @ v1

    U0 = np.linalg.matrix_power(V0, N // 2)
    U1 = np.linalg.matrix_power(V1, N // 2)

    if N % 2 == 1:
        U0 = np.linalg.matmul(U0, V0_HE)
        U1 = np.linalg.matmul(U1, V1_HE)

    return U0, U1


def computeL(H0, H1, timespace, N, as_delay=False):
    """
    Function to compute cluster L in conventional CCE
    @param H0: ndarray
        Hamiltonian projected on alpha qubit state
    @param H1: ndarray
        Hamiltonian projected on beta qubit state
    @param timespace: ndarray
        Time points at which to compute L
    @param N: int
        number of pulses in CPMG
    @param as_delay: bool
        True if time points are delay between pulses,
        False if time points are total time
    @return: L
    """
    # if timespace was given not as delay between pulses,
    # divide to obtain the delay
    if not as_delay and N > 0:
        timespace = timespace / (2 * N)

    U0, U1 = propagators(timespace, H0, H1, N)

    # L = np.trace(np.matmul(U0, np.transpose(
    #     U1.conj(), axes=(0, 2, 1))), axis1=1, axis2=2) / U0.shape[1]
    # L is computed as Tr[rho U0 U1dagger]; rho = Identity / dim
    L = np.einsum('zij,zij->z', U0, U1.conj()) / U0.shape[1]
    return L


def cluster_L(subclusters, nspin, ntype, I, S, B, timespace, N, as_delay=False):
    """
    Older version of the cluster expansion, specifically for L in conventional CCE
    @param subclusters: dict
        dict of subclusters included in different CCE order
        of structure {int order: np.array([[i,j],[i,j]])}
    @param nspin: ndarray with shape (n_atoms)
        array of all atoms (dtype contains N, xyz, A, V)
    @param ntype: dict
        dict with NSpinType objects inside, each key - name of the isotope
    @param I: dict
        dict with SpinMatrix objects inside, each key - spin
    @param S: QSpinMatrix
        QSpinMatrix of the central spin
    @param B: ndarray
        Magnetic field of B = np.array([Bx, By, Bz])
    @param timespace: ndarray with shape (n_points, )
        Time points at which to compute L. n_points is number of time points
    @param N: int
        number of pulses in CPMG
    @param as_delay: bool
        True if time points are delay between pulses,
        False if time points are total time
    @return: L
        L computed with conventional CCE
    """
    # List of orders from highest to lowest
    revorders = sorted(subclusters)[::-1]
    norders = len(revorders)

    # If there is only one set of indexes for only one order,
    # Then for this subcluster nelements < maximum CCE order
    if norders == 1 and subclusters[revorders[0]].shape[0] == 1:
        verticles = subclusters[revorders[0]][0]

        H0, H1 = projected_hamiltonian(nspin[verticles], ntype, I, S, B)
        L = computeL(H0, H1, timespace, N, as_delay=as_delay)

        return L

    # The Highest possible L will have all powers of 1
    power = {}

    # Number of visited orders from highest to lowest
    visited = 0
    L = np.ones(timespace.shape, dtype=np.complex128)
    for order in revorders:

        power[order] = np.ones(subclusters[order].shape[0], dtype=np.int32)
        # indexes of the cluster of size order are stored in v

        for index in range(subclusters[order].shape[0]):

            v = subclusters[order][index]
            # First, find the correct power. Iterate over all higher orders
            for higherorder in revorders[:visited]:
                # np.isin gives bool array of shape subclusters[higherorder],
                # which is np.array of
                # indexes of subclusters with order = higherorder.
                # Entries are True if value is
                # present in v and False if values are not present in v.
                # Sum bool entries in inside cluster,
                # if the sum equal to size of v,
                # then v is inside the given subcluster.
                # containv is 1D bool array with values of i-element True
                # if i-subcluster of
                # subclusters[higherorder] contains v
                containv = np.count_nonzero(
                    np.isin(subclusters[higherorder], v), axis=1) == v.size

                # Power of cluster v is decreased by sum of powers of all higher orders,
                # As all of them have to be divided by v
                power[order][index] -= np.sum(power[higherorder]
                                              [containv], dtype=np.int32)

            H0, H1 = projected_hamiltonian(nspin[v], ntype, I, S, B)

            Lv = computeL(H0, H1, timespace, N,
                          as_delay=as_delay) ** power[order][index]
            L *= Lv

        visited += 1
        print('Computed Ls of order {} for subcluster of size {}'.format(
            order, subclusters[1].size))
    return L


@cluster_expansion_decorator
def decorated_coherence_function(nspin, ntype, I, S, B, timespace, N, as_delay=False):
    """
        Overarching decorated function to compute L in conventional CCE. The call of the function includes:
    @param subclusters: dict
        dict of subclusters included in different CCE order
        of structure {int order: np.array([[i,j],[i,j]])}
    @param allnspin: ndarray
        array of all atoms
    @param ntype: dict
        dict with NSpinType objects inside, each key - name of the isotope
    @param I: dict
        dict with SpinMatrix objects inside, each key - spin
    @param S: QSpinMatrix
        QSpinMatrix of the central spin
    @param B: ndarray
        Magnetic field of B = np.array([Bx, By, Bz])
    @param timespace: ndarray
        Time points at which to compute L
    @param N: int
        number of pulses in CPMG
    @param as_delay: bool
        True if time points are delay between pulses,
        False if time points are total time
    @return: L
        L computed with conventional CCE
    """
    H0, H1 = projected_hamiltonian(nspin, ntype, I, S, B)
    L = computeL(H0, H1, timespace, N, as_delay=as_delay)
    return L
