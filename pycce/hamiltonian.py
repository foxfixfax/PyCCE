import numpy as np

from .units import HBAR, ELECTRON_GYRO


# HBAR = 1.05457172  # When everything else in rad, kHz, ms, G, A


class SpinMatrix:
    """
    Class containing the spin matrices in Sz basis

    Parameters
    ----------
    @param s: float
    total spin
    """

    def __init__(self, s):
        dim = np.int(2 * s + 1 + 1e-8)

        projections = np.linspace(-s, s, dim, dtype=np.complex128)

        plus = np.zeros((dim, dim), dtype=np.complex128)

        for i in range(dim - 1):
            plus[i, i + 1] += np.sqrt(s * (s + 1) -
                                      projections[i] * projections[i + 1])

        minus = plus.conj().T

        self.s = s
        self.dim = dim

        self.x = 1 / 2. * (plus + minus)
        self.y = 1 / 2j * (plus - minus)
        self.z = np.diag(projections[::-1])

        self.eye = np.eye(dim, dtype=np.complex128)

    def __repr__(self):
        return "Spin-{:.1f} matrices x, y, z".format(self.s)


class QSpinMatrix(SpinMatrix):
    """
    Class containing the spin matrices in Sz basis for central spin

    Parameters
    ----------
    @param s: float
        total spin
    @param alpha: ndarray
        0 state of the qubit
    @param beta: ndarray
        1 state of the qubit
    """

    def __init__(self, s, alpha, beta):
        super().__init__(s)

        alpha = np.asarray(alpha)
        beta = np.asarray(beta)
        self.alpha = alpha
        self.beta = beta

        self.projections_alpha = np.array([alpha.conj() @ self.x @ alpha,
                                           alpha.conj() @ self.y @ alpha,
                                           alpha.conj() @ self.z @ alpha],
                                          dtype=np.complex128)

        self.projections_beta = np.array([beta.conj() @ self.x @ beta,
                                          beta.conj() @ self.y @ beta,
                                          beta.conj() @ self.z @ beta],
                                         dtype=np.complex128)

    def get_projection(self, state):
        if np.all(state == self.alpha):
            return self.projections_alpha

        elif np.all(state == self.beta):
            return self.projections_beta
        else:
            raise KeyError('There is no such qubit state!')


def generate_spinmatrices(ntype):
    """
    Generate spin matrices for all bath spin types
    @param ntype: dict
        dict of SpinType objects
    @return: dict
        dict with keys corresponding total spin, values - SpinMatrix objects
    """
    nmatrices = {}

    for N in ntype:
        nmatrices[ntype[N].s] = SpinMatrix(ntype[N].s)

    return nmatrices


def expand(M, i, dim):
    """
    Expand matrix M from it's own dimensions to the total Hilbert space
    @param M: ndarray
        Inital matrix
    @param i: int
        Index of the spin in dim
    @param dim: list
        list of dimensions of all spins present in the cluster
    @return: ndarray
        Expanded matrix
    """
    dbefore = np.prod(dim[:i], dtype=np.int32)
    dafter = np.prod(dim[i + 1:], dtype=np.int32)

    M_expanded = np.kron(np.kron(np.eye(dbefore, dtype=np.complex128), M),
                         np.eye(dafter, dtype=np.complex128))

    return M_expanded


def zeeman(gyro, spin_matrix, B):
    """
    Zeeman interactions of the n spin
    @param n: np.void object with dtype _dtype_bath
        single bath spin
    @param ntype: dict
        dict of SpinTypes
    @param I: dict
        dict of SpinMatrix objects
    @param B: array_like
        magnetic field as (Bx, By, Bz)
    @return: ndarray of shape (2s+1, 2s+1)
    """
    BI = B[0] * spin_matrix.x + B[1] * spin_matrix.y + B[2] * spin_matrix.z

    H_zeeman = - gyro * BI

    return H_zeeman


def quadrupole(quadrupole_tensor, s, spin_matrix):
    """
    Quadrupole interaction of the n spin (for s>1)
    @param n:  np.void object with dtype _dtype_bath
        single bath spin
    @param ntype: dict
        dict of SpinTypes
    @param I: dict
        dict of SpinMatrix objects
    @return: ndarray of shape (2s+1, 2s+1)
    """
    Iv = np.asarray([spin_matrix.x, spin_matrix.y, spin_matrix.z])

    # VIvec = np.einsum('ij,jkl->ikl', n['Q'], Iv, dtype=np.complex128)
    # IQI = np.einsum('lij,ljk->ik', Iv, VIvec, dtype=np.complex128)
    IQI = np.einsum('lij,lp,pjk->ik', Iv, quadrupole_tensor, Iv, dtype=np.complex128)

    delI2 = np.sum(np.diag(quadrupole_tensor)) * np.eye(spin_matrix.x.shape[0]) * s * (s + 1)

    H_quad = IQI - delI2 / 3

    return H_quad


def dipole_dipole(coord_1, coord_2, g1, g2, Ivec_1, Ivec_2):
    """
    Compute dipole_dipole interactions between two bath spins
    @param nuclei: ndarray with shape (2,)
        two bath spins
    @param Ivec_1: ndarray with shape (3, d, d)
        where d is total size of the Hilbert space. Vector of [Ix, Iy, Iz] for the first bath spin
    @param Ivec_2: ndarray with shape (3, d, d)
        where d is total size of the Hilbert space. Vector of [Ix, Iy, Iz] for the second bath spin
    @param ntype: dict
        dict of SpinTypes
    @return: ndarray of shape (d, d)
    """
    pre = g1 * g2 * HBAR

    pos = coord_1 - coord_2
    r = np.linalg.norm(pos)

    PTensor = -pre * (3 * np.outer(pos, pos) -
                      np.eye(3, dtype=np.complex128) * r ** 2) / (r ** 5)

    # PIvec = np.einsum('ij,jkl->ikl', PTensor, Ivec_2,
    #                   dtype=np.complex128)  # PIvec = Ptensor @ Ivector
    # H_DD = np.einsum('lij,ljk->ik', Ivec_1, PIvec, dtype=np.complex128)

    # DD = IPI = IxPxxIx + IxPxyIy + ..
    H_DD = np.einsum('lij,lp,pjk->ik', Ivec_1, PTensor, Ivec_2, dtype=np.complex128)

    return H_DD


def projected_hyperfine(hyperfine_tensor, spin_matrix, projections):
    """
    Compute projected hyperfine Hamiltonian for one state of the central spin
    @param n: np.void object with dtype _dtype_bath
        single bath spin
    @return: ndarray of shape (d, d)
    """

    A_projected = projections @ hyperfine_tensor  # S.get_projection(state)
    H_Hyperfine = (A_projected[0] * spin_matrix.x +
                   A_projected[1] * spin_matrix.y +
                   A_projected[2] * spin_matrix.z)

    return H_Hyperfine


def projected_hamiltonian(nspin, I, S, B):
    """
    Compute projected hamiltonian on alpha and beta qubit states
    @param nspin: BathArray with shape (n,)
        ndarray of bath spins in the given cluster with size n
    @param ntype: dict
        dict of SpinTypes
    @param I: dict
        dict with SpinMatrix objects inside, each key - spin
    @param S: QSpinMatrix
        QSpinMatrix of the central spin
    @param B: ndarray with shape (3,)
        magnetic field of format (Bx, By, Bz)
    @return: H_alpha, H_beta
    """
    ntype = nspin.types
    dimensions = [I[ntype[n['N']].s].dim for n in nspin]
    nnuclei = nspin.shape[0]

    tdim = np.prod(dimensions, dtype=np.int32)

    H_alpha = np.zeros((tdim, tdim), dtype=np.complex128)
    H_beta = np.zeros((tdim, tdim), dtype=np.complex128)

    ivectors = []

    for j, n in enumerate(nspin):
        s = ntype[n].s

        if s > 1 / 2:
            H_quad = quadrupole(n['Q'], s, I[s])
            H_single = zeeman(ntype[n].gyro, I[s], B) + H_quad
        else:
            H_single = zeeman(ntype[n].gyro, I[s], B)

        H_HF_alpha = projected_hyperfine(n['A'], I[s], S.projections_alpha)
        H_HF_beta = projected_hyperfine(n['A'], I[s], S.projections_beta)

        H_j_alpha = H_single + H_HF_alpha
        H_j_beta = H_single + H_HF_beta

        H_alpha += expand(H_j_alpha, j, dimensions)
        H_beta += expand(H_j_beta, j, dimensions)

        ivec = np.array([expand(I[s].x, j, dimensions),
                         expand(I[s].y, j, dimensions),
                         expand(I[s].z, j, dimensions)],
                        dtype=np.complex128)

        ivectors.append(ivec)

    for i in range(nnuclei):
        for j in range(i + 1, nnuclei):
            n1 = nspin[i]
            n2 = nspin[j]

            ivec_1 = ivectors[i]
            ivec_2 = ivectors[j]

            H_DD = dipole_dipole(n1['xyz'], n2['xyz'], ntype[n1].gyro, ntype[n2].gyro, ivec_1, ivec_2)

            H_alpha += H_DD
            H_beta += H_DD

    return H_alpha, H_beta


def hyperfine(hyperfine_tensor, svec, ivec):
    """
    Compute hyperfine interactions between central spin S and bath spin I
    @param n: np.void object with dtype _dtype_bath
        single bath spin
    @param svec: ndarray with shape (3, d, d)
        d is the total size of the Hilbert space. [Sx, Sy, Sz] array of the central spin
    @param ivec: ndarray with shape (3, d, d)
        d is the total size of the Hilbert space. [Ix, Iy, Iz] array of the bath spin
    @return: ndarray
    """
    # AIvec = np.einsum('ij,jkl->ikl', ATensor, ivec,
    #                   dtype=np.complex128)  # AIvec = Atensor @ Ivector
    # HF = SPI = SxPxxIx + SxPxyIy + ..
    # H_HF = np.einsum('lij,ljk->ik', svec, AIvec, dtype=np.complex128)
    H_HF = np.einsum('lij,lp,pjk->ik', svec, hyperfine_tensor, ivec, dtype=np.complex128)
    return H_HF


def self_electron(B, S, D, E=0, gyro_e=ELECTRON_GYRO):
    """
    central spin Hamiltonian
    @param B: ndarray with shape (3,)
        magnetic field of format (Bx, By, Bz)
    @param S: QSpinMatrix
        QSpinMatrix of the central spin
    @param gyro_e: float
        gyromagnetic ratio (in rad/(msec*Gauss)) of the central spin    @return:
    @param D: float or ndarray with shape (3,3)
        D parameter in central spin ZFS OR total ZFS tensor
    @param E: float
        E parameter in central spin ZFS
    @return: ndarray
    """
    if isinstance(D, (np.floating, float, int)):
        H0 = D * (S.z @ S.z - 1 / 3 * S.s * (S.s + 1) * S.eye) + \
             E * (S.x @ S.x - S.y @ S.y)
    else:
        Svec = np.asarray([S.x, S.y, S.z], dtype=np.complex128)
        H0 = np.einsum('lij,lp,pjk->ik', Svec, D, Svec, dtype=np.complex128)

    H1 = -gyro_e * (B[0] * S.x + B[1] * S.y + B[2] * S.z)

    return H1 + H0


def total_hamiltonian(nspin, I, S, B, D, E=0, gyro_e=ELECTRON_GYRO):
    """
    Total hamiltonian for cluster including central spin
    @param nspin: ndarray with shape (n,)
        ndarray of bath spins in the given cluster with size n
    @param ntype: dict
        dict of SpinTypes
    @param I: dict
        dict with SpinMatrix objects inside, each key - spin
    @param S: QSpinMatrix
        QSpinMatrix of the central spin
    @param B: ndarray with shape (3,)
        magnetic field of format (Bx, By, Bz)
    @param gyro_e: float
        gyromagnetic ratio (in rad/(msec*Gauss)) of the central spin    @return:
    @param D: float or ndarray with shape (3,3)
        D parameter in central spin ZFS OR total ZFS tensor
    @param E: float
        E parameter in central spin ZFS
    @return: H, dimensions
        H: ndarray with shape (prod(dimensions), prod(dimensions))
        dimensions: list of dimensions for each spin, last entry - dimensions of central spin
    """
    ntype = nspin.types
    dimensions = [I[ntype[n].s].dim for n in nspin] + [S.dim]
    nnuclei = nspin.shape[0]

    tdim = np.prod(dimensions, dtype=np.int32)
    H = np.zeros((tdim, tdim), dtype=np.complex128)

    H_electron = self_electron(B, S, D, E, gyro_e)
    svec = np.array([expand(S.x, nnuclei, dimensions),
                     expand(S.y, nnuclei, dimensions),
                     expand(S.z, nnuclei, dimensions)],
                    dtype=np.complex128)

    H += expand(H_electron, nnuclei, dimensions)

    ivectors = []

    for j, n in enumerate(nspin):
        s = ntype[n].s
        ivec = np.array([expand(I[s].x, j, dimensions),
                         expand(I[s].y, j, dimensions),
                         expand(I[s].z, j, dimensions)],
                        dtype=np.complex128)
        if s > 1 / 2:
            H_quad = quadrupole(n['Q'], s, I[s])
            H_single = zeeman(ntype[n].gyro, I[s], B) + H_quad
        else:
            H_single = zeeman(ntype[n].gyro, I[s], B)

        H_HF = hyperfine(n['A'], svec, ivec)

        H += expand(H_single, j, dimensions) + H_HF

        ivectors.append(ivec)

    for i in range(nnuclei):
        for j in range(i + 1, nnuclei):
            n1 = nspin[i]
            n2 = nspin[j]

            ivec_1 = ivectors[i]
            ivec_2 = ivectors[j]

            H_DD = dipole_dipole(n1['xyz'], n2['xyz'], ntype[n1].gyro, ntype[n2].gyro, ivec_1, ivec_2)

            H += H_DD

    return H, dimensions


def mf_electron(S, others, others_state):
    """
    compute mean field effect for all bath spins not included in the cluster
    on central spin
    @param S: QSpinMatrix
        QSpinMatrix of the central spin
    @param others: ndarray of shape (n_bath - n_cluser,)
        ndarray of all bath spins not included in the cluster
    @param others_state: ndarray of shape (n_bath - n_cluser,)
        Sz projections of the state of all others nuclear spins not included in the given cluster
    @return: ndarray
    """
    # xfield = np.sum(others['A'][:, 2, 0] * others_state)
    # yfield = np.sum(others['A'][:, 2, 1] * others_state)
    zfield = np.sum(others['A'][:, 2, 2] * others_state)

    H_mf = zfield * S.z  # + xfield * S.x + yfield * S.y

    return H_mf


def mf_nucleus(n, g, gyros, spin_matrix, others, others_state):
    """
    compute mean field effect on the bath spin n from all other bath spins
    @param n: np.void object with dtype _dtype_bath
        single bath spin
    @param ntype: dict
        dict of SpinTypes
    @param I: dict
        dict with SpinMatrix objects inside, each key - spin
    @param others: ndarray of shape (n_bath - n_cluser,)
        ndarray of all bath spins not included in the cluster
    @param others_state: ndarray of shape (n_bath - n_cluser,)
        Sz projections of the state of all others nuclear spins not included in the given cluster
    @return: ndarray
    """
    pre = g * gyros * HBAR

    pos = n['xyz'] - others['xyz']
    r = np.linalg.norm(pos, axis=1)
    cos_theta = pos[:, 2] / r

    zfield = np.sum(pre / r ** 3 * (1 - cos_theta ** 2) * others_state)
    H_mf = zfield * spin_matrix.z

    return H_mf


def mf_hamiltonian(nspin, I, S, B, others, others_state, D, E=0, gyro_e=ELECTRON_GYRO):
    """
    compute total Hamiltonian for the given cluster including mean field effect of all nuclei
    outside of the given cluster
    @param nspin: ndarray with shape (n,)
        ndarray of bath spins in the given cluster with size n
    @param ntype: dict
        dict of SpinTypes
    @param I: dict
        dict with SpinMatrix objects inside, each key - spin
    @param S: QSpinMatrix
        QSpinMatrix of the central spin
    @param B: ndarray with shape (3,)
        magnetic field of format (Bx, By, Bz)
    @param gyro_e: float
        gyromagnetic ratio (in rad/(msec*Gauss)) of the central spin    @return:
    @param D: float or ndarray with shape (3,3)
        D parameter in central spin ZFS OR total ZFS tensor
    @param E: float
        E parameter in central spin ZFS
    @param others: ndarray of shape (n_bath - n_cluser,)
        ndarray of all bath spins not included in the cluster
    @param others_state: ndarray of shape (n_bath - n_cluser,)
        Sz projections of the state of all others nuclear spins not included in the given cluster
    @return: H, dimensions
        H: ndarray with shape (prod(dimensions), prod(dimensions)) hamiltonian
        dimensions: list of dimensions for each spin, last entry - dimensions of central spin
    """
    ntype = nspin.types
    dimensions = [I[ntype[n['N']].s].dim for n in nspin] + [S.dim]
    nnuclei = nspin.shape[0]

    tdim = np.prod(dimensions, dtype=np.int32)
    H = np.zeros((tdim, tdim), dtype=np.complex128)

    H_electron = self_electron(B, S, D, E, gyro_e) + mf_electron(S, others, others_state)

    svec = np.array([expand(S.x, nnuclei, dimensions),
                     expand(S.y, nnuclei, dimensions),
                     expand(S.z, nnuclei, dimensions)],
                    dtype=np.complex128)

    H += expand(H_electron, nnuclei, dimensions)

    ivectors = []

    for j, n in enumerate(nspin):
        s = ntype[n].s
        ivec = np.array([expand(I[s].x, j, dimensions),
                         expand(I[s].y, j, dimensions),
                         expand(I[s].z, j, dimensions)],
                        dtype=np.complex128)

        H_mf_nucleus = mf_nucleus(n, ntype[n].gyro, ntype[others].gyro,
                                  I[s], others, others_state)

        H_zeeman = zeeman(ntype[n].gyro, I[s], B)

        if s > 1 / 2:
            H_quad = quadrupole(n['Q'], s, I[s])
            H_single = H_zeeman + H_quad + H_mf_nucleus
        else:
            H_single = H_zeeman + H_mf_nucleus

        H_HF = hyperfine(n['A'], svec, ivec)

        H += expand(H_single, j, dimensions) + H_HF

        ivectors.append(ivec)

    for i in range(nnuclei):
        for j in range(i + 1, nnuclei):
            n1 = nspin[i]
            n2 = nspin[j]

            ivec_1 = ivectors[i]
            ivec_2 = ivectors[j]

            H_DD = dipole_dipole(n1['xyz'], n2['xyz'], ntype[n1].gyro, ntype[n2].gyro, ivec_1, ivec_2)

            H += H_DD

    return H, dimensions


def eta_hamiltonian(nspin, I, S, eta):
    """
    EXPERIMENTAL. Compute hamiltonian with eta-term - gradually turn off or turn on the secular interactions for
    alpha and beta qubit states
    @param nspin: ndarray with shape (n,)
        ndarray of bath spins in the given cluster with size n
    @param ntype: dict
        dict of SpinTypes
    @param I: dict
        dict with SpinMatrix objects inside, each key - spin
    @param S: QSpinMatrix
        QSpinMatrix of the central spin
    @param eta: value of dimensionless parameter eta (from 0 to 1)
    @return:
    """
    ntype = nspin.types
    nnuclei = nspin.shape[0]
    dimensions = [I[ntype[n['N']].s].dim for n in nspin] + [S.dim]

    AIzi = 0
    for j in range(nnuclei):
        s = ntype[nspin[j]['N']].s
        Ivec = np.array([expand(I[s].x, j, dimensions),
                         expand(I[s].y, j, dimensions),
                         expand(I[s].z, j, dimensions)],
                        dtype=np.complex128)

        AIzi += np.einsum('j,jkl->kl', nspin[j]['A'][2, :], Ivec, dtype=np.complex128)

    up_down = (1 - eta) / 2 * (np.tensordot(S.alpha, S.alpha, axes=0) + np.tensordot(S.beta, S.beta, axes=0))
    H_eta = expand(up_down, nnuclei, dimensions) @ AIzi
    return H_eta
