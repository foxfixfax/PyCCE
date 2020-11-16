import numpy as np

hbar = 1.05457172  # When everything else in rad, kHz, ms, G, A


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


def zeeman(n, ntype, I, B):
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
    s = ntype[n['N']].s

    BI = B[0] * I[s].x + B[1] * I[s].y + B[2] * I[s].z

    H_zeeman = - ntype[n['N']].gyro * BI

    return H_zeeman


def quadrupole(n, ntype, I):
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
    s = ntype[n['N']].s
    Iv = np.asarray([I[s].x, I[s].y, I[s].z])

    # VIvec = np.einsum('ij,jkl->ikl', n['V'], Iv, dtype=np.complex128)
    # IVI = np.einsum('lij,ljk->ik', Iv, VIvec, dtype=np.complex128)
    IVI = np.einsum('lij,lp,pjk->ik', Iv, n['V'], Iv, dtype=np.complex128)

    pref = ntype[n['N']].q / (6 * s * (2 * s - 1))
    delI2 = np.sum(np.diag(n['V'])) * np.eye(I[s].x.shape[0]) * s * (s + 1)

    H_quad = pref * (3 * IVI - delI2)

    return H_quad


def dipole_dipole(nuclei, Ivec_1, Ivec_2, ntype):
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
    n1 = nuclei[0]
    n2 = nuclei[1]

    g1 = ntype[n1['N']].gyro
    g2 = ntype[n2['N']].gyro

    pre = g1 * g2 * hbar

    pos = n1['xyz'] - n2['xyz']
    r = np.linalg.norm(pos)

    PTensor = -pre * (3 * np.outer(pos, pos) -
                      np.eye(3, dtype=np.complex128) * r ** 2) / (r ** 5)

    # PIvec = np.einsum('ij,jkl->ikl', PTensor, Ivec_2,
    #                   dtype=np.complex128)  # PIvec = Ptensor @ Ivector
    # H_DD = np.einsum('lij,ljk->ik', Ivec_1, PIvec, dtype=np.complex128)

    # DD = IPI = IxPxxIx + IxPxyIy + ..
    H_DD = np.einsum('lij,lp,pjk->ik', Ivec_1, PTensor, Ivec_2, dtype=np.complex128)

    return H_DD


def projected_hyperfine(n, state, ntype, I, S):
    """
    Compute projected hyperfine Hamiltonian for one state of the central spin
    @param n: np.void object with dtype _dtype_bath
        single bath spin
    @param state: ndarray
        central spin state for which to compute the projected hyperfine
    @param ntype: dict
        dict of SpinTypes
    @param I: dict
        dict with SpinMatrix objects inside, each key - spin
    @param S: QSpinMatrix
        QSpinMatrix of the central spin
    @return: ndarray of shape (d, d)
    """
    s = ntype[n['N']].s

    A_projected = S.get_projection(state) @ n['A']
    H_Hyperfine = (A_projected[0] * I[s].x +
                   A_projected[1] * I[s].y +
                   A_projected[2] * I[s].z)

    return H_Hyperfine


def projected_hamiltonian(nspin, ntype, I, S, B):
    """
    Compute projected hamiltonian on alpha and beta qubit states
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
    @return: H_alpha, H_beta
    """
    dimensions = [I[ntype[n['N']].s].dim for n in nspin]
    nnuclei = nspin.shape[0]

    tdim = np.prod(dimensions, dtype=np.int32)

    H_alpha = np.zeros((tdim, tdim), dtype=np.complex128)
    H_beta = np.zeros((tdim, tdim), dtype=np.complex128)

    Ivectors = []

    for j in range(nnuclei):
        s = ntype[nspin[j]['N']].s

        if s > 0.5:
            H_quad = quadrupole(nspin[j], ntype, I)
        else:
            H_quad = 0

        H_zeeman = zeeman(nspin[j], ntype, I, B)

        H_HF_alpha = projected_hyperfine(nspin[j], S.alpha, ntype, I, S)
        H_HF_beta = projected_hyperfine(nspin[j], S.beta, ntype, I, S)

        H_j_alpha = H_zeeman + H_HF_alpha + H_quad
        H_j_beta = H_zeeman + H_HF_beta + H_quad

        H_alpha += expand(H_j_alpha, j, dimensions)
        H_beta += expand(H_j_beta, j, dimensions)

        Ivec = np.array([expand(I[s].x, j, dimensions),
                         expand(I[s].y, j, dimensions),
                         expand(I[s].z, j, dimensions)],
                        dtype=np.complex128)

        Ivectors.append(Ivec)

    for i in range(nnuclei):
        for j in range(i + 1, nnuclei):
            Ivec_1 = Ivectors[i]
            Ivec_2 = Ivectors[j]

            H_DD = dipole_dipole(nspin[(i, j),], Ivec_1, Ivec_2, ntype)

            H_alpha += H_DD
            H_beta += H_DD

    return H_alpha, H_beta


def hyperfine(n, Svec, Ivec):
    """
    Compute hyperfine interactions between central spin S and bath spin I
    @param n: np.void object with dtype _dtype_bath
        single bath spin
    @param Svec: ndarray with shape (3, d, d)
        d is the total size of the Hilbert space. [Sx, Sy, Sz] array of the central spin
    @param Ivec: ndarray with shape (3, d, d)
        d is the total size of the Hilbert space. [Ix, Iy, Iz] array of the bath spin
    @return: ndarray
    """
    ATensor = n['A']
    # AIvec = np.einsum('ij,jkl->ikl', ATensor, Ivec,
    #                   dtype=np.complex128)  # AIvec = Atensor @ Ivector
    # HF = SPI = SxPxxIx + SxPxyIy + ..
    # H_HF = np.einsum('lij,ljk->ik', Svec, AIvec, dtype=np.complex128)
    H_HF = np.einsum('lij,lp,pjk->ik', Svec, ATensor, Ivec, dtype=np.complex128)
    return H_HF


def self_electron(B, S, gyro_e, D, E):
    """
    central spin Hamiltonian
    @param B: ndarray with shape (3,)
        magnetic field of format (Bx, By, Bz)
    @param S: QSpinMatrix
        QSpinMatrix of the central spin
    @param gyro_e: float
        gyromagnetic ratio (in rad/(msec*Gauss)) of the central spin    @return:
    @param D: float
        D parameter in central spin ZFS
    @param E: float
        E parameter in central spin ZFS
    @return: ndarray
    """

    H0 = D * (S.z @ S.z - 1 / 3 * S.s * (S.s + 1) * S.eye) + \
         E * (S.x @ S.x - S.y @ S.y)
    H1 = -gyro_e * (B[0] * S.x + B[1] * S.y + B[2] * S.z)

    return H1 + H0


def total_hamiltonian(nspin, ntype, I, S, B, gyro_e, D, E):
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
    @param D: float
        D parameter in central spin ZFS
    @param E: float
        E parameter in central spin ZFS
    @return: H, dimensions
        H: ndarray with shape (prod(dimensions), prod(dimensions))
        dimensions: list of dimensions for each spin, last entry - dimensions of central spin
    """
    dimensions = [I[ntype[n['N']].s].dim for n in nspin] + [S.dim]
    nnuclei = nspin.shape[0]

    tdim = np.prod(dimensions, dtype=np.int32)
    H = np.zeros((tdim, tdim), dtype=np.complex128)

    H_electron = self_electron(B, S, gyro_e, D, E)
    Svec = np.array([expand(S.x, nnuclei, dimensions),
                     expand(S.y, nnuclei, dimensions),
                     expand(S.z, nnuclei, dimensions)],
                    dtype=np.complex128)

    H += expand(H_electron, nnuclei, dimensions)

    Ivectors = []

    for j in range(nnuclei):
        s = ntype[nspin[j]['N']].s
        Ivec = np.array([expand(I[s].x, j, dimensions),
                         expand(I[s].y, j, dimensions),
                         expand(I[s].z, j, dimensions)],
                        dtype=np.complex128)
        if s > 1 / 2:
            H_quad = quadrupole(nspin[j], ntype, I)
        else:
            H_quad = 0

        H_single = zeeman(nspin[j], ntype, I, B) + H_quad
        H_HF = hyperfine(nspin[j], Svec, Ivec)

        H += expand(H_single, j, dimensions) + H_HF

        Ivectors.append(Ivec)

    for i in range(nnuclei):
        for j in range(i + 1, nnuclei):
            Ivec_1 = Ivectors[i]
            Ivec_2 = Ivectors[j]

            H_dd = dipole_dipole(nspin[(i, j),], Ivec_1, Ivec_2, ntype)

            H += H_dd

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


def mf_nucleus(n, ntype, I, others, others_state):
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
    g = ntype[n['N']].gyro
    s = ntype[n['N']].s

    gyros = np.empty(others.shape, dtype=np.float64)
    for nt in ntype:
        other_g = ntype[nt].gyro
        mask = others['N'] == nt
        gyros[mask] = other_g

    pre = g * gyros * hbar

    pos = n['xyz'] - others['xyz']
    r = np.linalg.norm(pos, axis=1)
    cos_theta = pos[:, 2] / r

    zfield = np.sum(pre / r ** 3 * (1 - cos_theta ** 2) * others_state)
    H_mf = zfield * I[s].z

    return H_mf


def mf_hamiltonian(nspin, ntype, I, S, B, gyro_e, D, E, others, others_state):
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
    @param D: float
        D parameter in central spin ZFS
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
    dimensions = [I[ntype[n['N']].s].dim for n in nspin] + [S.dim]
    nnuclei = nspin.shape[0]

    tdim = np.prod(dimensions, dtype=np.int32)
    H = np.zeros((tdim, tdim), dtype=np.complex128)

    H_electron = self_electron(B, S, gyro_e, D, E) + mf_electron(S, others, others_state)

    Svec = np.array([expand(S.x, nnuclei, dimensions),
                     expand(S.y, nnuclei, dimensions),
                     expand(S.z, nnuclei, dimensions)],
                    dtype=np.complex128)

    H += expand(H_electron, nnuclei, dimensions)
    Ivectors = []

    for j in range(nnuclei):
        s = ntype[nspin[j]['N']].s
        Ivec = np.array([expand(I[s].x, j, dimensions),
                         expand(I[s].y, j, dimensions),
                         expand(I[s].z, j, dimensions)],
                        dtype=np.complex128)

        H_zeeman = zeeman(nspin[j], ntype, I, B)
        H_HF = hyperfine(nspin[j], Svec, Ivec)
        H_mf_nucleus = mf_nucleus(nspin[j], ntype, I, others, others_state)

        if s > 1 / 2:
            H_quad = quadrupole(nspin[j], ntype, I)
            H_single = H_zeeman + H_mf_nucleus + H_quad
        else:
            H_single = H_zeeman + H_mf_nucleus

        H += expand(H_single, j, dimensions) + H_HF

        Ivectors.append(Ivec)

    for i in range(nnuclei):
        for j in range(i + 1, nnuclei):
            Ivec_1 = Ivectors[i]
            Ivec_2 = Ivectors[j]

            H_dd = dipole_dipole(nspin[(i, j),], Ivec_1, Ivec_2, ntype)

            H += H_dd

    return H, dimensions


def eta_hamiltonian(nspin, ntype, I, S, eta):
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
