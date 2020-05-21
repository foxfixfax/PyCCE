import numpy as np

hbar = 1.05457172  # When everything else in rad, kHz, ms, G, A


class SpinMatrix:
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

    def __getitem__(self, state):
        if np.all(state == self.alpha):
            return self.projections_alpha

        elif np.all(state == self.beta):
            return self.projections_beta
        else:
            raise KeyError('There is no such qubit state!')

    def __setitem__(self, state, value):
        if np.all(state == self.alpha):
            self.projections_alpha = value

        elif np.all(state == self.beta):
            self.projections_beta = value
        else:
            raise KeyError('There is no such qubit state!')


def generate_SpinMatricies(ntype):
    nmatrices = {}

    for N in ntype:
        nmatrices[ntype[N].s] = SpinMatrix(ntype[N].s)

    return nmatrices


def expand(M, i, dim):
    dbefore = np.prod(dim[:i], dtype=np.int32)
    dafter = np.prod(dim[i + 1:], dtype=np.int32)

    M_expanded = np.kron(np.kron(np.eye(dbefore, dtype=np.complex128), M),
                         np.eye(dafter, dtype=np.complex128))

    return M_expanded


def zeeman(n, ntype, I, B):
    s = ntype[n['N']].s

    BI = B[0] * I[s].x + B[1] * I[s].y + B[2] * I[s].z

    H_zeeman = - ntype[n['N']].gyro * BI

    return H_zeeman


def dipole_dipole(nuclei, Ivec_1, Ivec_2, ntype):
    n1 = nuclei[0]
    n2 = nuclei[1]

    g1 = ntype[n1['N']].gyro
    g2 = ntype[n2['N']].gyro

    pre = g1 * g2 * hbar

    pos = n1['xyz'] - n2['xyz']
    r = np.linalg.norm(pos)

    PTensor = -pre * (3 * np.outer(pos, pos) -
                      np.eye(3, dtype=np.complex128) * r ** 2) / (r ** 5)

    PIvec = np.einsum('ij,jkl->ikl', PTensor, Ivec_2,
                      dtype=np.complex128)  # PIvec = Ptensor @ Ivector

    # DD = IPI = IxPxxIx + IxPxyIy + ..

    H_DD = np.einsum('lij,ljk->ik', Ivec_1, PIvec, dtype=np.complex128)

    return H_DD


def projected_hyperfine(n, state, ntype, I, S):
    s = ntype[n['N']].s

    A_projected = S[state] @ n['A']
    H_Hyperfine = (A_projected[0] * I[s].x +
                   A_projected[1] * I[s].y +
                   A_projected[2] * I[s].z)

    return H_Hyperfine


def total_hamiltonian(nspin, ntype, I, B, S):
    dimensions = [I[ntype[n['N']].s].dim for n in nspin]
    nnuclei = nspin.shape[0]

    tdim = np.prod(dimensions, dtype=np.int32)

    H_alpha = np.zeros((tdim, tdim), dtype=np.complex128)
    H_beta = np.zeros((tdim, tdim), dtype=np.complex128)

    Ivectors = []

    for j in range(nnuclei):
        H_zeeman = zeeman(nspin[j], ntype, I, B)
        H_HF_alpha = projected_hyperfine(nspin[j], S.alpha, ntype, I, S)
        H_HF_beta = projected_hyperfine(nspin[j], S.beta, ntype, I, S)

        H_j_alpha = H_zeeman + H_HF_alpha
        H_j_beta = H_zeeman + H_HF_beta

        H_alpha += expand(H_j_alpha, j, dimensions)
        H_beta += expand(H_j_beta, j, dimensions)

        s = ntype[nspin[j]['N']].s
        Ivec = np.array([expand(I[s].x, j, dimensions),
                         expand(I[s].y, j, dimensions),
                         expand(I[s].z, j, dimensions)],
                        dtype=np.complex128)

        Ivectors.append(Ivec)

    for i in range(nnuclei):
        for j in range(i + 1, nnuclei):

            Ivec_1 = Ivectors[i]
            Ivec_2 = Ivectors[j]

            H_DD = dipole_dipole(nspin[(i, j), ], Ivec_1, Ivec_2, ntype)

            H_alpha += H_DD
            H_beta += H_DD

    return H_alpha, H_beta


def hyperfine(n, Svec, Ivec):
    ATensor = n['A']
    AIvec = np.einsum('ij,jkl->ikl', ATensor, Ivec,
                      dtype=np.complex128)  # AIvec = Atensor @ Ivector
    # HF = SPI = SxPxxIx + SxPxyIy + ..
    H_HF = np.einsum('lij,ljk->ik', Svec, AIvec, dtype=np.complex128)

    return H_HF


def self_electron(B, S, gyro_e, D, E):
    """compute spin hamiltonian matrix in Sz basis
    gyro_e in rad/(ms*G)
    B in Gauss
    D, E in rad * kHz"""

    H0 = D * (S.z @ S.z - 1 / 3 * S.s * (S.s + 1) * S.eye) + \
        E * (S.x @ S.x - S.y @ S.y)
    H1 = -gyro_e * (B[0] * S.x + B[1] * S.y + B[2] * S.z)

    return H1 + H0


def total_elhamiltonian(nspin, ntype, I, B, S, gyro_e, D, E):
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

        H_zeeman = zeeman(nspin[j], ntype, I, B)
        H_HF = hyperfine(nspin[j], Svec, Ivec)

        H += expand(H_zeeman, j, dimensions) + H_HF

        Ivectors.append(Ivec)

    for i in range(nnuclei):
        for j in range(i + 1, nnuclei):

            Ivec_1 = Ivectors[i]
            Ivec_2 = Ivectors[j]

            H_dd = dipole_dipole(nspin[(i, j), ], Ivec_1, Ivec_2, ntype)

            H += H_dd

    return H, dimensions


def mf_electron(S, others, others_state):
    H_mf = 0
    zfield = np.sum(others['A'][:, 2, 2] * others_state)
    H_mf += zfield * S.z

    return H_mf


def mf_nucleus(nspin, ntype, I, others, others_state):
    g = ntype[nspin['N']].gyro
    s = ntype[nspin['N']].s

    gyros = np.empty(others.shape, dtype=np.float64)
    for n in ntype:
        other_g = ntype[n].gyro
        mask = others['N'] == n
        gyros[mask] = other_g

    pre = g * gyros * hbar

    pos = nspin['xyz'] - others['xyz']
    r = np.linalg.norm(pos, axis=1)
    cos_theta = pos[:, 2] / r

    zfield = np.sum(pre / r ** 3 * (1 - cos_theta ** 2) * others_state)
    H_mf = zfield * I[s].z

    return H_mf


def mf_hamiltonian(nspin, ntype, I, B, S, gyro_e, D, E, allspins, bath_state=None):
    if bath_state is None:
        rgen = np.random.default_rng()
        bath_state = np.empty(allspins.shape, dtype=np.float64)

        for n in ntype:
            s = ntype[n].s
            snumber = int(round(2*s + 1))
            mask = allspins['N'] == n
            bath_state[mask] = rgen.integers(snumber, size=np.count_nonzero(mask)) - s

    others_mask = np.isin(allspins, nspin)
    others = allspins[~others_mask]
    others_state = bath_state[~others_mask]

    dimensions = [I[ntype[n['N']].s].dim for n in nspin] + [S.dim]
    nnuclei = nspin.shape[0]

    tdim = np.prod(dimensions, dtype=np.int32)
    H = np.zeros((tdim, tdim), dtype=np.complex128)

    H_electron = self_electron(B, S, gyro_e, D, E)
    H_mf_electron = mf_electron(S, others, others_state)
    Svec = np.array([expand(S.x, nnuclei, dimensions),
                     expand(S.y, nnuclei, dimensions),
                     expand(S.z, nnuclei, dimensions)],
                    dtype=np.complex128)

    H += expand(H_electron, nnuclei, dimensions) + expand(H_mf_electron, nnuclei, dimensions)
    Ivectors = []

    for j in range(nnuclei):
        s = ntype[nspin[j]['N']].s
        Ivec = np.array([expand(I[s].x, j, dimensions),
                         expand(I[s].y, j, dimensions),
                         expand(I[s].z, j, dimensions)],
                        dtype=np.complex128)

        H_zeeman = zeeman(nspin[j], ntype, I, B)
        H_HF = hyperfine(nspin[j], Svec, Ivec)
        H_mf_nucleus = mf_hamiltonian(nspin[j], ntype, ntype, I, others, others_state)

        H += expand(H_zeeman, j, dimensions) + expand(H_mf_nucleus, j, dimensions) + H_HF

        Ivectors.append(Ivec)

    for i in range(nnuclei):
        for j in range(i + 1, nnuclei):

            Ivec_1 = Ivectors[i]
            Ivec_2 = Ivectors[j]

            H_dd = dipole_dipole(nspin[(i, j), ], Ivec_1, Ivec_2, ntype)

            H += H_dd

    return H, dimensions

