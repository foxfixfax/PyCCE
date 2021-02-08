import numpy as np

from .sm import _smc
from .units import HBAR, ELECTRON_GYRO


# HBAR = 1.05457172  # When everything else in rad, kHz, ms, G, A


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
    dbefore = np.prod(dim[:i])
    dafter = np.prod(dim[i + 1:])

    M_expanded = np.kron(np.kron(np.eye(dbefore, dtype=np.complex128), M),
                         np.eye(dafter, dtype=np.complex128))

    return M_expanded


def generate_dimensions(nspin, central_spin=None):
    ntype = nspin.types
    dimensions = [_smc[ntype[n['N']].s].dim for n in nspin]
    if central_spin is not None:
        dimensions += [_smc[central_spin].dim]
    dimensions = np.asarray(dimensions, dtype=np.int32)
    return dimensions


def generate_projections(state):
    spin = (state.size - 1) / 2
    sm = _smc[spin]

    projections = np.array([state.conj() @ sm.x @ state,
                            state.conj() @ sm.y @ state,
                            state.conj() @ sm.z @ state],
                           dtype=np.complex128)
    return projections


def zeeman(gyro, s, B):
    """
    Zeeman interactions of the n spin
    @param gyro: float
        gyromagnetic ratio of the n spin
    @param spin_matrix: SpinMatrix
        spin matrix object for n spin
    @param B: array_like
        magnetic field as (Bx, By, Bz)
    @return: ndarray of shape (2s+1, 2s+1)
    """
    spin_matrix = _smc[s]
    BI = B[0] * spin_matrix.x + B[1] * spin_matrix.y + B[2] * spin_matrix.z

    H_zeeman = - gyro * BI

    return H_zeeman


def quadrupole(quadrupole_tensor, s):
    """
    Quadrupole interaction of the n spin (for s>1)
    @param quadrupole_tensor: np.array of shape (3,3)
        quadrupole interaction of n spin
    @param s: float
        total spin of n spin
    @param spin_matrix: SpinMatrix
        spin matrix object for n spin
    @return: ndarray of shape (2s+1, 2s+1)
    """
    spin_matrix = _smc[s]
    iv = np.asarray([spin_matrix.x, spin_matrix.y, spin_matrix.z])
    # v_ivec = np.einsum('lp,lij->ijp', quadrupole_tensor, iv);
    # iqi = np.einsum('ijp,pjk->ik', v_ivec, iv)
    v_ivec = np.einsum('ij,jkl->ikl', quadrupole_tensor, iv, dtype=np.complex128)
    iqi = np.einsum('lij,ljk->ik', iv, v_ivec, dtype=np.complex128)
    # iqi = np.einsum('lij,lp,pjk->ik', iv, quadrupole_tensor, iv, dtype=np.complex128)

    # delI2 = np.sum(np.diag(quadrupole_tensor)) * np.eye(spin_matrix.x.shape[0]) * s * (s + 1)

    H_quad = iqi  # - delI2 / 3

    return H_quad


def dipole_dipole(coord_1, coord_2, g1, g2, ivec_1, ivec_2):
    """
    Compute dipole_dipole interactions between two bath spins
    @param coord_1: ndarray with shape (3,)
        coordinates of the first spin
    @param coord_2: ndarray with shape (3,)
        coordinates of the second spin
    @param g1: float
        gyromagnetic ratio of the first spin
    @param g2: float
        gyromagnetic ratio of the second spin
    @param ivec_1: ndarray with shape (3, d, d)
        where d is total size of the Hilbert space. Vector of [Ix, Iy, Iz] for the first bath spin
    @param ivec_2: ndarray with shape (3, d, d)
        where d is total size of the Hilbert space. Vector of [Ix, Iy, Iz] for the second bath spin
    @return: ndarray of shape (d, d)
    dipole dipole interactions of two bath spins
    """
    pre = g1 * g2 * HBAR

    pos = coord_1 - coord_2
    r = np.linalg.norm(pos)

    p_tensor = -pre * (3 * np.outer(pos, pos) -
                       np.eye(3, dtype=np.complex128) * r ** 2) / (r ** 5)

    p_ivec = np.einsum('ij,jkl->ikl', p_tensor, ivec_2,
                       dtype=np.complex128)  # p_ivec = Ptensor @ Ivector
    H_DD = np.einsum('lij,ljk->ik', ivec_1, p_ivec, dtype=np.complex128)

    # DD = IPI = IxPxxIx + IxPxyIy + ..
    # H_DD = np.einsum('lij,lp,pjk->ik', ivec_1, p_tensor, ivec_2, dtype=np.complex128)

    return H_DD


def projected_hyperfine(hyperfine_tensor, s, projections):
    """
    Compute projected hyperfine Hamiltonian for one state of the central spin
    @param hyperfine_tensor: np.array of shape (3,3)
        hyperfine interactions of n spin
    @param spin_matrix: SpinMatrix
        spin matrix object for n spin
    @param projections: np.ndarray of shape (3,)
        projections of the central spin qubit levels [<Sx>, <Sy>, <Sz>]
    @return: ndarray of shape (d, d)
    """
    spin_matrix = _smc[s]

    A_projected = projections @ hyperfine_tensor  # spin_matrix.get_projection(state)
    H_Hyperfine = (A_projected[0] * spin_matrix.x +
                   A_projected[1] * spin_matrix.y +
                   A_projected[2] * spin_matrix.z)

    return H_Hyperfine


def projected_hamiltonian(nspin, projections_alpha, projections_beta, B):
    """
    Compute projected hamiltonian on state and beta qubit states
    @param nspin: BathArray with shape (n,)
        ndarray of bath spins
    @param projections_alpha: np.ndarray with shape (3,)
        projections of the central spin state level [<Sx>, <Sy>, <Sz>]
    @param projections_beta: np.ndarray with shape (3,)
        projections of the central spin beta level [<Sx>, <Sy>, <Sz>]
    @param B: ndarray with shape (3,)
        magnetic field of format (Bx, By, Bz)
    @return: H_alpha, H_beta
    """
    ntype = nspin.types
    dimensions = generate_dimensions(nspin, central_spin=None)
    nnuclei = nspin.shape[0]

    tdim = np.prod(dimensions, dtype=np.int32)

    H_alpha = np.zeros((tdim, tdim), dtype=np.complex128)
    H_beta = np.zeros((tdim, tdim), dtype=np.complex128)

    ivectors = []

    for j, n in enumerate(nspin):
        s = ntype[n].s

        if s > 1 / 2:
            H_quad = quadrupole(n['Q'], s)
            H_single = zeeman(ntype[n].gyro, s, B) + H_quad
        else:
            H_single = zeeman(ntype[n].gyro, s, B)

        H_HF_alpha = projected_hyperfine(n['A'], s, projections_alpha)
        H_HF_beta = projected_hyperfine(n['A'], s, projections_beta)

        H_j_alpha = H_single + H_HF_alpha
        H_j_beta = H_single + H_HF_beta

        H_alpha += expand(H_j_alpha, j, dimensions)
        H_beta += expand(H_j_beta, j, dimensions)

        ivec = np.array([expand(_smc[s].x, j, dimensions),
                         expand(_smc[s].y, j, dimensions),
                         expand(_smc[s].z, j, dimensions)],
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

    return H_alpha, H_beta, dimensions


def hyperfine(hyperfine_tensor, svec, ivec):
    """
    Compute hyperfine interactions between central spin spin_matrix and bath spin I
    Compute projected hyperfine Hamiltonian for one state of the central spin
    @param hyperfine_tensor: np.array of shape (3,3)
        hyperfine interactions of n spin
    @param svec: ndarray with shape (3, d, d)
        d is the total size of the Hilbert space. [Sx, Sy, Sz] array of the central spin
    @param ivec: ndarray with shape (3, d, d)
        d is the total size of the Hilbert space. [Ix, Iy, Iz] array of the bath spin
    @return: ndarray
    """
    aivec = np.einsum('ij,jkl->ikl', hyperfine_tensor, ivec)  # AIvec = Atensor @ Ivector
    # HF = SPI = SxPxxIx + SxPxyIy + ..
    H_HF = np.einsum('lij,ljk->ik', svec, aivec)
    # H_HF = np.einsum('lij,lp,pjk->ik', svec, hyperfine_tensor, ivec, dtype=np.complex128)
    return H_HF


def self_electron(B, s, D=0, E=0, gyro=ELECTRON_GYRO):
    """
    central spin Hamiltonian
    @param B: ndarray with shape (3,)
        magnetic field of format (Bx, By, Bz)
    @param spin_matrix: SpinMatrix
        SpinMatrix of the central spin
    @param gyro: float
        gyromagnetic ratio (in rad/(msec*Gauss)) of the central spin
    @param D: float or ndarray with shape (3,3)
        D parameter in central spin ZFS OR total ZFS tensor
    @param E: float
        E parameter in central spin ZFS
    @return: ndarray
    """
    spin_matrix = _smc[s]
    if isinstance(D, (np.floating, float, int)):
        H0 = D * (spin_matrix.z @ spin_matrix.z - 1 / 3 * spin_matrix.s * (spin_matrix.s + 1) * spin_matrix.eye) + \
             E * (spin_matrix.x @ spin_matrix.x - spin_matrix.y @ spin_matrix.y)
    else:
        svec = np.asarray([spin_matrix.x, spin_matrix.y, spin_matrix.z], dtype=np.complex128)
        dsvec = np.einsum('ij,jkl->ikl', D, svec,
                          dtype=np.complex128)  # AIvec = Atensor @ Ivector
        # H0 = SDS = SxDxxSx + SxDxySy + ..
        H0 = np.einsum('lij,ljk->ik', svec, dsvec, dtype=np.complex128)
        # H0 = np.einsum('lij,lp,pjk->ik', svec, D, svec, dtype=np.complex128)

    H1 = -gyro * (B[0] * spin_matrix.x + B[1] * spin_matrix.y + B[2] * spin_matrix.z)

    return H1 + H0


def total_hamiltonian(nspin, central_spin, B, D=0, E=0, central_gyro=ELECTRON_GYRO):
    """
    Total hamiltonian for cluster including central spin
    @param nspin: ndarray with shape (n,)
        ndarray of bath spins in the given cluster with size n
    @param central_spin: float
        total spin of the central spin
    @param B: ndarray with shape (3,)
        magnetic field of format (Bx, By, Bz)
    @param D: float or ndarray with shape (3,3)
        D parameter in central spin ZFS OR total ZFS tensor
    @param E: float
        E parameter in central spin ZFS
    @param central_gyro: float
        gyromagnetic ratio (in rad/(msec*Gauss)) of the central spin
    @return: H, dimensions
        H: ndarray with shape (prod(dimensions), prod(dimensions))
        dimensions: list of dimensions for each spin, last entry - dimensions of central spin
    """

    central_spin_matrix = _smc[central_spin]

    ntype = nspin.types
    dimensions = generate_dimensions(nspin, central_spin=central_spin)

    nnuclei = nspin.shape[0]

    tdim = np.prod(dimensions, dtype=np.int32)
    H = np.zeros((tdim, tdim), dtype=np.complex128)

    H_electron = self_electron(B, central_spin, D, E, central_gyro)
    svec = np.array([expand(central_spin_matrix.x, nnuclei, dimensions),
                     expand(central_spin_matrix.y, nnuclei, dimensions),
                     expand(central_spin_matrix.z, nnuclei, dimensions)],
                    dtype=np.complex128)

    H += expand(H_electron, nnuclei, dimensions)

    ivectors = []

    for j, n in enumerate(nspin):
        s = ntype[n].s
        ivec = np.array([expand(_smc[s].x, j, dimensions),
                         expand(_smc[s].y, j, dimensions),
                         expand(_smc[s].z, j, dimensions)],
                        dtype=np.complex128)
        if s > 1 / 2:
            H_quad = quadrupole(n['Q'], s)
            H_single = zeeman(ntype[n].gyro, s, B) + H_quad
        else:
            H_single = zeeman(ntype[n].gyro, s, B)

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


def mf_electron(s, others, others_state):
    """
    compute mean field effect for all bath spins not included in the cluster
    on central spin
    @param spin_matrix: SpinMatrix
        SpinMatrix of the central spin
    @param others: ndarray of shape (n_bath - n_cluser,)
        ndarray of all bath spins not included in the cluster
    @param others_state: ndarray of shape (n_bath - n_cluser,)
        Sz projections of the state of all others nuclear spins not included in the given cluster
    @return: ndarray
    """
    spin_matrix = _smc[s]
    # xfield = np.sum(others['A'][:, 2, 0] * others_state)
    # yfield = np.sum(others['A'][:, 2, 1] * others_state)
    zfield = np.sum(others['A'][:, 2, 2] * others_state)

    H_mf = zfield * spin_matrix.z  # + xfield * spin_matrix.x + yfield * spin_matrix.y

    return H_mf


def mf_nucleus(n, g, gyros, s, others, others_state):
    """
    compute mean field effect on the bath spin n from all other bath spins
    @param n: np.void object with dtype _dtype_bath
        single bath spin n
    @param g: float
        gyromagnetic ratio (in rad/(msec*Gauss)) of the nuclear spin n
    @param gyros: ndarray of shape (n_bath - n_cluster,)
        ndarray of gyromagnetic ratios for all bath spins not included in the cluster
    @param others: ndarray of shape (n_bath - n_cluser,)
        BathArray of all bath spins not included in the cluster
    @param others_state: ndarray of shape (n_bath - n_cluser,)
        Sz projections of the state of all others nuclear spins not included in the given cluster
    @return: ndarray
    """
    spin_matrix = _smc[s]

    pre = g * gyros * HBAR

    pos = n['xyz'] - others['xyz']
    r = np.linalg.norm(pos, axis=1)
    cos_theta = pos[:, 2] / r

    zfield = np.sum(pre / r ** 3 * (1 - 3 * cos_theta ** 2) * others_state)
    H_mf = zfield * spin_matrix.z

    return H_mf


def mf_hamiltonian(nspin, B, central_spin, others, others_state, D=0, E=0, central_gyro=ELECTRON_GYRO):
    """
    compute total Hamiltonian for the given cluster including mean field effect of all nuclei
    outside of the given cluster
    @param nspin: ndarray with shape (n,)
        ndarray of bath spins in the given cluster with size n
    @param B: ndarray with shape (3,)
        magnetic field of format (Bx, By, Bz)
    @param central_spin: float
        total spin of the central spin
    @param others: ndarray of shape (n_bath - n_cluser,)
        ndarray of all bath spins not included in the cluster
    @param others_state: ndarray of shape (n_bath - n_cluser,)
        Sz projections of the state of all others nuclear spins not included in the given cluster
    @param D: float or ndarray with shape (3,3)
        D parameter in central spin ZFS OR total ZFS tensor
    @param E: float
        E parameter in central spin ZFS
    @param central_gyro: float
        gyromagnetic ratio (in rad/(msec*Gauss)) of the central spin    @return:
    @return: H, dimensions
        H: ndarray with shape (prod(dimensions), prod(dimensions)) hamiltonian
        dimensions: list of dimensions for each spin, last entry - dimensions of central spin
    """
    ntype = nspin.types
    central_spin_matrix = _smc[central_spin]
    dimensions = generate_dimensions(nspin, central_spin=central_spin)

    nnuclei = nspin.shape[0]

    tdim = np.prod(dimensions, dtype=np.int32)
    H = np.zeros((tdim, tdim), dtype=np.complex128)

    H_electron = self_electron(B, central_spin, D, E, central_gyro) + mf_electron(central_spin, others,
                                                                                  others_state)

    svec = np.array([expand(central_spin_matrix.x, nnuclei, dimensions),
                     expand(central_spin_matrix.y, nnuclei, dimensions),
                     expand(central_spin_matrix.z, nnuclei, dimensions)],
                    dtype=np.complex128)

    H += expand(H_electron, nnuclei, dimensions)

    ivectors = []

    for j, n in enumerate(nspin):
        s = ntype[n].s
        ivec = np.array([expand(_smc[s].x, j, dimensions),
                         expand(_smc[s].y, j, dimensions),
                         expand(_smc[s].z, j, dimensions)],
                        dtype=np.complex128)

        H_mf_nucleus = mf_nucleus(n, ntype[n].gyro, ntype[others].gyro,
                                  s, others, others_state)

        H_zeeman = zeeman(ntype[n].gyro, s, B)

        if s > 1 / 2:
            H_quad = quadrupole(n['Q'], s)
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


def eta_hamiltonian(nspin, central_spin, alpha, beta, eta):
    """
    EXPERIMENTAL. Compute hamiltonian with eta-term - gradually turn off or turn on the secular interactions for
    state and beta qubit states
    @param nspin: ndarray with shape (n,)
        ndarray of bath spins in the given cluster with size n
    @param central_spin: float
        total spin of the central spin
    @param alpha: np.ndarray with shape (2s+1,)
        state state of the qubit
    @param beta: np.ndarray with shape (2s+1,)
        beta state of the qubit
    @param eta: value of dimensionless parameter eta (from 0 to 1)
    @return:
    """
    ntype = nspin.types
    nnuclei = nspin.shape[0]
    dimensions = generate_dimensions(nspin, central_spin=central_spin)

    AIzi = 0
    for j in range(nnuclei):
        s = ntype[nspin[j]['N']].s
        ivec = np.array([expand(_smc[s].x, j, dimensions),
                         expand(_smc[s].y, j, dimensions),
                         expand(_smc[s].z, j, dimensions)],
                        dtype=np.complex128)

        AIzi += np.einsum('j,jkl->kl', nspin[j]['A'][2, :], ivec, dtype=np.complex128)

    up_down = (1 - eta) / 2 * (np.tensordot(alpha, alpha, axes=0) + np.tensordot(beta, beta, axes=0))
    H_eta = expand(up_down, nnuclei, dimensions) @ AIzi
    return H_eta
