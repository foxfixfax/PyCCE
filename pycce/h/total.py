from .base import Hamiltonian
from .functions import *


def projected_hamiltonian(bath, center, mfield,
                          others=None):
    r"""
    Compute projected hamiltonian on state and beta qubit states. Wrapped function so the actual call does not
    follow the one above!

    Args:
        bath (BathArray):
            array of all bath spins in the cluster.

        mfield (ndarray with shape (3,)):
            Magnetic field of type ``mfield = np.array([Bx, By, Bz])``.

        others (BathArray with shape (m,)):
            array of all bath spins outside the cluster

        other_states (ndarray with shape (m,) or (m, 3)):
            Array of Iz projections for each bath spin outside of the given cluster.

    Returns:
        tuple: *tuple* containing:

            * **Hamiltonian**: Hamiltonian of the given cluster, conditioned on the alpha qubit state.
            * **Hamiltonian**: Hamiltonian of the given cluster, conditioned on the beta qubit state.
    """
    dims, vectors = dimensions_spinvectors(bath, central_spin=None)
    clusterint = bath_interactions(bath, vectors)

    halpha = 0 + clusterint
    hbeta = 0 + clusterint
    ncenters = len(center)
    for ivec, n in zip(vectors, bath):
        hsingle = expanded_single(ivec, n.gyro, mfield, n['Q'], n.detuning)

        if others is not None:
            hsingle += overhauser_bath(ivec, n['xyz'], n.gyro, others.gyro,
                                       others['xyz'], others.proj)
        hf_alpha = 0
        hf_beta = 0
        for i, c in enumerate(center):
            if ncenters > 1:
                hf = n['A'][i]
            else:
                hf = n['A']
            hf_alpha += conditional_hyperfine(hf, ivec, c.projections_alpha)
            hf_beta += conditional_hyperfine(hf, ivec, c.projections_beta)

        halpha += hsingle + hf_alpha
        hbeta += hsingle + hf_beta

    if center.energy_alpha is not None:
        for i, c in enumerate(center):
            if ncenters > 1:
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


def bath_hamiltonian(bath, mfield):
    r"""
    Compute projected hamiltonian on state and beta qubit states. Wrapped function so the actual call does not
    follow the one above!

    Args:
        bath (BathArray):
            array of all bath spins in the cluster.

        mfield (ndarray with shape (3,)):
            Magnetic field of type ``mfield = np.array([Bx, By, Bz])``.

        others (BathArray with shape (m,)):
            array of all bath spins outside the cluster

        other_states (ndarray with shape (m,) or (m, 3)):
            Array of Iz projections for each bath spin outside of the given cluster.

    Returns:
        Hamiltonian: Hamiltonian of the given cluster without qubit.
    """
    dims, vectors = dimensions_spinvectors(bath, central_spin=None)
    clusterint = bath_interactions(bath, vectors)

    for ivec, n in zip(vectors, bath):
        clusterint += expanded_single(ivec, n.gyro, mfield, n.Q, n.detuning)

    return Hamiltonian(dims, vectors, data=clusterint)


# TODO FINISH CONVERSION OF MEANFIELD OUTSIDE
def total_hamiltonian(bath, center, mfield):
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

    Returns:
        Hamiltonian: hamiltonian of the given cluster, including central spin.

    """

    dims, vectors = dimensions_spinvectors(bath, central_spin=center)

    totalh = bath_interactions(bath, vectors)
    ncenters = len(center)

    for i, c in enumerate(center):
        totalh += self_central(vectors[bath.size + i], mfield, c.zfs, c.gyro, c.detuning)

    totalh += center_interactions(center, vectors[bath.size:])

    for ivec, n in zip(vectors, bath):
        hsingle = expanded_single(ivec, n.gyro, mfield, n.Q, n.detuning)

        hhyperfine = 0

        for i in range(ncenters):
            if ncenters == 1:
                hf = n.A
            else:
                hf = n.A[i]

            hhyperfine += hyperfine(hf, vectors[bath.size + i], ivec)

        totalh += hsingle + hhyperfine

    return Hamiltonian(dims, vectors, data=totalh)


def central_hamiltonian(center, magnetic_field, hyperfine=None, bath_state=None):
    dims, vectors = dimensions_spinvectors(central_spin=center)
    try:
        ncenters = len(center)
        single_center = False
    except TypeError:
        single_center = True
        ncenters = None

    if single_center:
        totalh = self_central(vectors[0], magnetic_field,
                              center.zfs, center.gyro, center.detuning)
        if hyperfine is not None and bath_state is not None:
            totalh += overhauser_central(vectors[0], hyperfine, bath_state)
        return totalh

    totalh = 0

    for i, c in enumerate(center):
        totalh += self_central(vectors[i], magnetic_field, c.zfs, c.gyro)

        if hyperfine is not None and bath_state is not None:
            if ncenters == 1:
                hf = hyperfine
            else:
                hf = hyperfine[..., i, :, :]

            totalh += overhauser_central(vectors[i], hf, bath_state)

    totalh += center_interactions(center, vectors)

    return Hamiltonian(dims, vectors, data=totalh)


def projected_addition(vectors, bath, center, state):
    ncenters = len(center)
    addition = 0

    for ivec, n in zip(vectors, bath):
        try:

            iterator = iter(state)  # iterable
            state = state[0]

        except TypeError:
            for i, c in enumerate(center):
                if ncenters > 1:
                    hf = n.A[i]
                else:
                    hf = n.A
                projections = c.get_projections(state)
                addition += conditional_hyperfine(hf, ivec, projections)
        else:

            for i, (c, s) in enumerate(zip(center, iterator)):
                if ncenters > 1:
                    hf = n.A[i]
                else:
                    hf = n.A

                projections = c.get_projections(state)
                addition += conditional_hyperfine(hf, ivec, projections)

    energy = center.get_energy(state)

    if energy is not None:
        for i, c in enumerate(center):
            if ncenters > 1:
                hf = bath.A[:, i]
            else:
                hf = bath.A

            projections_state_all = c.get_projections_all(state)

            addition += bath_mediated(hf, vectors, energy,
                                      center.energies, projections_state_all)

    return addition


def center_zo_addition(vectors, cluster, outer_spin, outer_state):
    addition = 0
    ncenters = vectors.size - cluster.size

    for i, v in enumerate(vectors[cluster.size:]):
        if ncenters == 1:
            hf = outer_spin.A
        else:
            hf = outer_spin.A[:, i]

        addition += overhauser_central(v, hf, outer_state)

    return addition


def bath_pd_zo_addition(vectors, cluster, outer_spin, outer_state):
    addition = 0
    for ivec, n in zip(vectors, cluster):
        addition += overhauser_bath(ivec, n.xyz, n.gyro, outer_spin.gyro,
                                    outer_spin.xyz, outer_state)
    return addition


def zero_order_addition(vectors, cluster, outer_spin, outer_state):
    addition = center_zo_addition(vectors, cluster, outer_spin, outer_state) + bath_pd_zo_addition(vectors, cluster,
                                                                                                   outer_spin,
                                                                                                   outer_state)

    return addition


def zero_order_imap(vectors, indexes, bath, projected_state):
    outer_mask = np.ones(bath.size, dtype=bool)
    outer_mask[indexes] = False

    outer_spin = bath[outer_mask]
    outer_state = projected_state[outer_mask]
    cluster = bath[indexes]

    addition = center_zo_addition(vectors, cluster, outer_spin, outer_state)

    if bath.imap is None:
        addition += bath_pd_zo_addition(vectors, cluster, outer_spin, outer_state)

        return addition

    imap_indexes = bath.imap.indexes
    remove_j = np.ones(indexes.shape, dtype=bool)

    for j, (ind, ivec) in enumerate(zip(indexes, vectors)):

        outer_mask[:] = True
        outer_mask[indexes] = False

        where_index = (imap_indexes == ind)

        if where_index.any():
            remove_j[j] = False
            which_pairs = where_index.any(axis=1) & (~np.isin(imap_indexes, indexes[remove_j]).any(axis=1))
            remove_j[j] = True

            other_indexes = imap_indexes[which_pairs][~(where_index[which_pairs])]

            addition += overhauser_from_tensors(ivec, bath.imap.data[which_pairs], projected_state[other_indexes])


            outer_mask[other_indexes] = False

        if outer_mask.any():
            outer_spin = bath[outer_mask]
            outer_state = projected_state[outer_mask]
            n = bath[ind]
            addition += overhauser_bath(ivec, n.xyz, n.gyro, outer_spin.gyro,
                                        outer_spin.xyz, outer_state)

    return addition
