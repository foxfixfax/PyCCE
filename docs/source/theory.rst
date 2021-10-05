Theoretical Background
===========================

This document contains a brief list of the coupling parameters between
the central and the bath spins used in **PyCCE**, a description of the qubit dephasing, and a
summary of the cluster correlation expansion (CCE) method.
You can find more details in the following references [#code]_ [#yang2008]_ [#onizhuk2021]_.

Hamiltonian
----------------------------

The **PyCCE** package allows one to simulate the dynamics of a central spin interacting with a spin bath through
the following Hamiltonian:

.. math::
    \hat H = \hat H_S + \hat H_{SB} + \hat H_{B}

Where :math:`\hat H_S` is the Hamiltonian of the free central spin,
:math:`\hat H_{SB}` denotes interactions between central spin and a spin belonging to the bath,
and :math:`\hat H_B` are intrinsic bath spin interactions:

.. math::

        &\hat H_S = \mathbf{SDS} + \mathbf{B\gamma}_{S}\mathbf{S} \\
        &\hat H_{SB} = \sum_i \mathbf{S}\mathbf{A}_i\mathbf{I}_i \\
        &\hat H_{B} = \sum_i{\mathbf{I}_i\mathbf{P}_i \mathbf{I}_i +
                      \mathbf{B}\mathbf{\gamma}_i\mathbf{I}_i} +
                      \sum_{i>j} \mathbf{I}_i\mathbf{J}_{ij}\mathbf{I}_j

Where :math:`\mathbf{S}=(\hat{S}_x, \hat{S}_y, \hat{S}_z)` are the components of spin operators of the central spin,
:math:`\mathbf{I}=(\hat{I}_x, \hat{I}_y, \hat{I}_z)`  are the components of the bath spin operators,
and :math:`\mathbf{B}=(B_x,B_y,B_z)` is an external applied magnetic field.

The interactions are described by the following tensors
that are either required to be input by user or can be generated
by the package itself (see :doc:`parameters` for details):

- :math:`\mathbf{D}` (:math:`\mathbf{P}`)  is the self-interaction tensor of the central spin (bath spin).
  For the electron spin, the tensor corresponds to the zero-field splitting (ZFS) tensor.
  For nuclear spins corresponds to the quadrupole interactions tensor.
- :math:`\mathbf{\gamma}_i`$`is the magnetic field interaction tensor of the
  :math:`i`-spin describing the interaction of the spin and the external magnetic field :math:`B`.
  We assume that for the bath spins, it is isotropic.
- :math:`\mathbf{A}` is the interaction tensor between central and bath spins.
  In the case of the nuclear spin bath, it corresponds to the hyperfine couplings.
- :math:`\mathbf{J}` is the interaction tensor between bath spins.


Qubit dephasing
---------------------------------

Usually, two coherence times are measured to characterize the loss of a qubit coherence - :math:`T_1` and :math:`T_2`.
:math:`T_1` defines the timescale over which the qubit population is thermalized;
:math:`T_2` describes a purely quantum phenomenon - the loss of the phase of the qubit's superposition state.

In the pure dephasing regime (:math:`T_1 >> T_2`) the decoherence of the central spin is completely determined
by the decay of the off diagonal element of the density matrix of the qubit.

Namely, if the qubit is initially prepared in the
:math:`\left|{\psi}\right\rangle = \frac{1}{\sqrt{2}}(\left|{0}\right\rangle+e^{i\phi}\left|{1}\right\rangle)` state,
the loss of the relative phase of the :math:`\left|{0}\right\rangle` and :math:`\left|{1}\right\rangle`
levels is characterized by the coherence function:

.. math::

    \mathcal{L}(t) = \frac{\left\langle{1}\right|\hat{\rho}_S(t)\left|{0}\right\rangle}
    {\left\langle{1}\right|\hat{\rho}_S(0)\left|{0}\right\rangle} =
    \frac{\langle{\hat \sigma_{-}(t)}\rangle}{\langle{\hat \sigma_{-}(0)}\rangle}

Where :math:`\hat{\rho}_S(t)` is the density matrix of the central spin and
:math:`\left|{0}\right\rangle` and :math:`\left|{1}\right\rangle` are qubit levels.

The cluster correlation expansion (CCE) method was first introduced in ref. [#yang2008]_.
The core idea of the CCE approach is that the spin bath-induced decoherence
can be factorized into set of irreducible contributions from the bath spin clusters.
Written in terms of the coherence function:

.. math::
    \mathcal{L}(t) = \prod_{C} \tilde{L}_C = \prod_{i}\tilde{L}_{\{i\}}\prod_{i,j}\tilde{L}_{\{ij\}}...

Where each cluster contribution is defined recursively as:

.. math::
    \tilde{L}_C = \frac{L_{C}}{\prod_{C'}\tilde{L}_{C'\subset C}}

Where :math:`L_{C}` is a coherence function of the qubit,
interacting only with the bath spins in a given cluster :math:`C`
(with the cluster Hamiltonian :math:`\hat H_C`),
and :math:`\tilde{L}_{C'}` are contributions of :math:`C'` subcluster of :math:`C`.

For example, the contribution of the single spin :math:`i` is equal
to the coherence function of the bath with one isolated spin :math:`i`:

.. math::
    \tilde{L}_i = L_{i}

The contribution of pair of spins :math:`i` and :math:`j` is equal to:

.. math::
    \tilde{L}_{ij} = \frac{L_{ij}}{\tilde{L}_i \tilde{L}_j}

and so on.

Maximum size of the cluster included into the expansion determines the order of CCE approximation.
For example, in the CCE2 approximation, only contributions up to spin pairs are included, and
in CCE3 - up to triplets of bath spins are included, etc.

The way the coherence function for each cluster
is computed slightly varies between depending on whether the conventional or generalized CCE method is used.

Conventional CCE
..................................
In the original formulation of the CCE method, the total Hamiltonian of the system
is reduced to the sum of two effective Hamiltonians, conditioned on the qubit levels of the central spin:

.. math::

    \hat H = \ket{0}\bra{0}\otimes\hat H^{(0)} + \ket{1}\bra{1}\otimes\hat H^{(1)}

Where :math:`\hat H^{(\alpha)}` is an effective Hamiltonian acting on the bath
when the central qubit is in the :math:`\ket{\alpha}` state
(:math:`\ket{\alpha}=\ket{0},\ket{1}` is one of the two eigenstates of the :math:`\hat H_S` chosen as qubit levels).


Given an initial qubit state :math:`\ket{\psi}=\frac{1}{\sqrt{2}}(\ket{0}+e^{i\phi}\ket{1})`
and an initial state of the bath spin cluster :math:`C` characterized by the density matrix :math:`\hat \rho_{C}`,
the coherence function of the qubit interacting with the cluster :math:`C` is computed as:

.. math::

    L_{C}(t) = Tr[\hat U_C^{(0)}(t)\hat \rho_C \hat U_C^{(1) \dagger}(t)]

Where :math:`\hat U_C^{(\alpha)}(t)` is time propagator defined in terms of the effective Hamiltonian
:math:`\hat H_C^{(\alpha)}` and the number of decoupling pulses. Note that :math:`\hat H_C^{(\alpha)}` here includes
only degrees of freedom of the given cluster.

For free induction decay (FID) the time propagators are trivial:

.. math::

    \hat U_C^{(0)} = e^{-\frac{i}{\hbar} \hat H_C^{(0)} t};\
    \hat U_C^{(1)} = e^{-\frac{i}{\hbar} \hat H_C^{(1)} t}

And for the generic decoupling sequence with :math:`N` (even)
decoupling pulses applied at :math:`t_1, t_2...t_N` we write:

.. math::

    \hat U^{(\alpha)}(t) = e^{-\frac{i}{\hbar} \hat H_C^{(\alpha)} (t_{N} - t_{N-1})}
                           e^{-\frac{i}{\hbar} \hat H_C^{(\beta)} (t_{N-1} - t_{N-2})}
                           ...
                           e^{-\frac{i}{\hbar} \hat H_C^{(\beta)} (t_{2} - t_{1})}
                           e^{-\frac{i}{\hbar} \hat H_C^{(\alpha)} t_{1}}

Where :math:`\ket{\alpha} = \ket{0}, \ket{1}` and :math:`\ket{\beta} = \ket{1}, \ket{0}` accordingly
(when :math:`\ket{\alpha} = \ket{0}` one should take :math:`\ket{\beta} = \ket{1}` and vice versa).
:math:`t=\sum_i{t_i}` is the total evolution time.
In sequences with odd number of pulses `N`, the leftmost propagator is the exponent of :math:`\hat H_C^{(\beta)}`.

Generalized CCE
..................................


Instead of projecting the total Hamiltonian on the qubit levels,
one may directly include the central spin degrees of freedom to each clusters.
We refer to such formulation as gCCE.

In this case we write the cluster Hamiltonian as:

.. math::

    \hat H_C & {} =  \mathbf{SDS} + \mathbf{B\gamma}_{S}\mathbf{S} +
                     \sum_{i\in C} \mathbf{S} \mathbf{A}_i \mathbf{I}_i +
                     \sum_{i\in C} \mathbf{I}_i\mathbf{P}_i \mathbf{I}_i +
                     \mathbf{B}\mathbf{\gamma}_i\mathbf{I}_i +  \\
             & \sum_{i<j \in C} \mathbf{I}_i \mathbf{J}_{ij} \mathbf{I}_j +
               \sum_{a \notin C} \mathbf{S} \mathbf{A}_a \langle\mathbf{I}_a\rangle +
               \sum_{i\in C,\ a\notin C} {\mathbf{I}_i\mathbf{J}_{ia}\langle\mathbf{I}_a\rangle}


And the coherence function of the cluster :math:`L_C(t)` is computed as:

.. math::

    L_{C}(t) = \bra{0}\hat U_C(t)\hat \rho_{C+S} \hat U_C^{\dagger}(t)\ket{1}

Where :math:`\hat \rho_{C+S} = \hat \rho_{C} \otimes \hat \rho_S` is the combined initial density matrix
of the bath spins' cluster and central spin.

Further details on the theoretical background are available in the references below.

.. [#code] Mykyta Onizhuk and Giulia Galli. "PyCCE: A Python Package for Cluster Correlation Expansion Simulations of Spin Qubit Dynamic".
       Adv. Theory Simul. 2021, 2100254, https://onlinelibrary.wiley.com/doi/10.1002/adts.202100254
.. [#yang2008] Wen Yang  and  Ren-Bao  Liu.  “Quantum  many-body  theory  of qubit
       decoherence in a finite-size spin bath”.
       Phys. Rev. B78, p. 085315, https://link.aps.org/doi/10.1103/PhysRevB.78.085315
.. [#onizhuk2021] Mykyta  Onizhuk  et  al.
       “Probing  the  Coherence  of  Solid-State  Qubits  at Avoided  Crossings”.
       PRX Quantum 2, p. 010311. https://link.aps.org/doi/10.1103/PRXQuantum.2.010311.


