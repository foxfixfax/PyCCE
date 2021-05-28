Theoretical Background
===========================

Hamiltonian
----------------------------

The pyCCE package allows to simulate the dynamics of the central spin interacting with the spin bath by
the following Hamiltonian:

.. math::
    \hat H = \hat H_S + \hat H_{SB} + \hat H_{B}

Where :math:`\hat H_S` is the Hamiltonian of the free central spin,
:math:`\hat H_{SB}` denotes interactions between central spin and bath spin,
and :math:`\hat H_B1` are intrinsic bath spin interactions:

.. math::

        &\hat H_S = \mathbf{SDS} + \mathbf{Bg}_{S}\mathbf{S} \\
        &\hat H_{SB} = \sum_i \mathbf{S}\mathbf{A}_i\mathbf{I}_i \\
        &\hat H_{B} = \sum_i{\mathbf{I}_i\mathbf{P}_i \mathbf{I}_i +
                       \mathbf{B}\mathbf{g}_i\mathbf{I}_i} + \sum_{i>j} \mathbf{I}_i\mathbf{J}_{ij}\mathbf{I}_j

Where  :math:`\mathbf{S}=(\hat{S}_x, \hat{S}_y, \hat{S}_z)` is the vector of spin operators of the central spin,
:math:`\mathbf{I}=(\hat{I}_x, \hat{I}_y, \hat{I}_z)` is the vector of the bath spin operators.
The interactions are described with the tensors:

- :math:`\mathbf{D}` (:math:`\mathbf{P}`) is the self interaction tensor of the central spin (bath spin).
  For the electron spin, corresponds to the Zero field splitting (ZFS) tensor.
  For nuclear spins corresponds to the quadrupole interactions tensor.
- :math:`\mathbf{g}_i`$` is the g-tensor of the :math:`i`-spin describing the interaction of the spin
  and the external magnetic field.
- :math:`\mathbf{A}` is the interaction tensor between central and bath spins.
  In the case of nuclear spin bath, corresponds to the hyperfine couplings.
- :math:`\mathbf{J}` is the interaction tensor between bath spins.

Qubit dephasing
---------------------------------

In the pure dephasing regime (:math:`T_1 >> T_2`) the decoherence of the central spin is characterized by
the decay of the off diagonal element of the density matrix of the qubit.
I.e. if the qubit is initially prepared in the
:math:`\left|{+psi}\right\rangle = \frac{1}{\sqrt{2}}(\left|{0}\right\rangle+e^{i\phi}\left|{1}\right\rangle)` state,
the loss of the relative phase between :math:`\left|{0}\right\rangle` and :math:`\left|{1}\right\rangle`
levels is characterized by the coherence function:

.. math::

    \mathcal{L}(t) = \frac{\left\langle{0}\right|\hat{\rho}_S(t)\left|{1}\right\rangle}
     {\left\langle{0}\right|\hat{\rho}_S(0)\left|{1}\right\rangle} = \langle{\hat \sigma_+(t)}\rangle

Where :math:`\hat{\rho}_S(t)` is the density matrix of the central spin and
:math:`\left|{0}\right\rangle` and :math:`\left|{1}\right\rangle` are qubit levels.

The core idea of CCE approach is that the spin bath-induced decoherence
can be factorized into set of irreducible contributions from the bath spin clusters.
Written in terms of the coherence function:

.. math::
    \mathcal{L}(t) = \prod_{C} \tilde{L}_C = \prod_{i}\tilde{L}_{\{i\}}\prod_{i,j}\tilde{L}_{\{ij\}}...

Where each cluster contribution is defined recursively as:

.. math::
    \tilde{L}_C = \frac{L_{C}}{\prod_{C'}\tilde{L}_{C'\subset C}}

Where :math:`L_{C}` is a coherence function of the qubit,
interacting only with the nuclear spins in the given cluster :math:`C`,
and :math:`\tilde{L}_{C'}` are contributions of :math:`C'` subcluster of :math:`C`.

