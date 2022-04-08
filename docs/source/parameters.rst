Hamiltonian Parameters Input
==================================

The default total Hamiltonian of the system is set as:

.. math::
    \hat H = \hat H_S + \hat H_{SB} + \hat H_{B}

with

.. math::

        &\hat H_S = \sum_i (\mathbf{S}_i \mathbf{D}_i \mathbf{S}_i +
                    \mathbf{B\gamma}_{S_i}\mathbf{S}_i +
                    \sum_{i<j}\mathbf{S}_i \mathbf{K}_{ij} \mathbf{S}_j) \\
        &\hat H_{SB} = \sum_{i,k} \mathbf{S}_i \mathbf{A}_{ik} \mathbf{I}_k \\
        &\hat H_{B} = \sum_k{\mathbf{I}_k\mathbf{P}_k \mathbf{I}_k +
                      \mathbf{B}\mathbf{\gamma}_k\mathbf{I}_k} +
                      \sum_{k<l} \mathbf{I}_k\mathbf{J}_{kl}\mathbf{I}_l

Where :math:`\hat H_S` is the Hamiltonian of the free central spin,
:math:`\hat H_{SB}` denotes interactions between central spin and bath spin,
nd :math:`\hat H_B` are intrinsic bath spin interactions:

- :math:`\mathbf{D}` (:math:`\mathbf{P}`) is the self interaction tensor of the central spin (bath spin).
  For the electron spin, corresponds to the Zero field splitting (ZFS) tensor.
  For nuclear spins corresponds to the quadrupole interactions tensor.
- :math:`\mathbf{\gamma}_i`$` is the magnetic field interaction tensor
  of the :math:`i`-spin describing the interaction of the spin and the external magnetic field.
- :math:`\mathbf{A}` is the interaction tensor between central and bath spins.
  In the case of nuclear spin bath, corresponds to the hyperfine couplings.
- :math:`\mathbf{J}` (:math:`\mathbf{K}`) is the interaction tensor between bath (center) spins.

Each of this terms and additional terms of the Hamiltonian can be defined within **PyCCE** framework as following.

In general, central spin properties are stored in the ``CenterArray`` instance, bath properties are stored in the
``BathArray`` instance.

Central Spin Hamiltonian
..................................

The central spin Hamiltonian is provided as attributes of the ``CenterArray`` object:

- :math:`\mathbf{D}` is set with ``CenterArray.set_zfs`` method or during the initialization of the
  ``Simulator`` object either from observables *D* and *E* of the zero field
  splitting **OR** directly as tensor for the interaction :math:`\mathbf{SDS}` in  :math:`\mathrm{kHz}`.
  By default is zero.

  Examples::

    >>> c = CenterArray(spin=1)
    >>> print(c[0].zfs)
    [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
    >>> c[0].set_zfs(D=1e6)
    >>> print(c[0].zfs)
    [[-333333.33333       0.            0.     ]
     [      0.      -333333.33333       0.     ]
     [      0.            0.       666666.66667]]

- :math:`\mathbf{\gamma}_S`, the tensor describing
  the interaction of the spin and the external magnetic field in units of gyromagnetic ratio
  :math:`\mathrm{rad}\cdot\mathrm{kHz}\cdot\mathrm{G}^{-1}`.
  By default is equal to the gyromagnetic ratio of the free electron spin,
  :math:`-17609\ \mathrm{rad}\cdot\mathrm{ms}^{-1}\cdot\mathrm{G}^{-1}`.

  For the electron spin, it is proportional
  to g-tensor :math:`\mathbf{g}` as:

  .. math:: \mathbf{\gamma}_S=\mathbf{g}\frac{\mu_B}{\hbar},

  where :math:`\mu_B` is Bohr magneton.

  For the nuclear central spin, it is proportional to the chemical shift tensor :math:`\mathbf{\sigma}`
  and gyromagnetic ratio :math:`\gamma` as:

  .. math:: \mathbf{\gamma}_S=\gamma(1 - \mathbf{\sigma})

  Examples::

    >>> c = CenterArray(spin=1)
    >>> print(c[0].gyro)
    -17608.59705

  .. note::

      While all other coupling parameters are given in the units of frequency, the gyromagnetic ratio
      (and therefore tensors coupling magnetic field with the spin)
      are conventionally given in the units of **angular** frequency and differ by :math:`2\pi`.

- :math:`\mathbf{K}` is set with ``CenterArray.add_interaction`` method
  or by calling ``CenterArray.point_dipole`` method, assuming the interactions
  as the ones between magnetic point dipoles.

The magnetic field is set with  with ``Simulator.set_magnetic_field`` method or during the initialization of the
``Simulator`` object in Gauss (:math:`\mathrm{G}`).

User-defined terms of the single-particle central spin Hamiltonian
can be added by adding entries to the ``Center.h`` attribute
(Separate for each ``Center`` object in ``CenterArray``).

For example, to add Stevens operator :math:`B^q_k \hat O^q_k = 3 \hat S_z - s(s+1) \hat I`
with :math:`q=0`, :math:`k=2`, and :math:`B^q_k = 1 \mathrm{GHz}`
to the central spin Hamiltonian, one needs to add::

    >>> c = CenterArray(spin=1)
    >>> k, q = 2, 0
    >>> c.h[k, q] = 1e6 # in KHz

For details see ``Center`` documentation.

Spin-Bath Hamiltonian
........................................

The interactions between central spin and bath spins and are provided
in the ``.A`` attribute of the ``BathArray`` object in :math:`\mathrm{kHz}`.

Interaction tensors can be either:

- Directly provided by setting the values of ``bath.A`` in :math:`\mathrm{kHz}`
  for each bath spin.
- Approximated from magnetic point dipole–dipole interactions by calling ``BathArray.from_point_dipole`` method.
  Then the tensors are computed as:

  .. math::

    \mathbf{A}_{j} = -\gamma_{S} \gamma_{j} \frac{\hbar^2}{4\pi \mu_0}
                       \left[ \frac{3 \vec{r_{j}} \otimes \vec{r_j} - |r_{ij}|^2 I}{|r_{j}|^5} \right]

  Where :math:`\gamma_{j}` is gyromagnetic ratio of `j` spin, :math:`\vec{r_j}` is position of the bath spin,
  and :math:`I` is 3x3 identity matrix. The default option when reading the bath by ``Simulator`` object.

- Approximated from the spin density distribution of the central spin by calling ``BathArray.from_cube`` method.

  Examples::

    >>> bath = random_bath('13C', size=100, number=5, seed=1)
    >>> print(bath)
    [('13C', [  1.182,  45.046, -35.584], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
     ('13C', [ 44.865, -18.817,  -7.667], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
     ('13C', [ 32.77 ,  -9.08 ,   4.959], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
     ('13C', [-47.244,  25.351,   3.814], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
     ('13C', [-17.027,  28.843, -19.681], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])]
    >>> bath.A = 1
    >>> print(bath)
    [('13C', [  1.182,  45.046, -35.584], [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
     ('13C', [ 44.865, -18.817,  -7.667], [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
     ('13C', [ 32.77 ,  -9.08 ,   4.959], [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
     ('13C', [-47.244,  25.351,   3.814], [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
     ('13C', [-17.027,  28.843, -19.681], [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])]
    >>> bath.from_point_dipole([0, 0, 0])
    >>> print(bath)
    [('13C', [  1.182,  45.046, -35.584], [[-0.659,  0.032, -0.025], [ 0.032,  0.559, -0.963], [-0.025, -0.963,  0.1  ]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
     ('13C', [ 44.865, -18.817,  -7.667], [[ 1.558, -1.092, -0.445], [-1.092, -0.588,  0.187], [-0.445,  0.187, -0.97 ]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
     ('13C', [ 32.77 ,  -9.08 ,   4.959], [[ 5.32 , -2.327,  1.271], [-2.327, -2.434, -0.352], [ 1.271, -0.352, -2.886]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
     ('13C', [-47.244,  25.351,   3.814], [[ 1.06 , -1.   , -0.151], [-1.   , -0.268,  0.081], [-0.151,  0.081, -0.792]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
     ('13C', [-17.027,  28.843, -19.681], [[-0.903, -2.081,  1.42 ], [-2.081,  1.393, -2.405], [ 1.42 , -2.405, -0.49 ]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])]

Bath Hamiltonian
..................................

The self interaction tensors of the bath spins is stored in the ``.Q`` attribute of the ``BathArray`` object.
By default they are set to 0. They can be either:

- Directly provided by setting the values of ``bath.Q`` in :math:`\mathrm{kHz}`
  for each bath spin.
- Computed from the electric field gradient (EFG) tensors at each bath spin position,
  using ``BathArray.from_efg`` method.

The gyromagnetic ratio :math:`\gamma_j` of each bath spin type is stored in the ``BathArray.types``.

The couplings between bath spins are assumed to follow point dipole-dipole interactions as:

.. math::

    \mathbf{P}_{ij} = -\gamma_{i} \gamma_{j} \frac{\hbar^2}{4\pi \mu_0}
                       \left[ \frac{3 \vec{r_{ij}} \otimes \vec{r_ij} - |r_{ij}|^2 I}{|r_{ij}|^5} \right]

Where :math:`\gamma_{i}` is gyromagnetic ratio of `i` tensor, :math:`I` is 3x3 identity matrix, and
:math:`\vec{r_{ij}}` is distance between two vectors.

However, user can define the interaction tensors for specific bath spin pairs stored in the ```BathArray`` instance.
This can be achieved by:

    - Calling ``BathArray.add_interaction`` method of the ``BathArray`` instance.
    - Providing ``InteractionsMap`` instance as ``imap`` keyword to the ``Simulator.read_bath``.

Examples::

    >>> import numpy as np
    >>> bath = random_bath('13C', size=100, number=5, seed=1)
    >>> print(bath.types)
    SpinDict(13C: (13C, 0.5, 6.7283))
    >>> test_tensor = np.random.random((3, 3))
    >>> bath.add_interaction(0, 1, (test_tensor + test_tensor.T) / 2)
    >>> print(bath.imap[0, 1])
    [[0.786 0.53  0.404]
     [0.53  0.821 0.366]
     [0.404 0.366 0.655]]
    >>> print(bath.imap[0, 1])
    [[0.786 0.53  0.404]
     [0.53  0.821 0.366]
     [0.404 0.366 0.655]]

User-defined terms of the single-particle bath spin Hamiltonian
can be added by adding entries to the ``BathArray.h`` attribute
(Separate for each type of bath spin).

For example, to add non-linear term :math:`A I_x^4`
with :math:`A = 1 \mathrm{MHz}` to the :math:`^{13}C` bath spins (which for spin-1/2 is just proportional to identity,
but for higher spins can be relevant) to the bath spin Hamiltonian, one needs to add::

    >>> bath['13C'].h['xxxx'] = 1e3 # in kHz

For details see ``BathArray`` documentation.