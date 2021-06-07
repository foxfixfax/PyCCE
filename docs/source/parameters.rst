Hamiltonian parameters input
==================================

The total Hamiltonian of the system is set as:

.. math::
    \hat H = \hat H_S + \hat H_{SB} + \hat H_{B}

with

.. math::

        &\hat H_S = \mathbf{SDS} + \mathbf{B\gamma}_{S}\mathbf{S} \\
        &\hat H_{SB} = \sum_i \mathbf{S}\mathbf{A}_i\mathbf{I}_i \\
        &\hat H_{B} = \sum_i{\mathbf{I}_i\mathbf{P}_i \mathbf{I}_i +
                      \mathbf{B}\mathbf{\gamma}_i\mathbf{I}_i} +
                      \sum_{i>j} \mathbf{I}_i\mathbf{J}_{ij}\mathbf{I}_j

Where :math:`\hat H_S` is the Hamiltonian of the free central spin,
:math:`\hat H_{SB}` denotes interactions between central spin and bath spin,
nd :math:`\hat H_B1` are intrinsic bath spin interactions.

Each of this terms can be defined within **pyCCE** framework as following.

In general, central spin properties are stored in the ``Simulator`` instance, bath properties are stored in the
``BathArray`` instance.

Central Spin Hamiltonian
..................................

The central spin Hamiltonian is provided as attributes of the ``Simulator`` object:

- :math:`\mathbf{D}` is set with ``Simulator.set_zfs`` method or during the initialization of the
  ``Simulator`` object either from observables *D* and *E* of the zero field
  splitting **OR** directly as tensor for the interaction :math:`\mathbf{SDS}`. By default is zero.
  Examples::

    >>> c = Simulator(1)
    >>> print(c.zfs)
    [[ 0.  0.  0.]
     [ 0. -0.  0.]
     [ 0.  0.  0.]]
    >>> c.set_zfs(D=1e6)
    >>> print(c.zfs)
    [[-333333.333       0.          0.   ]
     [      0.    -333333.333       0.   ]
     [      0.          0.     666666.667]]

- :math:`\mathbf{\gamma}_S`, the tensor describing
  the interaction of the spin and the external magnetic field in units of gyromagnetic ratio
  :math:`\mathrm{rad}\cdot\mathrm{kHz}\cdot\mathrm{G}^{-1}`.
  By default is equal to the gyromagnetic ratio of the free electron spin,
  :math:`-17609\ \mathrm{rad}\cdot\mathrm{kHz}\cdot\mathrm{G}^{-1}`.

  For the electron spin, it is proportional
  to g-tensor :math:`\mathbf{g}` as:

  .. math:: \mathbf{\gamma}_S=\mathbf{g}\frac{\mu_B}{\hbar},

  where :math:`\mu_B` is Bohr magneton.

  For the nuclear central spin, it is proportional to the chemical shift tensor :math:`\mathbf{\sigma}`
  and gyromagnetic ratio :math:`\gamma` as:

  .. math:: \mathbf{\gamma}_S=\gamma(1 - \mathbf{\sigma})

  Examples::

    >>> c = Simulator(1)
    >>> print(c.gyro)
    -17608.59705

The magnetic field is set with  with ``Simulator.set_magnetic_field`` method or during the initialization of the
``Simulator`` object in :math:`\mathrm{G}`.

Spin-Bath Hamiltonian
........................................

The interactions between central spin and bath spins and are provided
in the ``['A']`` namefield of the ``BathArray`` object in :math:`\mathrm{rad}\cdot\mathrm{kHz}`.

Interaction tensors can be either:

- Directly provided by setting the values of ``bath['A']`` in :math:`\mathrm{rad}\cdot\mathrm{kHz}`
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

    >>> bath = random_bath('13C', size=100, number=5)
    >>> print(bath)
    [('13C', [-27.301,  41.65 ,  11.875], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
     ('13C', [ 35.592, -49.73 , -12.323], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
     ('13C', [ -4.312,  25.681,  20.731], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
     ('13C', [ 21.515, -42.781,  -8.355], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
     ('13C', [-25.785,  12.88 ,  17.051], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])]
    >>> bath['A'] = 1
    >>> print(bath)
    [('13C', [ -8.642,  -7.911,  35.306], [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
     ('13C', [ 48.173,  18.067, -18.275], [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
     ('13C', [  9.065,  34.015,  12.759], [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
     ('13C', [-31.95 ,  -9.597, -11.963], [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
     ('13C', [-22.77 ,  47.308,   0.334], [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])]
    >>> bath.from_point_dipole([0, 0, 0])
    >>> print(bath)
    [('13C', [-24.766, -40.571,  21.094], [[-0.284,  0.99 , -0.515], [ 0.99 ,  0.734, -0.843], [-0.515, -0.843, -0.45 ]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
     ('13C', [-21.571, -35.516,  22.077], [[-0.443,  1.245, -0.774], [ 1.245,  0.85 , -1.274], [-0.774, -1.274, -0.407]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
     ('13C', [-29.012,  41.195,  47.466], [[-0.178, -0.282, -0.325], [-0.282,  0.024,  0.461], [-0.325,  0.461,  0.155]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
     ('13C', [-48.929, -43.779,  19.717], [[ 0.205,  0.53 , -0.239], [ 0.53 ,  0.087, -0.214], [-0.239, -0.214, -0.292]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
     ('13C', [-46.998,  24.897,  43.546], [[ 0.155, -0.286, -0.5  ], [-0.286, -0.233,  0.265], [-0.5  ,  0.265,  0.078]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])]

Bath Hamiltonian
..................................
The self interaction tensors of the bath spins ae stored in the ``['Q']`` namefield of the ``BathArray`` object.
By default they are set to 0. They can be either:

- Directly provided by setting the values of ``bath['Q']`` in :math:`\mathrm{rad}\cdot\mathrm{kHz}`
  for each bath spin.
- Computed from the electric field gradient (EFG) tensors at each bath spin position,
  using ``BathArray.from_efg`` method.

The gyromagnetic ratio :math:`\gamma_j` of each bath spin type is stored in the ``BathArray.types``.

The couplings between bath spins are assumed to follow point dipole-dipole interactions as:

.. math::

    \mathbf{P}_{ij} = -\gamma_{i} \gamma_{j} \frac{\hbar^2}{4\pi \mu_0}
                       \left[ \frac{3 \vec{r_{ij}} \otimes \vec{r_ij} - |r_{ij}|^2 I}{|r_{ij}|^5} \right]

Where :math:`\gamma_{i}` is gyromagnetic ratio of `i` tensor, :math:`I` is 3x3 identity matrix, and
:math:`\vec{r_{ij}` is distance between two vectors.

However, user can define the interaction tensors for specific bath spin pairs stored in the ```BathArray`` instance.
This can be achieved by:

    - Calling ``BathArray.add_interaction`` method of the ``BathArray`` instance.
    - Providing ``InteractionsMap`` instance as ``imap`` keyword to the ``Simulator.read_bath``.

Examples::

    >>> import numpy as np
    >>> bath = random_bath('13C', size=100, number=5)
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

