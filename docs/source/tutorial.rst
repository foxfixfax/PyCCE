
Quick Start and Tutorials
====================================================================

The pyCCE module allows to conveniently generate spin bath
and conduct the CCE dynamics simulations in the realistic spin bath.

The simplest example includes the following steps:

1. Generate the ``BathCell`` object.
   Here we use the interface with ``ase`` which can generate unit cells of many
   materials conveniently and effortlessly.

.. literalinclude:: tutorials/nv_simple.py
   :language: python
   :lines: 1-5

2. Using the ``BathCell`` object, generate the spin bath
   of the most common isotopes in the material. Here we generate the spin bath
   of size 200 Angstrom and remove the one carbon where our spin is located.

.. literalinclude:: tutorials/nv_simple.py
   :language: python
   :lines: 6

3. Setup the ``Simulator`` using the generated spin bath.
   The first required argument is the total spin of the central spin,
   ``r_bath``, ``r_dipole`` and ``order`` are convergence parameters,
   ``magnetic_field`` is the external applied magnetic field along z-axis,
   and ``pulses`` is number of decoupling :math:`\pi` pulses.

.. literalinclude:: tutorials/nv_simple.py
   :language: python
   :lines: 8,9

4. Compute the coherence of the qubit using ``Simulator.compute`` method.

.. literalinclude:: tutorials/nv_simple.py
   :language: python
   :lines: 11, 12


That's it! The more detailed tutorials on the **pyCCE** usage are available in the examples below.

The following examples are available as jupyter notebooks in the Github repository.


.. toctree::
   :maxdepth: 1
   :caption: Examples of using pyCCE Code

   tutorials/diamond_nv
   tutorials/sic_vv
   tutorials/si_shallow
   tutorials/classical_noise

The recommended order of the tutorials is from the top to bottom:

* :doc:`tutorials/diamond_nv` example goes through the example above in more details.
* :doc:`tutorials/sic_vv` example introduces the way to work with DFT output of hyperfine tensors.
* :doc:`tutorials/si_shallow` example shows the way to include the custom hyperfine couplings.
* :doc:`tutorials/classical_noise` example explains the way to use autocorrelation function of the noise.