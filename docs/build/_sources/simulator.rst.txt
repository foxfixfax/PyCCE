
Running the simulations
=========================

Setting up the Simulator object
--------------------------------------
Documentation for the ``pycce.Simulator`` - main class for conducting CCE Simulations.

.. autoclass:: pycce.main.Simulator
   :members: set_magnetic_field, set_states, eigenstates, read_bath, generate_clusters
   :inherited-members: Environment
   :exclude-members: compute, cce_coherence, gcce_dm, cce_noise, gcce_noise

Calculate properties with Simulator
-------------------------------------
Documentation for the ``Simulator.compute`` method and it's dependencies -
the interface to run calculations with **pyCCE**.

.. automethod:: pycce.main.Simulator.compute
.. automethod:: pycce.main.Simulator.cce_coherence
.. automethod:: pycce.main.Simulator.gcce_dm
.. automethod:: pycce.main.Simulator.cce_noise
.. automethod:: pycce.main.Simulator.gcce_noise



