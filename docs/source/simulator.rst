Running the Simulations
=========================

Setting up the Simulator Object
--------------------------------------
Documentation for the ``pycce.Simulator`` - main class for conducting CCE Simulations.

.. autoclass:: pycce.main.Simulator
   :members:
   :inherited-members: Environment
   :exclude-members: compute, cce_coherence, gcce_coherence, cce_noise, gcce_noise, read_bath, generate_clusters,

Reading the Bath
--------------------------------------
Documentation for the ``Simulator.read_bath`` and ``Simulator.generate_clusters``
method. These methods are called automatically on the initialization of the ``Simulator`` object
if the necessary keywords are provided. Otherwise they can also be called by themselves
to update the properties of the spin bath in ``Simulator`` object.

.. automethod:: pycce.main.Simulator.read_bath
.. automethod:: pycce.main.Simulator.generate_clusters

Calculate Properties with Simulator
-------------------------------------
Documentation for the ``Simulator.compute`` method -
the interface to run calculations with **PyCCE**.

.. automethod:: pycce.main.Simulator.compute

Pulse sequences
------------------------
Documentation of the ``Pulse`` and ``Sequence`` classes, used in definition of the complicated
pulse sequences.

.. automodule:: pycce.run.pulses
   :members:




