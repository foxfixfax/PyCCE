
Generating the Spin bath
====================================================================

Random bath
---------------------------------------------------------------------
Documentation for the ``pycce.random_bath`` function, used to generate random bath.

.. autofunction:: pycce.bath.cell.random_bath


BathCell
---------------------------------------------------------------------

Documentation for the ``pycce.BathCell`` - class for convenient generation of ``BathArray`` and the
necessary helper functions.

.. automodule:: pycce.bath.cell
   :members:
   :exclude-members: random_bath


BathArray
-----------------------------------------------------------------------
Documentation for the ``pycce.BathArray`` - central class, containing properties of the bath spins.

.. automodule:: pycce.bath.array
   :members:
   :exclude-members: concatenate, implements, update_bath, transform, SpinType, SpinDict, common_isotopes, common_concentrations

.. automethod:: pycce.utilities.rotmatrix
   :noindex:

InteractionMap
..............................................................

.. automodule:: pycce.bath.map
   :members:

Cube
..............................................................

.. automodule:: pycce.bath.cube
   :members:

SpinDict and SpinType
-----------------------------------------------------------------------
Documentation for the ``SpinDict`` - dict-like class which describes
the properties of the different types of the spins in the bath.

.. autoclass:: pycce.SpinDict
   :members:

.. autoclass:: pycce.SpinType
   :members:

.. autodata:: pycce.bath.array.common_isotopes

.. autodata:: pycce.bath.array.common_concentrations
   :annotation: = {element ('H', 'He',...) : { isotope ('1H', '2H', ..) : concentration}}
