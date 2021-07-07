
CCE Calculators
==========================================

Documentation for the calculator objects called by ``Simulator`` object.

Base class
--------------------------------------------

.. automodule:: pycce.run.base
   :members:

Conventional CCE
------------------------------------------

.. automodule:: pycce.run.cce
   :members:

Generalized CCE
-------------------------------------------

.. automodule:: pycce.run.gcce
   :members:

Noise Autocorrelation
-----------------------------------------------

.. automodule:: pycce.run.corr
   :members:

Cluster-correlation Expansion Decorators
------------------------------------------

The way we find cluster in the code.

.. automodule:: pycce.find_clusters
   :members:

General decorators that are used to expand kernel of the ``RunObject`` class or subclasses to the whole bath *via* CCE.

.. automodule:: pycce.run.clusters
   :members:

Decorators that are used to perform bath state sampling over the kernel of ``RunObject``.

.. automodule:: pycce.run.mc
   :members:
