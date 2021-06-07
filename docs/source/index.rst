Welcome to pyCCE's documentation!
=================================

**pyCCE** is an open source Python library for conducting the simulations
of the spin qubit dynamics interacting with spin bath
by the means of Cluster Correlation Expansion (CCE) method.


Installation
----------------

The recommended way to install **pyCCE** package is to use the **pip**::

   $ pip install pycce

Otherwise you can install directly from the source code. First copy the repository to the desired folder::

    $ git clone https://github.com/foxfixfax/pycce.git


Then, execute **pip** in the folder containing **setup.py**::

    $ pip install .

or run the python install command::

    $ python setup.py install


**pyCCE** inherently supports parallelization with **mpi4py** package, which requires existing
MPI implementation to be installed on the system.
However, for serial implementation the **mpi4py** is not required.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   theory
   tutorial

.. toctree::
   :maxdepth: 1
   :caption: User Manual

   bath
   simulator
   parameters
   dft

.. toctree::
   :maxdepth: 1
   :caption: Additional Documentation

   calculators
   hamiltonian
   utilities



