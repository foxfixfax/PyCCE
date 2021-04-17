.. pyCCE documentation master file, created by
sphinx-quickstart on Mon Apr  5 20:36:31 2021.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive.

Welcome to pyCCE's documentation!
=================================

**pyCCE** is an open source Python library for conducting the simulations
of the spin qubit dynamics interacting with spin bath with the means of Cluster Correlation Expansion (CCE) method.

pyCCE inherently supports parallelization with **mpi4py** package, which requires existing
MPI implenetation to be installed on the system. The required

Installation
----------------

The recommended way to install **pyCCE** package is to use the **pip** ::
  $ pip install pycce

Otherwise you can install directly from the source code. First copy the repository to the desired folder ::
  $ git clone https://github.com/foxfixfax/pycce.git

Then, execute **pip** in the folder containing **setup.py**::
  $ pip install .

or run the python install command::
  $ python setup.py install


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart
   tutorial

.. toctree::
   :maxdepth: 1
   :caption: User Manual

   simulator

.. toctree::
   :maxdepth: 1
   :caption: Contributor Documentation

   contributing

