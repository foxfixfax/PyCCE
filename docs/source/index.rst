
PyCCE: A Python Package for CCE Simulations
=========================================================================================


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting Started

   Installation <self>
   theory
   quickstart
   tutorial

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: User Documentation

   bath
   center
   simulator
   parameters
   dft

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Developer Documentation

   calculators
   hamiltonian
   utilities
   dftbase

.. figure:: logo.png
  :scale: 40%
  :align: left
  :target: index.html

**PyCCE** is an open source Python library to simulate the dynamics of
a spin qubit interacting with a spin bath using the cluster-correlation expansion (CCE) method.

Updates
------------
The **PyCCE** 1.0 has been released!

Main highlights from the different

Installation
----------------

The recommended way to install **PyCCE** is to use **pip**::

    $ pip install pycce

Otherwise you can install  **PyCCE** directly using the source code.
First copy the repository to the desired folder::

    $ git clone https://github.com/foxfixfax/pycce.git


Then, execute **pip** in the folder containing **setup.py**::

    $ pip install .

or run the python install command::

    $ python setup.py install


Requirements
----------------
The following modules are required to run **PyCCE**.

*  `Python <http://www.python.org/>`_ (version >= 3.6).

* `NumPy <https://numpy.org/>`_ (version >= 1.16).

* `SciPy <https://www.scipy.org/>`_.

* `Numba <http://numba.pydata.org/>`_ (version >= 0.50).

* `Atomic Simulation Environment (ASE) <https://wiki.fysik.dtu.dk/ase/>`_.

* `Pandas <https://pandas.pydata.org/>`_.

**PyCCE** inherently supports parallelization with the **mpi4py** package, which requires the installation of MPI.
However, for serial implementation the **mpi4py** is not required.

How to cite
--------------------------
If you make use of **PyCCE** in a scientific publication, please cite the following paper:

   Mykyta  Onizhuk and Giulia Galli. "PyCCE: A Python Package for Cluster Correlation Expansion Simulations of Spin Qubit Dynamic"
   arXiv:2107.05843. https://arxiv.org/abs/2107.05843
