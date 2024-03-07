
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
  :scale: 30%
  :align: left
  :target: index.html


**PyCCE** is an open source Python library to simulate the dynamics of
a spin qubit interacting with a spin bath using the cluster-correlation expansion (CCE) method.


Major Updates
-----------------------------

PyCCE 1.1
++++++++++++++++++
New version of **PyCCE** includes new cluster solvers and a set of little bugfixes. Main changes include

* Implementation of the master equation-based CCE approaches. 
    Checkout the :doc:`tutorials/mecce` for examples of the usage.

* Various optimization and bugfixes.

PyCCE 1.0
++++++++++++++++++
The **PyCCE** 1.0 has been released!
Main changes from the previous version include:

* Support for several central spins with the new class ``CenterArray``!
    Check out a tutorial :doc:`tutorials/second_spin` on how to use the new class to study the decoherence
    of the hybrid qubit or entanglement of dipolarly coupled qubits.

* Direct definition of the bath spin states with ``BathArray.state`` attribute.
    Check out the updated tutorial :doc:`tutorials/diamond_nv` to see how one can use this functionality
    to study the effect of spin polarization on Hahn-echo signal.

* Expanded the control over pulse sequences.
    See documentation for ``Pulse`` class in :doc:`simulator` for details.

* *EXPERIMENTAL FEATURE*. Added ability to define your own single particle Hamiltonian.
    See ``BathArray.h`` and ``Center.h`` in :doc:`bath` and :doc:`center` respectively for further details.

* Significant overhaul of computational expensive parts of the code with Numba. This makes the first run of
  **PyCCE** quite slow, but after compilation it should run observably faster.

* Various bug fixes and QoL changes.

This is a major update. If you find any issues ot bugs, please let us know as soon as possible!

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

*  `Python <http://www.python.org/>`_ (version >= 3.9).

* `NumPy <https://numpy.org/>`_ (version >= 1.16).

* `SciPy <https://www.scipy.org/>`_ (version >= 1.10).

* `Numba <http://numba.pydata.org/>`_ (version >= 0.56).

* `Atomic Simulation Environment (ASE) <https://wiki.fysik.dtu.dk/ase/>`_.

* `Pandas <https://pandas.pydata.org/>`_.

**PyCCE** inherently supports parallelization with the **mpi4py** package, which requires the installation of MPI.
However, for serial implementation the **mpi4py** is not required.

How to cite
--------------------------
If you make use of **PyCCE** in a scientific publication, please cite the following paper:

   Mykyta  Onizhuk and Giulia Galli. "PyCCE: A Python Package for Cluster Correlation Expansion Simulations of Spin Qubit Dynamic"
   Adv. Theory Simul. 2021, 2100254 https://onlinelibrary.wiley.com/doi/10.1002/adts.202100254
