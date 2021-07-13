
![image](docs/source/logo.png)
# PyCCE Code Repository
Welcome to the repository, containing source code of **PyCCE** - a Python library for computing qubit dynamics
in the central spin model with cluster-correlation expansion (CCE) method.

### Installation
Run 
`python setup.py install`
in the main folder.

### Base Units

* Gyromagnetic ratios are given in rad / ms / G.
* Magnetic field in G.
* Timesteps in ms. 
* Distances in A.
* All coupling constants are given in kHz.


### Usage

Usage consists of two steps: preparation of spin bath `BathArray` from `BathCell` and calculations with `Simulator` class.

See `examples` folder for tutorials and scripts of calculations.

### Documentation

Full documentation is available online at [Read the Docs](https://pycce.readthedocs.io/en/latest/). 

