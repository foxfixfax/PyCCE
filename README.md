# pyCCE
Python code for computing qubit dynamics in the central spin model with CCE method

### Installation
run 
`python setup.py install`
in the main folder

### Base Units
Gyromagnetic ratios are given in rad * kHz / G.
Magnetic field in G.
Timesteps in ms. 
Distances in A.
All coupling constants are given in rad * kHz

### Usage
Usage consists of two steps: preparation of spin bath `bath.BathArray` from `bath.BathCell` and calculations with `Simulator` class.
See examples notebook for examples of calculations

**TODO** Write more comprehensive README
