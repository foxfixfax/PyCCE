"""
script usage:

mpirun python nv_ensemble.py [-h] [--r_bath R_BATH] [--r_dipole R_DIPOLE] [--order ORDER]
                             [--nbstates NBSTATES] [--start START] [--pulses PULSES]
                             [--magnetic_field MAGNETIC_FIELD]
                             [param] [values [values ...]] 

positional arguments:
param                 varied parameter
values                values of varied parameter

optional arguments:                                                                                                       
-h, --help                          show this help message and exit
--r_bath R_BATH, -rb R_BATH         cutoff bath radius
--r_dipole R_DIPOLE, -rd R_DIPOLE   pair cutoff radius
--order ORDER, -o ORDER             CCE order
--start START, -s START             configurations start
--pulses PULSES, -N PULSES          number of pulses
"""

import numpy as np
from mpi4py import MPI
import os
import time
import sys
from parser import pcparser

import pandas as pd

from ase.build import bulk
import pycce as pc

seed = 1

# parameters of calcuations
calc_param = {'magnetic_field': np.array([0., 0., 500.]), 'pulses': 1}

# position of central spin
center = np.array([0, 0, 0])
# qubit levels
alpha = np.array([0, 0, 1])
beta = np.array([0, 1, 0])

# A script to calculate ensemble average coherence for nv_center in diamond
# Parallelization here is over different spatial configurations of spin bath
# However, the results for a single configuration can be parallelized with 
# parallel and parallel_states Simulator keywords

if __name__ == '__main__':
    # parse argument for Simulator instance
    # with console parser (defined in parser.py)
    arguments = pcparser.parse_args()

    maxtime = 2
    # check how long the calculations was
    time_space = np.linspace(0, maxtime, 501)

    # MPI stuff
    comm = MPI.COMM_WORLD

    size = comm.Get_size()
    rank = comm.Get_rank()

    # Each configuration is determined by rng seed as a sum of seed + conf
    stime = time.time()
    conf = rank + arguments.start
    # Set up BathCell
    diamond = bulk('C', 'diamond', cubic=True)
    diamond = pc.bath.BathCell.from_ase(diamond)

    # Add types of isotopes
    # set z direction of the defect
    diamond.zdir = [1, 1, 1]
    atoms = diamond.gen_supercell(200, seed=seed + conf,
                                  remove=('C', center))
    # transform parser into dictionary
    calc_setup = vars(arguments)

    # list to store calculations results
    ls = []
    # argument.values contains values of the varied parameter
    # arguments.param

    for v in arguments.values:
        calc_setup[arguments.param] = v
        # initiallize Simulator instance
        calc = pc.Simulator(1, center, alpha=alpha, beta=beta, bath=atoms,
                            r_bath=calc_setup['r_bath'],
                            r_dipole=calc_setup['r_dipole'],
                            order=calc_setup['order'])
        # compute coherence
        result = calc.compute(time_space, as_delay=False, method='cce',
                              nbstates=None, interlaced=False,**calc_param)
        ls.append(result)

    ls = np.asarray(ls)
    ls[np.abs(ls) > 1] = 1

    if rank == 0:
        average_ls = np.zeros(ls.shape, dtype=ls.dtype)
    else:
        average_ls = None

    comm.Reduce(ls, average_ls)
    if rank == 0:
        average_ls /= comm.size

        etime = time.time()

        print(f'Calculation of {len(arguments.values)} {arguments.param} took '
              f'{etime - stime:.2f} s for configuration {conf}')

        df = pd.DataFrame(average_ls.T, columns=arguments.values,
                          index=time_space)

        df.index.rename('Time', inplace=True)

        # write the calculation parameters into file
        with open(f'nv_{arguments.param}.csv', 'w') as file:
            tw = ', '.join(f'{a} = {b}' for a, b in calc_setup.items())
            file.write('# ' + tw + '\n')
            df.to_csv(file)
