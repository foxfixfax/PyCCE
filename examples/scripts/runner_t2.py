import numpy as np
from mpi4py import MPI
import sys
import os
import time
from parser import pcparser

import pycce as pc

import pandas as pd
import ase.io

from ase.build import bulk


def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    try:
        os.mkdir(dir)

    except FileExistsError:
        pass

    return


def folname(calc_setup):
    name = 'thickness_{thickness}_rbath_{rbath}_rdipole_{rdipole}_order_{order}/'.format(**calc_setup)
    return name


seed = 1

calc_param = {'B': np.array([0., 0., 500.]), 'N': 1}


center = np.array([0, 0, 0])
alpha = np.array([0, 0, 1])
beta = np.array([0, 1, 0])


if __name__ == '__main__':

    arguments = pcparser.parse_args()

    comm = MPI.COMM_WORLD

    size = comm.Get_size()
    rank = comm.Get_rank()

    conf = rank + arguments.start


    diamond = bulk('C', 'diamond', cubic=True)
    diamond = pc.bath.BathCell.from_ase(diamond)
    # Add types of isotopes
    diamond.add_isotopes(('13C', 0.011))
    # set z direction of the defect
    diamond.zdir = [1, 1, 1]

    offset = diamond.to_cartesian([1, 0, 0])
    length = np.linalg.norm(offset)

    R = pc.bath.rotmatrix([0, 0, 1], ofs)

    atoms = diamond.gen_supercell(400, seed=seed + conf)
    atoms = atoms[np.einsum('ij,kj->ki', R.T, atoms['xyz'])[:, 2] >= 0]


    fp = [1.278e-03, 1.836e+00, 9.568e-01]

    calc_setup = vars(arguments)

    ls = []

    for v in arguments.values:

        calc_setup[arguments.param] = v
        vfol = fol + folname(calc_setup)
        mkdir_p(vfol)
        thickness = calc_setup['thickness']
        maxtime = 4 * (fp[0] * (thickness * 2 * length)**fp[1] + fp[2])

        time_space = np.linspace(0, maxtime, 501)

        r_bath = calc_setup['rbath'] + thickness * length

        ntop['xyz'] = top['xyz'] + ofs * thickness
        nbottom['xyz'] = bottom['xyz'] - ofs * thickness

        atoms = np.r_[ntop, nbottom, ai]

        if thickness == 0:
            atoms = pc.defect(diamond.cell, atoms, remove=[('C', [0., 0, 0]),
                                                           ('C', [0.5, 0.5, 0.5])])

        calc = pc.Simulator(1, center, alpha=alpha, beta=beta, bath=atoms,
                            r_bath=r_bath,
                            r_dipole=calc_setup['rdipole'],
                            order=calc_setup['order'])

        L = calc.compute_coherence(time_space, as_delay=False, **calc_param)

        if arguments.param == 'thickness':
            v = v * length * 2

        cfl = pd.Series(np.abs(L), index=time_space, name=v)
        cfl.index.name = 'Time (ms)'

        cfl.to_csv(os.path.join(vfol, 'pyCCE_{}.csv'.format(conf)))
        ls.append(cfl)

