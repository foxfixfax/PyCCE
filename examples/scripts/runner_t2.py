import numpy as np
from mpi4py import MPI
import os
import time
import sys
from parser import pcparser

import pandas as pd

from ase.build import bulk

sys.path.append('/home/onizhuk/codes_development/pyCCE/')
import pycce as pc


# helper function to make folder
def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    try:
        os.mkdir(dir)

    except FileExistsError:
        pass

    return


seed = 1

# parameters of calcuations
calc_param = {'magnetic_field': np.array([0., 0., 500.]), 'N': 1}

# position of central spin
center = np.array([0, 0, 0])
# qubit levels
alpha = np.array([0, 0, 1])
beta = np.array([0, 1, 0])

# A script to calculate coherence time for various thickness
# of isotopically purified diamond (growth along 100 axis)

if __name__ == '__main__':
    # check how long the calculations was
    stime = time.time()
    # parse argument for Simulator instance
    # with console parser (defined in parser.py)
    arguments = pcparser.parse_args()
    maxtime = 50
    time_space = np.geomspace(0.001, maxtime, 1001)

    # MPI stuff
    comm = MPI.COMM_WORLD

    size = comm.Get_size()
    rank = comm.Get_rank()

    # Each configuration is determined by rng seed as a sum of seed + conf
    conf = rank + arguments.start
    # Set up BathCell
    diamond = bulk('C', 'diamond', cubic=True)
    diamond = pc.bath.BathCell.from_ase(diamond)

    # Add types of isotopes
    diamond.add_isotopes(('13C', 0.011))
    # set z direction of the defect
    diamond.zdir = [1, 1, 1]
    # direction along which the material growth is performed
    offset = diamond.to_cartesian([1, 0, 0])
    # distance between two unit cell along chosen growth axis
    length = np.linalg.norm(offset)
    # rotation matrix, necessary to transform z direction to offset
    R = pc.bath.rotmatrix([0, 0, 1], offset)

    atoms = diamond.gen_supercell(400, seed=seed + conf)
    # top layers of spin bath are obtained as the ones with coordinate >= 0
    # along growth axis
    top = atoms[np.einsum('ij,kj->ki', R.T, atoms['xyz'])[:, 2] >= 0]
    # the bottom layer has coordinate < 0 along growth axis
    bottom = atoms[np.einsum('ij,kj->ki', R.T, atoms['xyz'])[:, 2] < 0]

    # transform parser into dictionary
    calc_setup = vars(arguments)

    # directory to store results
    fol = f'var_{arguments.param}'
    mkdir_p(fol)
    # list to store calculations results
    ls = []
    # argument.values contains values of the varied parameter
    # arguments.param
    for v in arguments.values:

        calc_setup[arguments.param] = v
        # Thickness of isotopically purified layer
        thickness = calc_setup['thickness']

        # r_bath is a sum of 'rbath' parameter and thickness of
        # purified layer
        r_bath = calc_setup['rbath'] + thickness * length
        ntop = top.copy()
        nbottom = bottom.copy()
        # move top and bottom by the thickness
        ntop['xyz'] = top['xyz'] + offset * thickness
        nbottom['xyz'] = bottom['xyz'] - offset * thickness
        # cimbine top and bottom back
        atoms = np.r_[ntop, nbottom]

        # if no isotopical layer, remove C at the locations of the defect
        if thickness == 0:
            atoms = pc.defect(diamond.cell, atoms, remove=[('C', [0., 0, 0]),
                                                           ('C', [0.5, 0.5, 0.5])])

        # initiallize Simulator instance
        calc = pc.Simulator(1, center, alpha=alpha, beta=beta, bath=atoms,
                            r_bath=r_bath,
                            r_dipole=calc_setup['rdipole'],
                            order=calc_setup['order'])
        # compute coherence
        result = calc.cce_coherence(time_space, as_delay=False, **calc_param)
        # for simplicity of further analysis, save actual thickness
        if arguments.param == 'thickness':
            v = v * length * 2

        result = pd.Series(np.abs(result), index=time_space, name=v)
        result.index.name = 'Time (ms)'
        ls.append(result)

    df = pd.DataFrame(ls).T

    # write the calculation parameters into file
    with open(os.path.join(fol, f'pyCCE_{conf}.csv'), 'w') as file:
        calc_setup.pop(arguments.param)
        # first line is the comment containing parameters
        tw = ', '.join(f'{a} = {b}' for a, b in calc_setup.items())
        file.write('# ' + tw + '\n')
        df.to_csv(file)

    etime = time.time()

    print(f'Calculation of {len(arguments.values)} {arguments.param} took '
          f'{etime - stime:.2f} s for configuration {conf}')
