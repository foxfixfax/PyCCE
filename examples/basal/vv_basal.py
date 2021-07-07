import numpy as np
import os
import time
import sys
from mpi4py import MPI

from parser import pcparser

import pandas as pd
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
calc_param = {'magnetic_field': np.array([0., 0., 500.]), 'pulses': 1}

# ZFS parameters of basal divacancy
MHZ_KHZ = 1e3
GHZ_KHZ = 1e6 

D = 1.334 * GHZ_KHZ
E = 0.0184 * GHZ_KHZ

if __name__ == '__main__':
    # check how long the calculations was
    stime = time.time()
    # parse argument for Simulator instance
    # with console parser (defined in parser.py)
    arguments = pcparser.parse_args()
    maxtime = 2.5
    time_space = np.linspace(0, maxtime, 501)

    conf = arguments.start

    # MPI stuff
    comm = MPI.COMM_WORLD

    size = comm.Get_size()
    rank = comm.Get_rank()

    # Set up BathCell
    # Set up unit cell with (a, b, c, alpha, beta, gamma)
    sic = pc.bath.BathCell(3.073, 3.073, 10.053, 90, 90, 120, 'deg')
    # z axis in cell coordinates
    sic.zdir = [0, 0, 1]
    # Add types of isotopes

    # position of atoms

    sic.add_atoms(('Si', [0.00000000, 0.00000000, 0.1880]),
                  ('Si', [0.00000000, 0.00000000, 0.6880]),
                  ('Si', [0.33333333, 0.66666667, 0.4380]),
                  ('Si', [0.66666667, 0.33333333, 0.9380]),
                  ('C',  [0.00000000, 0.00000000, 0.0000]),
                  ('C',  [0.00000000, 0.00000000, 0.5000]),
                  ('C',  [0.33333333, 0.66666667, 0.2500]),
                  ('C',  [0.66666667, 0.33333333, 0.7500]))

    # isotopes
    sic.add_isotopes(('29Si', 0.047), ('13C', 0.011))

    vsi_cell = -np.array([1 / 3, 2 / 3, 0.0620])
    vc_cell = np.array([0, 0, 0])

    sic.zdir = [0, 0, 1]

    # Rotation matrix for DFT supercell
    R = pc.rotmatrix([0, 0, 1], sic.to_cartesian(vsi_cell - vc_cell))

    sic.zdir = vsi_cell - vc_cell

    # Generate bath spin positions
    sic.add_isotopes(('29Si', 0.047), ('13C', 0.011))
    atoms = sic.gen_supercell(200, remove=[('Si', vsi_cell),
                                           ('C', vc_cell)],
                              seed=seed + conf)

    # Prepare rotation matrix to alling with z axis of generated atoms
    M = np.array([[0, 0, -1],
                  [0, -1, 0],
                  [-1, 0, 0]])

    # Position of (0,0,0) point in cell coordinates
    center = np.array([0.59401, 0.50000, 0.50000])

    # Read GIPAW results
    exatoms = pc.read_qe('./pw.in',
                         hyperfine='./gipaw.out',
                         center=center, rotation_matrix=(M.T @ R),
                         rm_style='col',
                         isotopes={'C': '13C', 'Si': '29Si'})

    atoms.update(exatoms[exatoms.dist(center) < 10])
    atoms = atoms[np.abs(atoms.A[:,2,2]) < 1.1 * MHZ_KHZ]
    calc_setup = vars(arguments)

    # argument.values contains values of the varied parameter
    # arguments.param
    fol = f'{arguments.pulses}_var_{arguments.param}'
    mkdir_p(fol)

    # list to store calculations results
    ls = []

    # argument.values contains values of the varied parameter

    # arguments.param

    for v in arguments.values:
        calc_setup[arguments.param] = v
        # initiallize Simulator instance
        calc = pc.Simulator(1, center, alpha=0, beta=-1, bath=atoms,
                            D=D, E=E,
                            r_bath=calc_setup['r_bath'],
                            r_dipole=calc_setup['r_dipole'],
                            ext_r_bath=10,
                            magnetic_field=calc_setup['magnetic_field'],
                            pulses=calc_setup['pulses'],
                            order=calc_setup['order'])

        # compute coherence
        result = calc.compute(time_space, as_delay=False,
                              quantity='coherence', method='gcce',
                              mean_field=True, parallel_states=True,
                              nbstates=calc_setup['nbstates'],
                              )
        # for simplicity of further analysis, save actual thickness
        ls.append(result)

    if rank == 0:
        ls = np.asarray(ls)

        df = pd.DataFrame(ls.T, columns=arguments.values, index=time_space)
        df.index.rename('Time', inplace=True)

        # write the calculation parameters into file
        with open(os.path.join(fol, f'pyCCE_{conf}.csv'), 'w') as file:
            tw = ', '.join(f'{a} = {b}' for a, b in calc_setup.items())
            file.write('# ' + tw + '\n')
            df.to_csv(file)

        etime = time.time()

        print(f'Calculation of {len(arguments.values)} {arguments.param} took '
              f'{etime - stime:.2f} s for configuration {conf}')
