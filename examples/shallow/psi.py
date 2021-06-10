import numpy as np
from mpi4py import MPI
import os
import time
import sys
from parser import pcparser
from scipy.spatial.transform import Rotation

import pandas as pd

from ase import io

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

# PHYSICAL REVIEW B 68, 115322 (2003)
n = 0.81
a = 25.09


def factor(x, y, z, n=0.81, a=25.09, b=14.43):
    top = np.exp(-np.sqrt(x**2/(n*b)**2 + (y**2 + z**2)/(n*a)**2))
    bottom = np.sqrt(np.pi * (n * a)**2 * (n * b) )

    return top / bottom


def contact_si(r, gamma_n, gamma_e=pc.ELECTRON_GYRO, a_lattice=5.43, nu=186, n=0.81, a=25.09, b=14.43):
    k0 = 0.85 * 2 * np.pi / a_lattice
    pre = 16 * np.pi / 9 * gamma_n * gamma_e * pc.HBAR * nu
    xpart = factor(r[0], r[1], r[2], n=n, a=a, b=b) * np.cos(k0 * r[0])
    ypart = factor(r[1], r[2], r[0], n=n, a=a, b=b) * np.cos(k0 * r[1])
    zpart = factor(r[2], r[0], r[1], n=n, a=a, b=b) * np.cos(k0 * r[2])
    return pre * (xpart + ypart + zpart) ** 2


def gen_hf(atoms, position):

    # Generate hyperfine from point dipole
    atoms.from_point_dipole(position)

    # Following PRB paper
    atoms['A'][atoms.dist() < n*a] = 0
    atoms['A'] += np.eye(3)[np.newaxis,:,:] * contact_si(atoms['xyz'].T, atoms['29Si'].gyro)[:,np.newaxis, np.newaxis]

    return atoms

seed = 1

# position of central spin
center = np.array([0, 0, 0])
axis = np.array([1, -1, 0])
axis = axis / np.linalg.norm(axis)


if __name__ == '__main__':
    arguments = pcparser.parse_args()

    maxtime = 1
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
    # Generate unitcell from ase+
    s = pc.bath.BathCell.from_ase(io.read('si.cif'))
    # Add types of isotopes
    s.add_isotopes(('29Si', 0.047))
    # set z direction of the defect
    s.zdir = [0, 0, 1]

    # Generate supercell
    atoms = s.gen_supercell(200, remove=[('Si', center)], seed=seed + conf)
    # transform parser into dictionary
    calc_setup = vars(arguments)

    # list to store calculations results
    ls = []
    # argument.values contains values of the varied parameter
    # arguments.param
    for v in arguments.values:
        calc_setup[arguments.param] = v
        # initiallize Simulator instance
        bath = atoms.copy()
        ran = calc_setup['angle'] / 360 * 2 * np.pi
        r = Rotation.from_quat([*(axis * np.sin(ran / 2)), np.cos(ran / 2)])
        bath.xyz = r.apply(bath.xyz)

        gen_hf(bath, center)

        calc = pc.Simulator(0.5, center,
                            bath=bath,
                            r_bath=calc_setup['r_bath'],
                            magnetic_field=calc_setup['magnetic_field'],
                            pulses=calc_setup['pulses'],
                            r_dipole=calc_setup['r_dipole'],
                            order=calc_setup['order'])
        # compute coherence
        result = calc.cce_coherence(time_space, as_delay=False)
        # for simplicity of further analysis, save actual thickness
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
        with open(f'si_{arguments.param}.csv', 'w') as file:
            tw = ', '.join(f'{a} = {b}' for a, b in calc_setup.items())
            file.write('# ' + tw + '\n')
            df.to_csv(file)
