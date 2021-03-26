import numpy as np
import os
import argparse


def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    try:
        os.mkdir(dir)

    except FileExistsError:
        pass

    return


def floatint(string):
    vfloat = float(string)
    if vfloat.is_integer():
        return int(vfloat)
    else:
        return vfloat


def folname(calc_setup):
    name = 'thickness_{thickness}_rbath_{r_bath}_rdipole_{r_dipole}_order_{order}'.format(**calc_setup)
    return name


calc_setup = {'r_bath': 80, 'r_dipole': 8, 'order': 2, 'thickness': 1}
calc_param = {'B': np.array([0., 0., 500.]), 'N': 1}


pcparser = argparse.ArgumentParser()
pcparser.add_argument("param", default=None, nargs='?')
pcparser.add_argument("values", nargs='*', type=floatint)

pcparser.add_argument("--rbath", "-rb", default=60, type=floatint,
                      help='cutoff bath radius')
pcparser.add_argument("--rdipole", "-rd", default=8, type=floatint,
                      help='pair cutoff radius')
pcparser.add_argument("--order", "-o", default=2, type=int,
                      help='CCE order')
pcparser.add_argument("--thickness", "-t", default=1, type=floatint)
pcparser.add_argument("--start", "-s", default=0, type=int)

if __name__ == '__main__':

    stuff = pcparser.parse_args()
    print(stuff)
