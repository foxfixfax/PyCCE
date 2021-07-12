import argparse


def floatint(string):
    vfloat = float(string)
    if vfloat.is_integer():
        return int(vfloat)
    else:
        return vfloat


pcparser = argparse.ArgumentParser()
pcparser.add_argument("param", default=None, nargs='?', help='varied parameter')
pcparser.add_argument("values", nargs='*', type=floatint, help='values of varied parameter')

pcparser.add_argument("--r_bath", "-rb", default=50, type=floatint,
                      help='cutoff bath radius')
pcparser.add_argument("--r_dipole", "-rd", default=8, type=floatint,
                      help='pair cutoff radius')
pcparser.add_argument("--order", "-o", default=2, type=int,
                      help='CCE order')
pcparser.add_argument("--start", "-s", default=0, type=int, help='configurations start')
pcparser.add_argument("--pulses", "-N", default=1, type=int, help='number of pulses')
pcparser.add_argument("--thickness", "-t", default=1, type=floatint, help='thickness of inner layer in unit cells')

if __name__ == '__main__':

    stuff = pcparser.parse_args()
    print(stuff)
