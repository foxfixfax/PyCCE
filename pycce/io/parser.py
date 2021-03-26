import argparse
parser = argparse.ArgumentParser()

parser.add_argument("param", default=None, nargs='?')
parser.add_argument("values", nargs='*', type=floatint)

parser.add_argument("--r_bath", "-rb", default=60, type=floatint,
                      help='cutoff bath radius')
parser.add_argument("--r_dipole", "-rd", default=8, type=floatint,
                      help='pair cutoff radius')
parser.add_argument("--order", "-o", default=2, type=int,
                      help='CCE order')
parser.add_argument("--type", "-t", default=1, type=str)
parser.add_argument("-n", "-N", default=1, type=int)
parser.add_argument("--start", "-s", default=0, type=int)
