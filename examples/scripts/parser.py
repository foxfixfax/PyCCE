import argparse


def floatint(string):
    vfloat = float(string)
    if vfloat.is_integer():
        return int(vfloat)
    else:
        return vfloat


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
