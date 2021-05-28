import numpy as np

MHZ_TO_RADKHZ = 2 * np.pi * 1000
BOHR_TO_ANGSTROM = 5.29177E-01

HARTREE_TO_MHZ = 6579680000.0
M_TO_BOHR = 18897300000.0

ELECTRON_GYRO = -17608.597050  # rad / (ms * Gauss) or rad * kHz / G
HBAR = 1.05457172  # When everything else in rad, kHz, ms, G, A

COMPLEX_DTYPE = np.complex128

BARN_TO_BOHR2 = M_TO_BOHR ** 2 * 1E-28
EFG_CONVERSION = BARN_TO_BOHR2 * HARTREE_TO_MHZ * MHZ_TO_RADKHZ  # units to convert EFG