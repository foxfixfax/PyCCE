import sys

sys.path.append('/home/onizhuk/midway/codes_development')
import pycce
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    np.random.seed(42055)
    np.set_printoptions(suppress=True, precision=5)

    sic = pycce.bath.BathCell(3.073, 3.073, 10.053, 90, 90, 120, 'deg')
    sic.zdir = [0, 0, 1]

    sic.add_atoms(('Si', [0.00000000, 0.00000000, 0.1880]),
                  ('Si', [0.00000000, 0.00000000, 0.6880]),
                  ('Si', [0.33333333, 0.66666667, 0.4380]),
                  ('Si', [0.66666667, 0.33333333, 0.9380]),
                  ('C', [0.00000000, 0.00000000, 0.0000]),
                  ('C', [0.00000000, 0.00000000, 0.5000]),
                  ('C', [0.33333333, 0.66666667, 0.2500]),
                  ('C', [0.66666667, 0.33333333, 0.7500]))

    print(sic.add_isotopes(('29Si', 0.047), ('13C', 0.011)))

    vsi_cell = [0, 0, 0.1880]
    vc_cell = [0, 0, 0]

    atoms = sic.gen_supercell(100, remove=[('Si', vsi_cell),
                                           ('C', vc_cell)])

    exatoms = pycce.bath.read_qe('./gipaw/pw.in',
                                 './gipaw/gipaw.out')
    M = np.array([[0, 0, -1],
                  [0, -1, 0],
                  [-1, 0, 0]])

    cell = np.array([[20.272033, 0.000000, 0.000000],
                     [0.000000, 26.811944, 0.000000],
                     [0.000000, 0.000000, 27.863846]])

    center = [0.6, 0.5, 0.5]
    exatoms = pycce.bath.transform(
        exatoms, center=center, cell=cell, rotation_matrix=M, style='row', inplace=False)

    # Setting up CCE calculations
    pos = sic.cell_to_cartesian(vsi_cell)
    N = 0
    CCE_order = 2
    r_bath = 40
    r_dipole = 8
    time_space = np.linspace(0, 0.01, 101)

    B = np.array([0, 0, 500])
    calc = pycce.Simulator(1, pos)

    ntype = calc.add_spintype(('13C', 1 / 2, 6.72828),
                              ('29Si', 1 / 2, -5.3188))

    calc.alpha = np.array([0, 0, 1])
    calc.beta = np.array([0, 1, 0])

    nspin = calc.read_bath(atoms, r_bath,
                           external_bath=exatoms)

    calc.generate_graph(r_dipole)

    subclusters = calc.generate_clusters(CCE_order)

    L = calc.compute_coherence(time_space, B, N, as_delay=False)

    plt.plot(time_space, L)
    plt.show()
