import matplotlib.pyplot as plt
import numpy as np
import pycce
from ase.build import bulk

if __name__ == '__main__':
    np.random.seed(42055)
    np.set_printoptions(suppress=True, precision=5)

    # Generate unitcell from ase
    diamond = bulk('C', 'diamond', orthorhombic=True)
    diamond = pycce.bath.NSpinCell.from_ase_Atoms(diamond)
    # Add types of common_isotopes
    diamond.add_isotopes(('13C', 0.011))
    # set z direction of the defect
    diamond.zdir = [1, 1, 1]
    # Generate supercell

    atoms = diamond.gen_supercell(200, remove=[('C', [0., 0, 0]),
                                               ('C', [0.5, 0.5, 0.5])],
                                  add=('14N', [0.5, 0.5, 0.5]))

    # Parameters of CCE calculations
    N = 1  # Number of pulses
    CCE_order = 2
    time_space = np.linspace(0, 2, 201)  # in ms
    r_bath = 40  # in A
    r_dipole = 8  # in A
    B = np.array([0, 0, 500])  # in G
    # Setting the runner engine
    calc = pycce.Simulator(1, [0, 0, 0])
    # Parameters of nuclear spins
    #                      name   spin   gyromagnetic ratio rad/G/ms
    ntype = calc.add_spintype(('13C', 1 / 2, 6.72828),
                              ('29Si', 1 / 2, -5.3188),
                              ('14N', 1, 1.9338, 20.44))
    # Qubit levels
    calc.alpha = np.array([0, 0, 1])
    calc.beta = np.array([0, 1, 0])
    # Read bath and generate clusters
    nspin = calc.read_bath(atoms, r_bath)
    # Set model EFG at N atom
    nspin['Q'][nspin['N'] == '14N'] = np.asarray([[-2.5, 0, 0],
                                                  [0, -2.5, 0],
                                                  [0, 0, 5.0]]) * 1e3 * 2 * np.pi

    calc.generate_graph(r_dipole)
    subclusters = calc.generate_clusters(CCE_order)
    # Compute coherence function
    L = calc.compute_coherence(time_space, B, N, as_delay=False)

    plt.plot(time_space, L)
    plt.show()
