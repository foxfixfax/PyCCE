import numpy as np
import pycce as pc
from ase.build import bulk

cell = pc.BathCell.from_ase(bulk('C', 'diamond', cubic=True))
atoms = cell.gen_supercell(200, remove=('C', [0, 0, 0]))

calc = pc.Simulator(1, position=[0, 0, 0], bath=atoms, r_bath=40,
                    r_dipole=6, order=2, magnetic_field=500, pulses=1)

time_points = np.linspace(0, 2, 101)
coherence = calc.compute(time_points)
