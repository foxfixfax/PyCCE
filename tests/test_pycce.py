import numpy as np
import pycce


def test_compute():
    b = pycce.random_bath('13C', 100, 1000, seed=1)
    c = pycce.Simulator(1, bath=b, order=2,
                        r_bath=50, r_dipole=6, pulses=1, magnetic_field=0,
                        alpha=[0, 1, 0], beta=[1, 0, 0],
                        D=1e7)

    ts = np.linspace(0, 0.2, 11)
    r1 = np.abs(c.compute(ts, method='cce'))
    r2 = np.abs(c.compute(ts, method='gcce'))

    t = np.array([1, 0.9998434, 0.9983664, 0.9941485, 0.9862164, 0.9707977, 0.9449915,
                  0.9078238, 0.8626642, 0.8094291, 0.7379904])

    assert np.isclose(r1, t).all() and np.isclose(r2, t).all()
