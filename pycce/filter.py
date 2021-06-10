"""
Module with helper functions to obtain CPMG coherence from the noise autocorrelation function.
"""
import numba
import numpy as np
import scipy.integrate
from numba import cfunc, carray, jit
from numba.types import intc, CPointer, float64, int32
from scipy import LowLevelCallable
from .constants import PI2

def _jit_integrand_function(integrand_function):
    jitted_function = numba.jit(integrand_function, nopython=True)

    @cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):
        values = carray(xx, n)
        return jitted_function(values)

    return LowLevelCallable(wrapped.ctypes)


@cfunc(float64(float64, float64, float64))
def _yfunc(x, tau, npulses):
    delay = tau / 2 / npulses
    pulses_passed = (x // delay + 1) // 2

    if pulses_passed % 2 == 0:
        return 1
    else:
        return -1


@_jit_integrand_function
def _integrand(args):
    v, u, tau, np = args[0], args[1], args[2], args[3]
    return _yfunc((v + u) / 2, tau, np) * _yfunc((v - u) / 2, tau, np)


def filterfunc(ts, tau, npulses):
    """
    Time-domain filter function for the given CPMG sequence.

    Args:
        ts (ndarray with shape (n,)): Time points at which filter function will be computed.
        tau (float): Delay between pulses.
        npulses (int): Number of pulses in CPMG sequence.

    Returns:
        ndarray with shape (n,): Filter function for the given CPMG sequence
    """
    fs = np.empty(ts.shape)
    ks = np.empty(npulses) * 2
    ks[:npulses] = np.arange(npulses)
    ks[npulses:] = np.arange(npulses)

    for i, u in enumerate(ts):
        bad_points = (2 * ks + 1) * tau / npulses
        bad_points[:npulses] += u
        bad_points[npulses:] -= u
        bad_points = np.unique(bad_points[(bad_points > u) & (bad_points < 2 * tau - u)])

        if bad_points.size == 0:
            fs[i] = scipy.integrate.quad(_integrand, u, 2 * tau - u, args=(u, tau, npulses))[0]

        else:
            fs[i] = scipy.integrate.quad(_integrand, u, bad_points[0], args=(u, tau, npulses))[0]

            for j in range(1, bad_points.size):
                fs[i] += scipy.integrate.quad(_integrand, bad_points[j - 1], bad_points[j],
                                              args=(u, tau, npulses))[0]

            fs[i] += scipy.integrate.quad(_integrand, bad_points[-1], 2 * tau - u, args=(u, tau, npulses))[0]

    return fs


def gaussian_phase(timespace, corr, npulses, units='khz'):
    """
    Compute average random phase squared assuming Gaussian noise.

    Args:
        timespace (ndarray with shape (n,)): Time points at which correlation function was computed.
        corr (ndarray with shape (n,)): Noise autocorrelation function.
        npulses (int): Number of pulses in CPMG sequence.
        units (str): If units contain frequency or angular frequency ('rad' in ``units``).
    Returns:
        ndarray with shape (n,): Random phase accumulated by the qubit.
    """
    if 'rad' not in units:
        corr *= PI2**2
    timespace = np.asarray(timespace)
    chis = np.zeros(timespace.shape)
    for i, tau in enumerate(timespace):
        if tau == 0:
            chis[i] = 0
        else:
            chis[i] = np.trapz(corr[timespace <= tau] * filterfunc(timespace[timespace <= tau], tau, npulses),
                               timespace[timespace <= tau]) / 2
    return chis
