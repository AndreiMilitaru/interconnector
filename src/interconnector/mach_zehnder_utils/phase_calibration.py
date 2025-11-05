"""
author: Andrei Militaru
organization: Institute of Science and Technology Austria (ISTA)
date: October 2025
Description: Utilities for calibrating the phase signal at the Mach Zehnder output. 
"""

import numpy as np
from scipy.optimize import curve_fit
from .mach_zehnder_lock import df2tc, toggle_locks

detector_offset = 20e-3  # Volts

def drive_phase(mdrec, dev='dev30794', drive_demodulator=1, drive_oscillator=1, drive_freq=100, 
                drive_amp=1, trace_duration=1, reset_pids=False, rate=53.57e3):
    """
    Drive the phase of the interferometer and collect demodulated timetraces.
    
    Args:
        mdrec: Demodulation recorder instance
        dev (str): Device ID
        drive_demodulator (int): Demodulator index for phase drive
        drive_oscillator (int): Oscillator index for phase drive
        drive_freq (float): Drive frequency in Hz
        drive_amp (float): Drive amplitude
        trace_duration (float): Duration of the measurement in seconds
        reset_pids (bool): Whether to re-enable PIDs after measurement
    
    Returns:
        numpy.ndarray: Imaginary part of the demodulated signal
    """

    # Turn off the PID temporarily
    toggle_locks(mdrec, False, dev=dev)

    # Set the drive parameters
    mdrec.lock_in.set('/{:s}/oscs/{:d}/freq'.format(dev, drive_oscillator), drive_freq)
    mdrec.lock_in.set('/{:s}/demods/{:d}/oscselect'.format(dev, drive_demodulator), drive_oscillator)
    mdrec.lock_in.set('/{:s}/demods/{:d}/adcselect'.format(dev, drive_demodulator), 174)
    mdrec.lock_in.set('/{:s}/demods/{:d}/timeconstant'.format(dev, drive_demodulator), df2tc(drive_freq*100))
    mdrec.lock_in.set('/{:s}/demods/{:d}/rate'.format(dev, drive_demodulator), rate)

    # Activating the phase drive
    mdrec.lock_in.set('/{:s}/auxouts/0/offset'.format(dev), 2.5)
    mdrec.lock_in.set('/{:s}/auxouts/0/demodselect'.format(dev), drive_demodulator)
    mdrec.lock_in.set('/{:s}/auxouts/0/outputselect'.format(dev), 0)
    mdrec.lock_in.set('/{:s}/auxouts/0/preoffset'.format(dev), 0.)
    mdrec.lock_in.set('/{:s}/auxouts/0/scale'.format(dev), drive_amp)

    # Collecting demodulated timetrace
    dat = mdrec.record_timtrace(T=trace_duration)

    # Turning off the phase drive
    mdrec.lock_in.set('/{:s}/auxouts/0/outputselect'.format(dev), -1)
    mdrec.lock_in.set('/{:s}/auxouts/0/offset'.format(dev), 2.5)

    # Resetting the PID if needed
    if reset_pids:
        toggle_locks(mdrec, True, dev=dev)

    return np.imag(dat['signal']['trace'])


def calibrate_range(mdrec, dev='dev30794', **kwargs):
    """
    Calibrate the interferometer phase range by analyzing the phase distribution.
    
    Args:
        mdrec: Demodulation recorder instance
        dev (str): Device ID
        **kwargs: Additional arguments passed to drive_phase
    
    Returns:
        tuple: (fit parameters, covariance matrix, histogram, bin edges)
    """
    trace = drive_phase(mdrec, dev=dev, **kwargs)
    vmin, vmax = np.min(trace), np.max(trace)
    hist, bin_edges = np.histogram(trace, bins=200, density=True, range=(vmin, vmax))
    guess = [np.min(hist[1:-2])*(vmax-vmin)/2, vmin, vmax]
    par, cov = curve_fit(unlock_model, bin_edges[1:-3], hist[1:-2], guess)
    return par, cov, hist, bin_edges


def evaluate_visibility(par):
    """
    Calculate the interferometer visibility from calibration parameters.
    
    Args:
        par (list): Calibration parameters [offset, Vmin, Vmax]
    
    Returns:
        float: Visibility of the interference pattern
    """
    vmin = par[1]
    vmax = par[2]
    return (vmax - vmin) / (vmax + vmin - 2*detector_offset)


def evaluate_lock_precision(mdrec, par, dev='dev30794', duration=10):
    """
    Evaluate the precision of the phase lock by analyzing phase fluctuations.
    
    Args:
        mdrec: Demodulation recorder instance
        dev (str): Device ID
        duration (float): Measurement duration in seconds
        par (list): Calibration parameters
    
    Returns:
        tuple: (lock parameters, covariance matrix, histogram, bin edges)
    """
    dat = mdrec.record_timtrace(T=duration)
    trace = np.imag(dat['signal']['trace'])
    hist, bin_edges = np.histogram(trace, bins=200, density=True, range=(par[1], par[2]))
    phi, fphi = convert(bin_edges[:-1], hist, par)
    fphi_max = np.max(fphi)
    mu_guess = phi[np.argmax(fphi)]
    sig2_guess = (mu_guess - phi(np.where(fphi > fphi_max/np.exp(-1/2))[0][0]))**2
    guess = [mu_guess, sig2_guess]
    par_lock, cov_lock = curve_fit(lock_model, phi, fphi, guess)
    return par_lock, cov_lock, hist, bin_edges


def V2phi(V, par):
    """
    Convert voltage to phase using calibration parameters.
    
    Args:
        V (float or numpy.ndarray): Voltage values
        par (list): Calibration parameters [offset, Vmin, Vmax]
    
    Returns:
        float or numpy.ndarray: Phase values in radians
    """
    V0 = par[1]
    V1 = par[2]
    return np.arcsin( 2*(V-V0)/(V1-V0) - 1 ) + np.pi/2


def correction(V, par):
    """
    Calculate correction factor for probability density transformation.
    
    Args:
        V (float or numpy.ndarray): Voltage values
        par (list): Calibration parameters [offset, Vmin, Vmax]
    
    Returns:
        float or numpy.ndarray: Correction factors
    """
    V0 = par[1]
    V1 = par[2]
    return np.sqrt(-(V-V0)**2 + (V-V0)*(V1-V0))


def convert(V, fV, par):
    """
    Convert voltage probability distribution to phase probability distribution.
    
    Args:
        V (numpy.ndarray): Voltage values
        fV (numpy.ndarray): Voltage probability distribution
        par (list): Calibration parameters
    
    Returns:
        tuple: (phase values, phase probability distribution)
    """
    print(len(V), len(fV))
    return V2phi(V, par), fV*correction(V, par)


def lock_model(x, mu, sig2):
    """
    Gaussian model for locked phase distribution.
    
    Args:
        x (numpy.ndarray): Phase values
        mu (float): Mean phase
        sig2 (float): Phase variance
    
    Returns:
        numpy.ndarray: Probability density values
    """
    return 1/np.sqrt(2*np.pi*sig2) * np.exp(- (x-mu)**2/(2*sig2))


def unlock_model(x, A, x0, x1):
    """
    Model for unlocked phase distribution.
    
    Args:
        x (numpy.ndarray): Phase values
        A (float): Amplitude parameter
        x0 (float): Lower bound
        x1 (float): Upper bound
    
    Returns:
        numpy.ndarray: Probability density values
    """
    return A/np.sqrt( (x-x0)*(x1-x) )
