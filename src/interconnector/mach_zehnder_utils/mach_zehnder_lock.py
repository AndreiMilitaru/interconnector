"""
author: Andrei Militaru
organization: Institute of Science and Technology Austria (ISTA)
date: October 2025
Description: Utilities for locking a Mach-Zehnder interferometer using Zurich Instruments devices.
I will be adding utilities as time passes based on what we need.
"""

import numpy as np

# Default parameters for the piezo and laser locks as of 3rd October 2025
default_piezo_parameters = [0, 1e3, 0]
default_laser_parametrs = [-100e-3, -15e-3, 0]


def df2tc(freq):
    """
    Convert demodulator bandwidth to time constant for Zurich Instruments lock-in.
    
    Args:
        freq (float): Bandwidth frequency in Hz
    
    Returns:
        float: Time constant in seconds
    """
    return 1/(2*np.pi*float(freq))


def set_demodulators(mdrec, dev='dev30794', oscillator=0, demodulator=1, order=1, rate=53.57e3, bandwidth=20e3):
    """Configure demodulator settings for the Zurich Instruments lock-in amplifier."""
    mdrec.lock_in.set(f'/{dev}/oscs/{oscillator}/freq', 0)
    mdrec.lock_in.set(f'/{dev}/demods/{demodulator}/oscselect', oscillator)
    mdrec.lock_in.set(f'/{dev}/demods/{demodulator}/adcselect', 0)
    mdrec.lock_in.set(f'/{dev}/demods/{demodulator}/order', order)
    mdrec.lock_in.setDouble(f'/{dev}/demods/{demodulator}/timeconstant', df2tc(bandwidth))
    mdrec.lock_in.set(f'/{dev}/demods/{demodulator}/rate', rate)
    mdrec.lock_in.set(f'/{dev}/demods/{demodulator}/enable', 1)
    mdrec.set_demod_list({'signal': (dev, 0)})


def set_aux_limits(mdrec, dev='dev30794', aux_lim=None, laser_lim=None):
    """
    Set the limits for auxiliary outputs controlling piezo and laser.
    
    Args:
        mdrec: Demodulation recorder instance
        dev (str): Device ID
        aux_lim (list): [min, max] voltage limits for piezo auxiliary output
        laser_lim (list): [min, max] voltage limits for laser auxiliary output
    """
    if aux_lim is None:
        aux_lim = [0, 5]
    if laser_lim is None:
        laser_lim = [-0.1, 0.1]
    mdrec.lock_in.set(f'/{dev}/auxouts/0/limitlower', aux_lim[0])
    mdrec.lock_in.set(f'/{dev}/auxouts/0/limitupper', aux_lim[1])
    mdrec.lock_in.set(f'/{dev}/auxouts/0/offset', (aux_lim[0] + aux_lim[1])/2)
    mdrec.lock_in.set(f'/{dev}/auxouts/3/limitlower', laser_lim[0])
    mdrec.lock_in.set(f'/{dev}/auxouts/3/limitupper', laser_lim[1])
    mdrec.lock_in.set(f'/{dev}/auxouts/3/offset', 0)


def toggle_locks(mdrec, enable, dev='dev30794'):
    """
    Enable or disable the PID controllers for piezo and laser locks.
    
    Args:
        mdrec: Demodulation recorder instance
        enable (int): 1 or True to enable, 0 or False to disable
        dev (str): Device ID
    """
    if type(enable) is bool:
        enable = int(enable)
    mdrec.lock_in.setInt(f'/{dev}/pids/0/enable', enable)
    mdrec.lock_in.setInt(f'/{dev}/pids/3/enable', enable)


def check_channel(mdrec, dev: str, pid_num: int, aux_num: int, tolerance_percent: float = 85) -> None:
    """Check if a specific channel's lock offset is within acceptable range and reset if needed.
    
    Args:
        mdrec: Demodulation recorder instance
        dev (str): Device ID
        pid_num (int): PID controller number
        aux_num (int): Auxiliary output number
        tolerance_percent (float): Percentage of range to consider as valid
    """
    center = mdrec.lock_in.getDouble(f'/{dev}/pids/{pid_num}/center')
    lower = mdrec.lock_in.getDouble(f'/{dev}/pids/{pid_num}/limitlower')
    upper = mdrec.lock_in.getDouble(f'/{dev}/pids/{pid_num}/limitupper')
    total_range = upper - lower
    allowed_deviation = (total_range * tolerance_percent) / 100 / 2

    val = mdrec.lock_in.getDouble(f'/{dev}/auxouts/{aux_num}/offset')
    if not (center - allowed_deviation) < val < (center + allowed_deviation):
        mdrec.lock_in.setInt(f'/{dev}/pids/{pid_num}/enable', 0)
        mdrec.lock_in.setDouble(f'/{dev}/auxouts/{aux_num}/offset', center)
        mdrec.lock_in.setInt(f'/{dev}/pids/{pid_num}/enable', 1)


def check_locks(mdrec, dev='dev30794', channels=None, tolerance_percent=85,
                piezo_pid=0, piezo_aux=0, laser_pid=3, laser_aux=3):
    """Check if lock offsets are within acceptable ranges and reset if needed.
    
    Args:
        mdrec: Demodulation recorder instance
        dev (str): Device ID
        channels (list): List of channels to check ('piezo', 'laser')
        tolerance_percent (float): Percentage of range to consider as valid
        piezo_pid (int): PID number for piezo channel
        piezo_aux (int): Auxiliary output number for piezo
        laser_pid (int): PID number for laser channel
        laser_aux (int): Auxiliary output number for laser
    """
    if channels is None:
        channels = ['piezo']

    if 'piezo' in channels:
        check_channel(mdrec, dev, piezo_pid, piezo_aux, tolerance_percent)
    if 'laser' in channels:
        check_channel(mdrec, dev, laser_pid, laser_aux, tolerance_percent)


def set_setpoint(mdrec, new_value, dev='dev30794'):
    """
    Set the PID controller setpoints for both piezo and laser channels.
    
    Args:
        mdrec: Demodulation recorder instance
        new_value (float): New setpoint value
        dev (str): Device ID
    """
    mdrec.lock_in.set('/{:s}/pids/0/setpoint', new_value)
    mdrec.lock_in.set('/{:s}/pids/3/setpoint', new_value)


def set_pid_params(mdrec, dev='dev30794', piezo_params=None, laser_params=None, 
                   piezo_pid=0, laser_pid=3, demodulator=1, 
                   piezo_out=0, laser_out=3, piezo_center=2.5, laser_range=0.1):
    """
    Configure PID controller parameters for both piezo and laser channels.
    
    Args:
        mdrec: Demodulation recorder instance
        dev (str): Device ID
        piezo_params (list): [P, I, D] parameters for piezo controller
        laser_params (list): [P, I, D] parameters for laser controller
        piezo_aux (int): Auxiliary output index for piezo
        laser_aux (int): Auxiliary output index for laser
        demodulator (int): Demodulator index
        piezo_out (int): Piezo output channel
        laser_out (int): Laser output channel
        piezo_center (float): Center voltage for piezo
        laser_range (float): Voltage range for laser
    """
    if piezo_params is None:
        piezo_params = default_piezo_parameters
    if laser_params is None:
        laser_params = default_laser_parametrs
    
    mdrec.lock_in.set(f'/{dev}/pids/0/p', float(piezo_params[0]))
    mdrec.lock_in.set(f'/{dev}/pids/0/i', float(piezo_params[1]))
    mdrec.lock_in.set(f'/{dev}/pids/3/p', float(laser_params[0]))
    mdrec.lock_in.set(f'/{dev}/pids/3/i', float(laser_params[1]))

    mdrec.lock_in.set(f'/{dev}/pids/{piezo_pid}/input', 1)
    mdrec.lock_in.set(f'/{dev}/pids/{piezo_pid}/inputchannel', demodulator)
    mdrec.lock_in.set(f'/{dev}/pids/{piezo_pid}/output', 5)
    mdrec.lock_in.set(f'/{dev}/pids/{piezo_pid}/outputchannel', piezo_out)
    mdrec.lock_in.set(f'/{dev}/pids/{piezo_pid}/center', piezo_center)
    mdrec.lock_in.set(f'/{dev}/pids/{piezo_pid}/limitlower', -piezo_center)
    mdrec.lock_in.set(f'/{dev}/pids/{piezo_pid}/limitupper', piezo_center)

    mdrec.lock_in.set(f'/{dev}/pids/{laser_pid}/input', 1)
    mdrec.lock_in.set(f'/{dev}/pids/{laser_pid}/inputchannel', demodulator)
    mdrec.lock_in.set(f'/{dev}/pids/{laser_pid}/output', 5)
    mdrec.lock_in.set(f'/{dev}/pids/{laser_pid}/outputchannel', laser_out)
    mdrec.lock_in.set(f'/{dev}/pids/{laser_pid}/center', 0)
    mdrec.lock_in.set(f'/{dev}/pids/{laser_pid}/limitlower', -laser_range)
    mdrec.lock_in.set(f'/{dev}/pids/{laser_pid}/limitupper', laser_range)