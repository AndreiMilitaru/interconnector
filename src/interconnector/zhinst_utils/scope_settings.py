"""
Module to record demodulator time traces from Zurich Instruments lock-in amplifiers.
Code was originally written in the Photonics Laboratory of ETH Zurich by Felix Tebbenjohanns.
"""

import numpy as np
import time


def config_scope_settings(zidrec, dev, samp_rate, T, channels, enables, pwr_two=True, bwlimit=(1, 1)):
    """Configure basic scope settings for Zurich Instruments device.

    Parameters
    ----------
    zidrec : zhinst_demod_recorder
        Demodulation recorder instance
    dev : str
        Device ID
    samp_rate : float
        Desired sampling rate in Hz
    T : float
        Total acquisition time in seconds
    channels : list
        List of input channels [ch0, ch1] to record
    enables : list
        List of booleans [enable_ch0, enable_ch1] to enable/disable channels
    pwr_two : bool, optional
        If True, round number of points to nearest power of 2
    bwlimit : tuple, optional
        Bandwidth limits for channels (ch0_limit, ch1_limit)
    """
    clockbase = zidrec.lock_in.getInt('/{:s}/clockbase'.format(dev))
    samp_rate = clockbase / 2 ** round(np.log2(clockbase / samp_rate))
    if pwr_two:
        T_pts = 2 ** round(np.log2(samp_rate * T))
    else:
        T_pts = round(samp_rate * T)

    # Settings scope
    zidrec.lock_in.setInt('/{:s}/scopes/0/time'.format(dev), int(np.log2(clockbase / samp_rate)))  # 60/2**4 = 3.75 MHz
    zidrec.lock_in.setInt('/{:s}/scopes/0/length'.format(dev), T_pts)

    zidrec.lock_in.setInt('/{:s}/scopes/0/channels/0/inputselect'.format(dev), channels[0])
    zidrec.lock_in.setInt('/{:s}/scopes/0/channels/1/inputselect'.format(dev), channels[1])

    zidrec.lock_in.setInt('/{:s}/scopes/0/channel'.format(dev),
                          np.sum([(idx + 1) * is_enable for idx, is_enable in enumerate(enables)]))

    zidrec.lock_in.setInt('/{:s}/scopes/0/channels/0/bwlimit'.format(dev), bwlimit[0])
    zidrec.lock_in.setInt('/{:s}/scopes/0/channels/1/bwlimit'.format(dev), bwlimit[1])


def config_scope_trigger(zidrec, dev, channel, slope, level, hysteresis=0, holdoff=0, reference=0.5, delay=0):
    """Configure scope trigger settings.

    Parameters
    ----------
    zidrec : zhinst_demod_recorder
        Demodulation recorder instance
    dev : str
        Device ID
    channel : int
        Trigger channel number
    slope : int
        Trigger slope (1=rising, 2=falling)
    level : float
        Trigger level in Volts
    hysteresis : float, optional
        Trigger hysteresis in Volts
    holdoff : float, optional
        Trigger holdoff time in seconds
    reference : float, optional
        Trigger reference position (0-1)
    delay : float, optional
        Trigger delay in seconds
    """
    zidrec.lock_in.setInt('/{:s}/scopes/0/trigchannel'.format(dev), channel)  # 3=trigger in 2
    zidrec.lock_in.setInt('/{:s}/scopes/0/trigslope'.format(dev), slope)  # 1=rise
    zidrec.lock_in.setDouble('/{:s}/scopes/0/triglevel'.format(dev), level)
    zidrec.lock_in.setDouble('/{:s}/scopes/0/trighysteresis/absolute'.format(dev), hysteresis)
    zidrec.lock_in.setDouble('/{:s}/scopes/0/trigholdoff'.format(dev), holdoff)
    zidrec.lock_in.setDouble('/{:s}/scopes/0/trigreference'.format(dev), reference)
    zidrec.lock_in.setDouble('/{:s}/scopes/0/trigdelay'.format(dev), delay)


def enable_scope_trigger(zidrec, dev, enable):
    """Enable or disable scope triggering.

    Parameters
    ----------
    zidrec : zhinst_demod_recorder
        Demodulation recorder instance
    dev : str
        Device ID
    enable : bool
        True to enable triggering, False to disable
    """
    zidrec.lock_in.setInt('/{:s}/scopes/0/trigenable'.format(dev), enable)


def config_scope_module(zidrec, mode, averages=1, history=0):
    """Configure scope module settings.

    Parameters
    ----------
    zidrec : zhinst_demod_recorder
        Demodulation recorder instance
    mode : int
        Scope data processing mode:
        0 = Pass through
        1 = Moving average
        3 = FFT of every segment
    averages : int, optional
        Number of averages (1 = no averaging)
    history : int, optional
        Number of records to keep in memory
    """
    zidrec.scope = zidrec.lock_in.scopeModule()
    zidrec.scope.set('mode', mode)  # Time mode
    zidrec.scope.set("averager/weight", averages)  # no averages
    zidrec.scope.set('averager/restart', 1)
    zidrec.scope.set('historylength', history)
    zidrec.scope.set('fft/power', 1)  # Time mode
    zidrec.scope.set('fft/spectraldensity', 1)  # Time mode
    zidrec.scope.set('fft/window', 1)  # 1=Hann


def get_data_scope(zidrec, dev, num_records=1, timeout=300, verbose=False, disable_when_done=False):
    """Acquire data from scope.

    Parameters
    ----------
    zidrec : zhinst_demod_recorder
        Demodulation recorder instance
    dev : str
        Device ID
    num_records : int, optional
        Number of records to acquire
    timeout : float, optional
        Maximum wait time in seconds
    verbose : bool, optional
        If True, print progress information

    Returns
    -------
    dict
        Dictionary containing acquired scope data
    """
    zidrec.scope = zidrec.lock_in.scopeModule()  # Initialize scope module if not already done
    zidrec.scope.set('averager/restart', 1)
    zidrec.scope.subscribe('/{:s}/scopes/0/wave'.format(dev))
    # get_scope_records
    zidrec.scope.execute()
    zidrec.lock_in.setInt('/{:s}/scopes/0/enable'.format(dev), 1)
    zidrec.lock_in.sync()
    start = time.time()
    records = 0
    progress = 0
    while (records < num_records) or (progress < 1.0):
        time.sleep(0.1)
        records = zidrec.scope.getInt('records')
        progress = zidrec.scope.progress()[0]
        if verbose:
            print(
                f"Scope module has acquired {records} records (requested {num_records}). "
                f"Progress of current segment {100.0 * progress}%.",
                end="\r",
            )
    if disable_when_done:
        zidrec.lock_in.setInt('/{:s}/scopes/0/enable'.format(dev), 0)
        
    data = zidrec.scope.read(True)
    zidrec.scope.finish()
    return data
