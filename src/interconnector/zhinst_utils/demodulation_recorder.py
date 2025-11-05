"""
Module to record demodulator time traces from Zurich Instruments lock-in amplifiers.
Code was originally written in the Photonics Laboratory of ETH Zurich by Felix Tebbenjohanns.
"""

import zhinst.ziPython as ziPython
import zhinst.utils as ziutils
import time
import numpy as np
    

class mfli_demod_recorder(object):
    def __init__(self, data_server_ip):
        print('Depreciated! Change class name to \'zhinst_demod_recorder\'.')
        
        
class zhinst_demod_recorder(object):
    """
    Class to handle demodulator recording from Zurich Instruments devices.
    
    Parameters
    ----------
    data_server_ip : str
        IP address of the Zurich Instruments data server
    devtype : str, optional
        Type of device ('MFLI' or 'HF2LI')
    """
    def __init__(self, data_server_ip, devtype = 'MFLI'):
        if devtype == 'MFLI':
            self.lock_in = ziPython.ziDAQServer(data_server_ip, 8004, 5)
        elif devtype == 'HF2LI':
            self.lock_in = ziPython.ziDAQServer(data_server_ip, 8005, 1)
        else:
            print('Device type not known.')
            return 
        self.lock_in.sync()
        self.demod_dict = {}

    def convertFilter_3dBfreq_tc(self, filter_order, filter_tc_or3dBfreq):
        """
        Convert between filter 3dB frequency and time constant.
        
        Parameters
        ----------
        filter_order : int
            Order of the filter
        filter_tc_or3dBfreq : float
            Time constant or 3dB frequency to convert
            
        Returns
        -------
        float
            Converted value
        
        Notes
        -----
        See MFLI UserManual, page 315, table 6.1
        chapter 6, Signal Processing Basics
        """
        # see MFLI UserManual, page 315, table 6.1
        # chapter 6, Signal Processing Basics
        FO = np.array([1, 0.6436, 0.5098, 0.4350, 0.3856, 0.3499, 0.3226, 0.3008])
        return FO[filter_order-1]/2/np.pi/filter_tc_or3dBfreq
    
    
    def config_demods(self, device, demod_index, config_dict):
        """
        Configure demodulator settings.
        
        Parameters
        ----------
        device : str
            Device ID
        demod_index : int
            Index of the demodulator to configure
        config_dict : dict
            Dictionary containing configuration parameters
            Possible keys: 'filter3dB', 'oscselect', 'harmonic', 'order',
            'adcselect', 'enable', 'phaseshift', 'timeconstant', 'rate', 'freq'
        """
        processed_keys = []
        if 'filter3dB' in config_dict: # this needs to be converted to a timeconstant
            processed_keys.append('filter3dB')
            if 'order' in config_dict:
                order = config_dict['order']
            else:
                order = self.lock_in.getInt('/%s/demods/%i/order' % (device, demod_index))
            config_dict['timeconstant'] = self.convertFilter_3dBfreq_tc(order, config_dict['filter3dB'])
            
        #first, worry about the 'Int' types
        for key in ['oscselect', 'harmonic', 'order', 'adcselect', 'enable']:
            processed_keys.append(key)
            if key in config_dict:
                self.lock_in.setInt('/%s/demods/%i/%s' % (device, demod_index, key), config_dict[key])
        # next, the 'Doubles':
        for key in ['phaseshift', 'timeconstant', 'rate']:
            processed_keys.append(key)
            if key in config_dict:
                self.lock_in.setDouble('/%s/demods/%i/%s' % (device, demod_index, key), config_dict[key])
        
        if 'freq' in config_dict:
            processed_keys.append('freq')
            oscselect = self.lock_in.getInt('/%s/demods/%i/oscselect' % (device, demod_index))
            self.lock_in.setDouble('/%s/oscs/%i/freq' % (device, oscselect), config_dict['freq'])
        
        for key in config_dict:
            if not key in processed_keys:
                print('Did not process following setting: %s' % key)
            
        
        
    def get_demod_info(self, device, demod_index):
        """
        Get current demodulator settings.
        
        Parameters
        ----------
        device : str
            Device ID
        demod_index : int
            Index of the demodulator
            
        Returns
        -------
        dict
            Dictionary containing demodulator settings:
            - freq: Oscillator frequency
            - order: Filter order
            - TC: Time constant
            - BW3dB: 3dB bandwidth
            - fs: Sampling rate
            - phaseshift: Phase shift
            - harmonic: Harmonic number
            - input_channel: Input channel number
        """
        oscselect = self.lock_in.getInt('/%s/demods/%i/oscselect' % (device, demod_index))
        freq = self.lock_in.getDouble('/%s/oscs/%i/freq' % (device, oscselect))
        harmonic = self.lock_in.getInt('/%s/demods/%i/harmonic' % (device, oscselect))
        phaseshift = self.lock_in.getDouble('/%s/demods/%i/phaseshift' % (device, oscselect))
        filter_order = self.lock_in.getInt('/%s/demods/%i/order' % (device, demod_index))
        filter_tc = self.lock_in.getDouble('/%s/demods/%i/timeconstant' % (device, demod_index))
        fs = self.lock_in.getDouble('/%s/demods/%i/rate' % (device, demod_index))
        input_channel = self.lock_in.getInt('/%s/demods/%i/adcselect' % (device, demod_index))
        
            # see MFLI UserManual, page 315, table 6.1
            # chapter 6, Signal Processing Basics
        FO = np.array([1, 0.6436, 0.5098, 0.4350, 0.3856, 0.3499, 0.3226, 0.3008]) 
        filter_bw3db = FO[filter_order-1]/2/np.pi/filter_tc
        
        return {'freq': freq,
               'order': filter_order,
               'TC': filter_tc,
               'BW3dB': filter_bw3db,
               'fs': fs,
               'phaseshift' : phaseshift,
               'harmonic': harmonic,
               'input_channel': input_channel}
    
    def set_demod_list(self, demod_dict):
        #demod_dict must be a dictionary of tuples of the form (dev, demod_index)
        #the keys are used as names
        self.demod_dict = demod_dict
    
    def demod_path(self, device, demod_index):
        return '/%s/demods/%d/sample' % (device, demod_index)
    
    def record_timtrace(self, T=0.1, other_tracks_to_store = []):
        # subscribe to the demodulator's node path
        for demod_name in self.demod_dict:
            self.lock_in.subscribe(self.demod_path(self.demod_dict[demod_name][0], 
                                                   self.demod_dict[demod_name][1]))
        
        # Poll the subscribed data from the data server. Poll will block and record
        # for poll_length seconds.
        poll_length = T # [s]
        poll_timeout = 2000  # [ms]
        poll_flags = 0
        poll_return_flat_dict = True

        self.lock_in.sync()
        data = self.lock_in.poll(poll_length, poll_timeout, poll_flags, poll_return_flat_dict) 
        
        # unsubscribe 
        for demod_name in self.demod_dict:
            self.lock_in.unsubscribe(self.demod_path(self.demod_dict[demod_name][0], 
                                                   self.demod_dict[demod_name][1]))
                    
        tmp = {}
        
        
        for demod_name in self.demod_dict:
            if len(self.demod_dict[demod_name]) >= 3:
                other_tracks_to_store = self.demod_dict[demod_name][2]
                # other_tracks_to_store can be 'frequency', 'phase', 'dio' 'trigger', 'auxin0', 'auxin1', 'time'
                
            device = self.demod_dict[demod_name][0]
            demod_index = self.demod_dict[demod_name][1]
            path = self.demod_path(device, demod_index)
            z = (data[path]['x'] +1j * data[path]['y'])
        
            demod_info = self.get_demod_info(device, demod_index)
            t = data[path]['timestamp']
            dt = 1/self.lock_in.getDouble('/%s/clockbase' % device)
        
            tmp[demod_name] = {'T': t, 'dt':dt, 'trace': z, 'demod_info': demod_info}
            for other_track_to_store in other_tracks_to_store:
                tmp[demod_name][other_track_to_store] = data[path][other_track_to_store]

        self.lock_in.sync()
        return tmp
    
    def get_scope_data(self,
                  dev,
                  T=0.1,
                  samp_rate=1.88e6,
                  inputenable=3,
                  inputselect=[0, 1],
                  mode=3,
                  num_averages=10,
                  historylength=2,
                  power=1,
                  spectraldensity=1,
                  pwr_two=True):
        """
        Added by Massiliano Rossi on 03.08.21
        Seetings the scope and acquire data with the ScopeModule.
        
        Parameters
        ----------
        dev : str
            Device ID
        T : float
            Duration for recording [s]
        samp_rate : float
            Desired sampling rate [Hz]
        inputenable : int
            Bit-encoded value to enable scope channels
        inputselect : list of int
            List of input channels to select for the scope
        mode : int
            Scope data processing mode
        num_averages : int
            Number of averages for the moving average
        historylength : int
            Number of scope records to keep in the module's memory
        power : int
            FFT power
        spectraldensity : int
            Spectral density flag
        pwr_two : bool
            Flag to adjust T_pts to power of two
            
        Returns
        -------
        dict
            Dictionary containing scope data
        """
        clockbase = self.lock_in.getInt('/{:s}/clockbase'.format(dev))
        samp_rate = clockbase/2**round(np.log2(clockbase/samp_rate))
        if pwr_two:
            T_pts = 2**round(np.log2(samp_rate*T))
        else:
            T_pts = round(samp_rate*T)
            
        # 'time' : timescale of the wave, sets the sampling rate to clockbase/2**time.
        #   0 - sets the sampling rate to 1.8 GHz
        #   1 - sets the sampling rate to 900 MHz
        #   ...
        #   16 - sets the samptling rate to 27.5 kHz
        self.lock_in.setInt('/{:s}/scopes/0/time'.format(dev), int(np.log2(clockbase/samp_rate))) # 60/2**4 MHz
        # Configure the instrument's scope via the /devx/scopes/0/ node tree branch.
        # 'length' : the length of each segment
        self.lock_in.setInt('/{:s}/scopes/0/length'.format(dev), T_pts) # Pts
        # 'channel' : select the scope channel(s) to enable.
        #  Bit-encoded as following:
        #   1 - enable scope channel 0
        #   2 - enable scope channel 1
        #   3 - enable both scope channels (requires DIG option)
        self.lock_in.setInt('/{:s}/scopes/0/channel'.format(dev), inputenable)
        # 'channels/0/bwlimit' : bandwidth limit the scope data. Enabling bandwidth
        # limiting avoids antialiasing effects due to subsampling when the scope
        # sample rate is less than the input channel's sample rate.
        #  Bool:
        #   0 - do not bandwidth limit
        #   1 - bandwidth limit
        self.lock_in.setInt('/{:s}/scopes/0/channels/0/bwlimit'.format(dev), 1)
        self.lock_in.setInt('/{:s}/scopes/0/channels/1/bwlimit'.format(dev), 1)
        # 'channels/0/inputselect' : the input channel for the scope:
        #   0 - signal input 1
        #   1 - signal input 2
        #   2, 3 - trigger 1, 2 (front)
        #   8-9 - auxiliary inputs 1-2
        #   The following inputs are additionally available with the DIG option:
        #   10-11 - oscillator phase from demodulator 3-7
        #   16-23 - demodulator 0-7 x value
        #   32-39 - demodulator 0-7 y value
        #   48-55 - demodulator 0-7 R value
        #   64-71 - demodulator 0-7 Phi value
        #   80-83 - pid 0-3 out value
        #   96-97 - boxcar 0-1
        #   112-113 - cartesian arithmetic unit 0-1
        #   128-129 - polar arithmetic unit 0-1
        #   144-147 - pid 0-3 shift value
        for index, which_input in enumerate(inputselect):
            self.lock_in.setInt('/{:s}/scopes/0/channels/{:d}/inputselect'.format(dev, index), which_input)
        # 'single' : only get a single scope record.
        #   0 - acquire continuous records
        #   1 - acquire a single record
        self.lock_in.setInt('/{:s}/scopes/0/single'.format(dev), 0)
        # 'trigenable' : enable the scope's trigger (boolean).
        #   0 - acquire continuous records
        #   1 - only acquire a record when a trigger arrives
        self.lock_in.setInt('/{:s}/scopes/0/trigenable'.format(dev), 0)
        # 'segments/enable' : Disable segmented data recording.
        self.lock_in.setInt('/{:s}/scopes/0/segments/enable'.format(dev), 0)
        # Perform a global synchronisation between the device and the data server:
        # Ensure that the settings have taken effect on the device before acquiring
        # data.
        self.lock_in.sync()
        
        # Now initialize and configure the Scope Module.
        self.scope = self.lock_in.scopeModule()
        # 'mode' : Scope data processing mode.
        # 0 - Pass through scope segments assembled, returned unprocessed, non-interleaved.
        # 1 - Moving average, scope recording assembled, scaling applied, averaged, if averaging is enabled.
        # 2 - Not yet supported.
        # 3 - As for mode 1, except an FFT is applied to every segment of the scope recording.
        self.scope.set('mode', mode)
        # 'averager/weight' : Averager behaviour.
        #   weight=1 - don't average.
        #   weight>1 - average the scope record shots using an exponentially weighted moving average.
        self.scope.set('averager/weight', num_averages)
        # 'historylength' : The number of scope records to keep in the Scope Module's memory, when more records
        #   arrive in the Module from the device the oldest records are overwritten.
        self.scope.set('historylength', historylength)
        
        self.scope.set('fft/power', power)
        self.scope.set('fft/spectraldensity', spectraldensity)
        
        
        # Use a Hann window function.
        self.scope.set('fft/window', 1)
        self.scope.set('averager/restart', 1)

        # Subscribe to the scope's data in the module.
        wave_nodepath = '/{:s}/scopes/0/wave'.format(dev)
        self.scope.subscribe(wave_nodepath)
        
        data_scope = get_scope_records(dev, self.lock_in, self.scope, num_records=num_averages)
        assert wave_nodepath in data_scope, f"The Scope Module did not return data for {wave_nodepath}."
        print(f"Number of scope records with triggering disabled: {len(data_scope[wave_nodepath])}.")
        check_scope_record_flags(data_scope[wave_nodepath])
        return data_scope    
    
    
    def synchronize_devices(self, device_ids, synchronize=True):
        """
        Added by Massiliano Rossi on 03.08.21
        Synchronize different devices together vis the MDS module
        
        Parameters
        ----------
        device_ids : list of str
            List of device IDs to synchronize
        synchronize : bool, optional
            Flag to enable or disable synchronization
        """
        discovery = ziPython.ziDiscovery()
        props = []
        for device_id in device_ids:
            device_serial = discovery.find(device_id).lower()
            props.append(discovery.get(device_serial))
        devices = props[0]["deviceid"]
        for prop in props[1:]:
            devices += "," + prop["deviceid"]
        # Switching between MFLI and UHFLI
        device_type = props[0]["devicetype"]
        for prop in props[1:]:
            if prop["devicetype"] != device_type:
                raise Exception(
                    "This example needs 2 or more MFLI instruments or 2 or more UHFLI instruments."
                    "Mixing device types is not possible"
                )
        for prop in props:
                if prop["devicetype"] == "UHFLI":
                    self.lock_in.connectDevice(prop["deviceid"], prop["interfaces"][0])
                else:
                    self.lock_in.connectDevice(prop["deviceid"], "1GbE")
        #  Device synchronization
        if synchronize:
            print("Synchronizing devices %s ...\n" % devices)
            self.mds = self.lock_in.multiDeviceSyncModule()
            self.mds.set("start", 0)
            self.mds.set("group", 0)
            self.mds.execute()
            self.mds.set("devices", devices)
            self.mds.set("start", 1)
        
            timeout = 20
            start = time.time()
            status = 0
            while status != 2:
                time.sleep(0.2)
                status = self.mds.getInt("status")
                if status == -1:
                    raise Exception("Error during device sync")
                if (time.time() - start) > timeout:
                    raise Exception("Timeout during device sync")
        
            print("Devices successfully synchronized.")
            
    def set_PID(self,
            dev,
            pid,
            mode=0,
            inputchannel=0,
            inputselect=1,
            setpoint=0,
            phaseunwrap=1,
            bw=5e3,
            order=4,
            harmonic=1, 
            outputselect=5,
            outputchannel=0,
            out_center=0,
            out_low=-0.5,
            out_high=0.5,
            gain_p=-8e3,
            gain_i=-10e3,
            gain_d=0,
            bw_d=0,
            rate=2.1e6):
        """
        Set PID controller parameters.
        
        Parameters
        ----------
        dev : str
            Device ID
        pid : int
            PID controller index
        mode : int, optional
            Mode of the PID controller
        inputchannel : int, optional
            Input channel for the PID controller
        inputselect : int, optional
            Input select for the PID controller
        setpoint : float, optional
            Setpoint value for the PID controller
        phaseunwrap : int, optional
            Phase unwrap flag
        bw : float, optional
            Bandwidth for the PID controller
        order : int, optional
            Order of the PID controller
        harmonic : int, optional
            Harmonic number for the PID controller
        outputselect : int, optional
            Output select for the PID controller
        outputchannel : int, optional
            Output channel for the PID controller
        out_center : float, optional
            Center value for the output
        out_low : float, optional
            Lower limit for the output
        out_high : float, optional
            Upper limit for the output
        gain_p : float, optional
            Proportional gain
        gain_i : float, optional
            Integral gain
        gain_d : float, optional
            Derivative gain
        bw_d : float, optional
            Bandwidth for the derivative filter
        rate : float, optional
            Rate for the PID controller
        """
        self.lock_in.setInt('/{:s}/pids/{:d}/mode'.format(dev, pid), mode)
        # Input
        self.lock_in.setInt('/{:s}/pids/{:d}/inputchannel'.format(dev, pid), inputchannel)
        self.lock_in.setInt('/{:s}/pids/{:d}/input'.format(dev, pid), inputselect)
        self.lock_in.setDouble('/{:s}/pids/{:d}/setpoint'.format(dev, pid), setpoint)
        self.lock_in.setInt('/{:s}/pids/{:d}/phaseunwrap'.format(dev, pid), phaseunwrap)
        self.lock_in.setDouble('/{:s}/pids/{:d}/demod/timeconstant'.format(dev, pid), ziutils.bw2tc(bw, order)) 
        self.lock_in.setInt('/{:s}/pids/{:d}/demod/order'.format(dev, pid), order)
        self.lock_in.setDouble('/{:s}/pids/{:d}/demod/harmonic'.format(dev, pid), harmonic)
        # Output
        self.lock_in.setInt('/{:s}/pids/{:d}/output'.format(dev, pid), outputselect) # Aux output offset
        self.lock_in.setInt('/{:s}/pids/{:d}/outputchannel'.format(dev, pid), outputchannel) # 1
        self.lock_in.setDouble('/{:s}/pids/{:d}/center'.format(dev, pid), out_center)
        self.lock_in.setDouble('/{:s}/pids/{:d}/limitlower'.format(dev, pid), out_low)
        self.lock_in.setDouble('/{:s}/pids/{:d}/limitupper'.format(dev, pid), out_high)
        # PID settings
        self.lock_in.setDouble('/{:s}/pids/{:d}/p'.format(dev, pid), gain_p)
        self.lock_in.setDouble('/{:s}/pids/{:d}/i'.format(dev, pid), gain_i)
        self.lock_in.setDouble('/{:s}/pids/{:d}/d'.format(dev, pid), gain_d)
        self.lock_in.setDouble('/{:s}/pids/{:d}/dlimittimeconstant'.format(dev, pid), bw_d)
        self.lock_in.setDouble('/{:s}/pids/{:d}/rate'.format(dev, pid), rate)  
    
def get_scope_records(device, daq, scopeModule, num_records=1):
    """
    Obtain scope records from the device using an instance of the Scope Module.
    
    Parameters
    ----------
    device : str
        Device ID
    daq : ziPython.ziDAQServer
        Instance of the ziDAQServer class
    scopeModule : object
        Instance of the Scope Module
    num_records : int, optional
        Number of records to acquire
        
    Returns
    -------
    dict
        Dictionary containing scope records
    """

    # Tell the module to be ready to acquire data; reset the module's progress to 0.0.
    scopeModule.execute()

    # Enable the scope: Now the scope is ready to record data upon receiving triggers.
    daq.setInt("/%s/scopes/0/enable" % device, 1)
    daq.sync()

    start = time.time()
    timeout = 30  # [s]
    records = 0
    progress = 0
    # Wait until the Scope Module has received and processed the desired number of records.
    while (records < num_records) or (progress < 1.0):
        time.sleep(0.5)
        records = scopeModule.getInt("records")
        progress = scopeModule.progress()[0]
        print(
            f"Scope module has acquired {records} records (requested {num_records}). "
            f"Progress of current segment {100.0 * progress}%.",
            end="\r",
        )
        # Advanced use: It's possible to read-out data before all records have been recorded (or even before all
        # segments in a multi-segment record have been recorded). Note that complete records are removed from the Scope
        # Module and can not be read out again; the read-out data must be managed by the client code. If a multi-segment
        # record is read-out before all segments have been recorded, the wave data has the same size as the complete
        # data and scope data points currently unacquired segments are equal to 0.
        #
        # data = scopeModule.read(True)
        # wave_nodepath = f"/{device}/scopes/0/wave"
        # if wave_nodepath in data:
        #   Do something with the data...
        if (time.time() - start) > timeout:
            # Break out of the loop if for some reason we're no longer receiving scope data from the device.
            print(f"\nScope Module did not return {num_records} records after {timeout} s - forcing stop.")
            break
    print("")
    daq.setInt("/%s/scopes/0/enable" % device, 0)

    # Read out the scope data from the module.
    data = scopeModule.read(True)

    # Stop the module; to use it again we need to call execute().
    scopeModule.finish()

    return data
    
def check_scope_record_flags(scope_records):
    """
    Loop over all records and print a warning to the console if an error bit in
    flags has been set.

    Warning: This function is intended as a helper function for the API's
    examples and it's signature or implementation may change in future releases.
    """
    num_records = len(scope_records)
    for index, record in enumerate(scope_records):
        if record[0]["flags"] & 1:
            print(f"Warning: Scope record {index}/{num_records} flag indicates dataloss.")
        if record[0]["flags"] & 2:
            print(f"Warning: Scope record {index}/{num_records} indicates missed trigger.")
        if record[0]["flags"] & 4:
            print(f"Warning: Scope record {index}/{num_records} indicates transfer failure (corrupt data).")
        totalsamples = record[0]["totalsamples"]
        for wave in record[0]["wave"]:
            # Check that the wave in each scope channel contains the expected number of samples.
            assert len(wave) == totalsamples, f"Scope record {index}/{num_records} size does not match totalsamples."