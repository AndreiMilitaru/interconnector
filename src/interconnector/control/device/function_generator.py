"""
Author: Andrei Militaru
Organization: Institute of Science and Technology Austria (ISTA)
Date: October 2025
Description: Module to remotely control function generators.
Code is adapted from a prior skeleton code developed by Andrei Militaru and Massimiliano Rossi in the Photonics Lab of ETH Zurich.
"""

import pyvisa
import time


class FunctionGenerator:
    """
    Parent class for function generators.
    Provides common interface for remote control.
    """

    def __init__(self, device):
        """
        :param device: str, VISA resource string of desired instrument.
        """
        rm = pyvisa.ResourceManager()
        self.instrument = rm.open_resource(device)
        self.idn = self.query("*IDN?")

    def send_command(self, command):
        """
        Send a command to the instrument and return response if any.
        """
        return self.query(command)

    def query(self, scpi):
        """
        Query an instrument, returns the response if scpi asks for information.
        scpi : string
        message : bytes
        """
        self.instrument.write(scpi)
        if '?' in scpi:
            message = b''
            while True:
                charac = self.instrument.read_bytes(1)
                message += charac
                if charac == b'\n':
                    break
            return message
        else:
            return b''

class DualOutput(FunctionGenerator):
    """
    Generic class for dual-output function generators (e.g., PeakTech 4046).
    """

    def all_on(self):
        """
        Turns on both outputs and the sync signal.
        Warning: the sync signal refers to the last channel that has been updated!
        """
        command = 'OUTPut1 ON;OUTPut2 ON;OUTP:SYNC ON'
        return self.send_command(command)

    def all_off(self):
        """
        Turns off both outputs and the sync signal.
        """
        command = 'OUTPut1 OFF;OUTPut2 OFF;OUTP:SYNC OFF'
        return self.send_command(command)

    @property
    def sync(self):
        command = 'OUTP:SYNC?'
        state = self.send_command(command)
        if isinstance(state, bytes):
            state = state.decode().strip()
        if state in ('0', '0\n'):
            return False
        elif state in ('1', '1\n'):
            return True
        else:
            raise Exception('Response not recognized.')

    @sync.setter
    def sync(self, value):
        state = 'ON' if value else 'OFF'
        command = 'OUTP:SYNC ' + state
        self.send_command(command)

    @property
    def out1_frequency(self):
        command = 'source1:frequency?'
        value = self.send_command(command)
        if isinstance(value, bytes):
            value = value.decode().strip()
        try:
            return float(value)
        except Exception:
            return value

    @out1_frequency.setter
    def out1_frequency(self, new_value):
        if new_value < 0:
            raise Exception('Frequencies must be positive.')
        else:
            command = 'source1:frequency {:f}Hz'.format(new_value)
            self.send_command(command)

    @property
    def out2_frequency(self):
        command = 'source2:frequency?'
        value = self.send_command(command)
        if isinstance(value, bytes):
            value = value.decode().strip()
        try:
            return float(value)
        except Exception:
            return value

    @out2_frequency.setter
    def out2_frequency(self, new_value):
        if new_value < 0:
            raise Exception('Frequencies must be positive.')
        else:
            command = 'source2:frequency {:f}Hz'.format(new_value)
            self.send_command(command)

    @property
    def out1_waveform(self):
        command = 'source1:function?'
        value = self.send_command(command)
        if isinstance(value, bytes):
            value = value.decode().strip()
        return value

    @out1_waveform.setter
    def out1_waveform(self, new_wave):
        if new_wave not in ['sinusoid', 'square', 'ramp', 'pulse', 'noise']:
            raise Exception('Waveform not recognized.')
        else:
            command = 'source1:function ' + new_wave
            self.send_command(command)

    @property
    def out2_waveform(self):
        command = 'source2:function?'
        value = self.send_command(command)
        if isinstance(value, bytes):
            value = value.decode().strip()
        return value

    @out2_waveform.setter
    def out2_waveform(self, new_wave):
        if new_wave not in ['sinusoid', 'square', 'ramp', 'pulse', 'noise']:
            raise Exception('Waveform not recognized.')
        else:
            command = 'source2:function ' + new_wave
            self.send_command(command)

    @property
    def out1(self):
        command = 'OUTPut1?'
        state = self.send_command(command)
        if isinstance(state, bytes):
            state = state.decode().strip()
        if state in ('0', '0\n'):
            return False
        elif state in ('1', '1\n'):
            return True
        else:
            raise Exception('Response not recognized.')

    @out1.setter
    def out1(self, value):
        state = 'ON' if value else 'OFF'
        command = 'OUTPut1 ' + state
        self.send_command(command)

    @property
    def out2(self):
        command = 'OUTPut2?'
        state = self.send_command(command)
        if isinstance(state, bytes):
            state = state.decode().strip()
        if state in ('0', '0\n'):
            return False
        elif state in ('1', '1\n'):
            return True
        else:
            raise Exception('Response not recognized.')

    @out2.setter
    def out2(self, value):
        state = 'ON' if value else 'OFF'
        command = 'OUTPut2 ' + state
        self.send_command(command)

    @property
    def out1_amplitude(self):
        command = 'SOURce1:VOLT?'
        rsp = self.send_command(command)
        if isinstance(rsp, bytes):
            rsp = rsp.decode().strip()
        return float(rsp)

    @out1_amplitude.setter
    def out1_amplitude(self, value):
        command = 'SOURce1:VOLT {:f}Vpp'.format(value)
        self.send_command(command)

    @property
    def out2_amplitude(self):
        command = 'SOURce2:VOLT?'
        rsp = self.send_command(command)
        if isinstance(rsp, bytes):
            rsp = rsp.decode().strip()
        return float(rsp)

    @out2_amplitude.setter
    def out2_amplitude(self, value):
        command = 'SOURce2:VOLT {:f}Vpp'.format(value)
        self.send_command(command)

    @property
    def out1_phase(self):
        command = 'SOURce1:PHAS?'
        rsp = self.send_command(command)
        if isinstance(rsp, bytes):
            rsp = rsp.decode().strip()
        return float(rsp)

    @out1_phase.setter
    def out1_phase(self, value):
        command = 'SOURce1:PHAS {:f}deg'.format(value)
        self.send_command(command)

    @property
    def out2_phase(self):
        command = 'SOURce2:PHAS?'
        rsp = self.send_command(command)
        if isinstance(rsp, bytes):
            rsp = rsp.decode().strip()
        return float(rsp)

    @out2_phase.setter
    def out2_phase(self, value):
        command = 'SOURce2:PHAS {:f}deg'.format(value)
        self.send_command(command)

    @property
    def out1_pulse_width(self):
        command = 'source1function:pulse:width?'
        rsp = self.send_command(command)
        if isinstance(rsp, bytes):
            rsp = rsp.decode().strip()
        return float(rsp)

    @out1_pulse_width.setter
    def out1_pulse_width(self, value):
        command = 'source1:function:pulse:width {:f}s'.format(value)
        self.send_command(command)

    @property
    def out2_pulse_width(self):
        command = 'source2function:pulse:width?'
        rsp = self.send_command(command)
        if isinstance(rsp, bytes):
            rsp = rsp.decode().strip()
        return float(rsp)

    @out2_pulse_width.setter
    def out2_pulse_width(self, value):
        command = 'source2:function:pulse:width {:f}s'.format(value)
        self.send_command(command)
        

class SingleOutput(FunctionGenerator):
    """
    Generic class for single-output function generators (e.g., Agilent 33250A).
    """

    @property
    def out_frequency(self):
        command = 'source:frequency?'
        value = self.send_command(command)
        if isinstance(value, bytes):
            value = value.decode().strip()
        try:
            return float(value)
        except Exception:
            return value

    @out_frequency.setter
    def out_frequency(self, new_value):
        """
        Setting the frequency of output1.
        :param new_value: float, value (in Hz) of the output frequency.
        """
        if new_value < 0:
            raise Exception('Frequencies must be positive.')
        else:
            command = 'source:frequency {:f}'.format(new_value)
            self.send_command(command)

    @property
    def out_waveform(self):
        command = 'source:function?'
        value = self.send_command(command)
        if isinstance(value, bytes):
            value = value.decode().strip()
        return value

    @out_waveform.setter
    def out_waveform(self, new_wave):
        """
        Setting the waveform of output1.
        :param new_wave: str, must be in ['sin', 'square', 'ramp', 'pulse', 'noise']
        """
        if new_wave not in ['sin', 'square', 'ramp', 'pulse', 'noise']:
            raise Exception('Waveform not recognized.')
        else:
            command = 'source:function ' + new_wave
            self.send_command(command)

    @property
    def out(self):
        command = 'OUTPut?'
        state = self.send_command(command)
        if isinstance(state, bytes):
            state = state.decode().strip()
        if state in ('0', '0\n'):
            return False
        elif state in ('1', '1\n'):
            return True
        else:
            raise Exception('Response not recognized.')
    
    @out.setter
    def out(self, value):
        state = 'ON' if value else 'OFF'
        command = 'OUTPut ' + state
        self.send_command(command)

    @property
    def sync(self):
        command = 'OUTP:SYNC?'
        state = self.send_command(command)
        if isinstance(state, bytes):
            state = state.decode().strip()
        if state in ('0', '0\n'):
            return False
        elif state in ('1', '1\n'):
            return True
        else:
            raise Exception('Response not recognized.')

    @property
    def out_amplitude(self):
        command = 'SOURce:VOLT?'
        rsp = self.send_command(command)
        if isinstance(rsp, bytes):
            rsp = rsp.decode().strip()
        return float(rsp)

    @out_amplitude.setter
    def out_amplitude(self, value):
        """
        Amplitude (in volts) of output1.
        :param value: float, desired amplitude in V.
        """
        command = 'SOURce:VOLT {:f}'.format(value)
        self.send_command(command)

    @property
    def out_offset(self):
        command = 'SOURce:VOLT:OFFset?'
        rsp = self.send_command(command)
        if isinstance(rsp, bytes):
            rsp = rsp.decode().strip()
        return float(rsp)

    @out_offset.setter
    def out_offset(self, value):
        """
        Offset (in volts) of output1.
        :param value: float, desired offset in V.
        """
        command = 'SOURce:VOLT:OFFset {:f}'.format(value)
        self.send_command(command)

    @property
    def out_phase(self):
        command = 'SOURce:PHAS?'
        rsp = self.send_command(command)
        if isinstance(rsp, bytes):
            rsp = rsp.decode().strip()
        return float(rsp)

    @out_phase.setter
    def out_phase(self, value):
        """
        Phase delay (in deg) of output1.
        :param value: float, desired phase in deg.
        """
        command = 'SOURce:PHAS {:f}deg'.format(value)
        self.send_command(command)

    @property
    def out_pulse_width(self):
        command = 'source:function:pulse:width?'
        rsp = self.send_command(command)
        if isinstance(rsp, bytes):
            rsp = rsp.decode().strip()
        return float(rsp)

    @out_pulse_width.setter
    def out_pulse_width(self, value):
        """
        Pulse width (in seconds) of output1. Useful when the output waveform is Pulse mode.
        :param value: float, desired pulse width in s.
        """
        command = 'source:function:pulse:width {:f}s'.format(value)
        self.send_command(command)