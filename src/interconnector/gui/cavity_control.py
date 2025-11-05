"""
Cavity Control GUI
Author: Andrei Militaru (event handlers and logic), GitHub Copilot (layout and structure)
Organization: Institute of Science and Technology Austria (ISTA)
Date: October 2025
Description: GUI for controlling narrow-linewidth optical cavity with 
demodulation recorder and function generator.
Values that work well with the offset adjustment routine, as of 29th October 2025, are
-- PID, p gain 1500 and I gain 1000, with bandwidth 50Hz. 
-- function generator 40mV (or 60mV also) amplitude and 5Hz frequency (with option keep_offset_zero enabled)
-- dither tone 500kHz and 5mV strength, with 0 phase shift.

Warnings: the two main bugs known as of November 2025 are the following: the button for stopping the routine does not
appear when it should, and the Output Settings group is not always disabled once the PID is enabled at the end of
the mode finding routine. 
"""

import sys
from pathlib import Path
import os
import numpy as np
import time
import threading
import peakutils
import logging
from datetime import datetime
from interconnector.mach_zehnder_utils.mach_zehnder_lock import df2tc
from interconnector.zhinst_utils.scope_settings import get_data_scope
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QDoubleSpinBox, QCheckBox, QComboBox, QPushButton,
    QGroupBox, QGridLayout, QFrame, QSizePolicy, QTabWidget,
    QSlider, QTextEdit
)
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal, QObject, QMetaObject, Q_ARG, QTimer
from PyQt5.QtGui import QFont, QIcon


class QTextEditLogger(logging.Handler):
    """Custom logging handler that emits to a QTextEdit widget"""
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit
        
    def emit(self, record):
        msg = self.format(record)
        # Thread-safe GUI update
        QMetaObject.invokeMethod(
            self.text_edit,
            "append",
            Qt.QueuedConnection,
            Q_ARG(str, msg)
        )
        # Auto-scroll to bottom
        QMetaObject.invokeMethod(
            self.text_edit.verticalScrollBar(),
            "setValue",
            Qt.QueuedConnection,
            Q_ARG(int, self.text_edit.verticalScrollBar().maximum())
        )


class CavityControlGUI(QMainWindow):
    """Main GUI for optical cavity control"""
    # Update waveform list to match device capabilities, removing triangle
    WAVEFORMS = ["sin", "square", "ramp"]

    def __init__(self, mdrec=None, fg=None, parent=None, device_id=None,
                  dither_pid=None, dither_drive_demod=None, dither_in_demod=None,
                  verbose=False, mdrec_lock=None, fg_lock=None, slow_offset=2,
                  keep_offset_zero=True, mode_finding_settings=None, mid_baseline_threshold=2.0,
                  locked_reflection_threshold=2.0, logfile=None):
        """Initialize the GUI with optional verbose mode"""
        super().__init__(parent)
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d_%H-%M-%S')
        
        # Add file handler if logfile is provided
        if logfile is not None:
            logfile.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(logfile)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Add console handler if verbose
        if verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        self.verbose = verbose
        self.mdrec = mdrec
        self.fg = fg
        self.device_id = device_id
        self.dither_pid = dither_pid
        self.dither_drive_demod = dither_drive_demod
        self.dither_in_demod = dither_in_demod
        # Add locks for thread safety
        self.mdrec_lock = mdrec_lock
        self.fg_lock = fg_lock
        # Add slow offset aux index
        self.slow_offset = slow_offset
        # Base offset for slow offset control (set during initialization)
        self.slow_offset_base = 0.0
        self.keep_offset_zero = keep_offset_zero
        self.locked_reflection_threshold = locked_reflection_threshold
        
        # Initialize reflection monitoring thread control
        self.reflection_thread = None
        self.reflection_thread_running = False
        
        # Initialize auto-offset management thread control
        self.auto_offset_thread = None
        self.auto_offset_thread_running = False

        # Initialize auto mode finder thread control
        self.auto_mode_finder_thread = None
        self.auto_mode_finder_thread_running = False

        # Initialize offset monitor thread control
        self.offset_monitor_thread = None
        self.offset_monitor_thread_running = False

        # Add lock to prevent overlapping critical routines (mode finding and offset ramping)
        self.routine_lock = threading.Lock()

        self.mode_finding_settings = mode_finding_settings
        self.mid_baseline_threshold = mid_baseline_threshold
        # Initialize UI
        self.init_ui()
        
        # Add GUI handler after text_edit is created (in init_ui)
        gui_handler = QTextEditLogger(self.log_text_edit)
        gui_handler.setFormatter(formatter)
        self.logger.addHandler(gui_handler)
        
        # Mode finding stop flag
        self.mode_finding_stop_requested = False

        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Narrow-linewidth cavity control')
        # Reduce the window width to half (from 800 to 400)
        self.setGeometry(1000, 50, 350, 500)

        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Create two main panels
        top_panel = self.create_controls_panel()
        bottom_panel = self.create_monitoring_panel()

        # Add panels to main layout
        main_layout.addWidget(top_panel, 4)  # Control panel takes more space
        main_layout.addWidget(bottom_panel, 1)  # Status panel

        # Set central widget
        self.setCentralWidget(central_widget)
        # Set initial values from devices
        self.set_initial_values_from_devices()

    def pid_output_value(self):
        """Get current PID output value from mdrec"""
        with self.mdrec_lock:
            return self.mdrec.lock_in.get(f'/{self.device_id}/pids/{self.dither_pid}/value')

    def recenter_PID_output(self):
        """Recenter PID range around the current output value"""
        current_output = self.get_mdrec_output_offset()
        with self.mdrec_lock:
            self.mdrec.lock_in.set(f'/{self.device_id}/pids/{self.dither_pid}/center', current_output)
            self.mdrec.lock_in.set(f'/{self.device_id}/pids/{self.dither_pid}/limitlower', -current_output)
            self.mdrec.lock_in.set(f'/{self.device_id}/pids/{self.dither_pid}/limitupper', 1.0 - current_output)

    def get_mdrec_p_gain(self):
        """Get P gain from mdrec"""
        with self.mdrec_lock:
            response = self.mdrec.lock_in.get(f'/{self.device_id}/pids/{self.dither_pid}/p')
            return float(response[self.device_id]['pids'][str(self.dither_pid)]['p']['value'][0])

    def get_mdrec_i_gain(self):
        """Get I gain from mdrec"""
        with self.mdrec_lock:
            response = self.mdrec.lock_in.get(f'/{self.device_id}/pids/{self.dither_pid}/i') 
            return float(response[self.device_id]['pids'][str(self.dither_pid)]['i']['value'][0])

    def get_mdrec_bandwidth(self):
        """Get bandwidth from mdrec"""
        with self.mdrec_lock:
            response = self.mdrec.lock_in.get(f'/{self.device_id}/pids/{self.dither_pid}/demod/timeconstant')
            return df2tc(float(response[self.device_id]['pids'][str(self.dither_pid)]['demod']['timeconstant']['value'][0]))

    def get_mdrec_pid_enabled(self):
        """Get PID enabled state from mdrec"""
        with self.mdrec_lock:
            response = self.mdrec.lock_in.get(f'/{self.device_id}/pids/{self.dither_pid}/enable')
            return float(response[self.device_id]['pids'][str(self.dither_pid)]['enable']['value'][0]) == 1

    def get_mdrec_keep_i(self):
        """Get keep I value from mdrec"""
        with self.mdrec_lock:
            response = self.mdrec.lock_in.get(f'/{self.device_id}/pids/{self.dither_pid}/keepint')
            return float(response[self.device_id]['pids'][str(self.dither_pid)]['keepint']['value'][0]) == 1

    def get_mdrec_output_offset(self):
        """Get output offset from mdrec"""
        with self.mdrec_lock:
            response = self.mdrec.lock_in.get(f'/{self.device_id}/sigouts/0/offset')
            return float(response[self.device_id]['sigouts']['0']['offset']['value'][0])

    def get_mdrec_dither_freq(self):
        """Get dither frequency from mdrec"""
        with self.mdrec_lock:
            response = self.mdrec.lock_in.get(f'/{self.device_id}/oscs/{self.dither_drive_demod}/freq')
            # Convert Hz to kHz for display
            return float(response[self.device_id]['oscs'][str(self.dither_drive_demod)]['freq']['value'][0]) / 1000.0

    def get_mdrec_dither_strength(self):
        """Get dither strength from mdrec"""
        with self.mdrec_lock:
            response = self.mdrec.lock_in.get(f'/{self.device_id}/sigouts/0/amplitudes/{self.dither_drive_demod}')
            return float(response[self.device_id]['sigouts']['0']['amplitudes'][str(self.dither_drive_demod)]['value'][0])

    def get_mdrec_demod_phase(self):
        """Get demodulation phase from mdrec"""
        with self.mdrec_lock:
            response = self.mdrec.lock_in.get(f'/{self.device_id}/demods/{self.dither_in_demod}/phaseshift')
            return float(response[self.device_id]['demods'][str(self.dither_in_demod)]['phaseshift']['value'][0])

    def get_fg_waveform(self):
        """Get waveform from fg"""
        with self.fg_lock:
            # Strip any whitespace and convert to lowercase to ensure proper matching
            return self.fg.out_waveform.strip().lower()

    def get_fg_amplitude(self):
        """Get amplitude from fg in mV"""
        with self.fg_lock:
            return self.fg.out_amplitude * 1000  # Convert V to mV

    def get_fg_frequency(self):
        """Get frequency from fg"""
        with self.fg_lock:
            return self.fg.out_frequency

    def get_fg_offset(self):
        """Get offset from fg"""
        with self.fg_lock:
            return self.fg.out_offset

    def get_fg_output_enabled(self):
        """Get output enabled state from fg"""
        with self.fg_lock:
            return self.fg.out

    def get_mdrec_dither_enable(self):
        """Get dither enable state from mdrec"""
        with self.mdrec_lock:
            response = self.mdrec.lock_in.get(f'/{self.device_id}/sigouts/0/enables/{self.dither_drive_demod}')
            return float(response[self.device_id]['sigouts']['0']['enables'][str(self.dither_drive_demod)]['value'][0]) == 1

    def get_mdrec_slow_offset(self):
        """Get slow offset control voltage from mdrec"""
        with self.mdrec_lock:
            response = self.mdrec.lock_in.get(f'/{self.device_id}/auxouts/{self.slow_offset}/offset')
            return float(response[self.device_id]['auxouts'][str(self.slow_offset)]['offset']['value'][0])

    def set_initial_values_from_devices(self):
        """Set initial values for widgets from mdrec and fg"""
        # Block signals during initialization to prevent unnecessary updates
        self.p_gain_spinbox.blockSignals(True)
        self.i_gain_spinbox.blockSignals(True)
        self.bandwidth_spinbox.blockSignals(True)
        self.pid_enable_checkbox.blockSignals(True)
        self.keep_i_checkbox.blockSignals(True)
        self.offset_spinbox.blockSignals(True)
        self.dither_freq_spinbox.blockSignals(True)
        self.dither_strength_spinbox.blockSignals(True)
        self.demod_phase_spinbox.blockSignals(True)
        self.dither_enable_checkbox.blockSignals(True)
        
        # Set values
        self.p_gain_spinbox.setValue(self.get_mdrec_p_gain())
        self.i_gain_spinbox.setValue(self.get_mdrec_i_gain())
        self.bandwidth_spinbox.setValue(self.get_mdrec_bandwidth())
        self.pid_enable_checkbox.setChecked(self.get_mdrec_pid_enabled())
        self.keep_i_checkbox.setChecked(self.get_mdrec_keep_i())
        
        # Set dither and demodulation values
        self.dither_freq_spinbox.setValue(self.get_mdrec_dither_freq())
        self.dither_strength_spinbox.setValue(self.get_mdrec_dither_strength() * 1000.0)  # Convert V to mV
        self.demod_phase_spinbox.setValue(self.get_mdrec_demod_phase())
        self.dither_enable_checkbox.setChecked(self.get_mdrec_dither_enable())
        
        # Get offset value first and block signals
        self.offset_spinbox.blockSignals(True)
        self.offset_slider.blockSignals(True)
        
        # Get current offset value from device
        offset_value = self.get_mdrec_output_offset()
        
        # Set base offset to the device value and initialize fine adjustment to 0
        self.base_offset = offset_value
        
        # Initialize fine offset slider to 0
        self.fine_offset_slider.blockSignals(True)
        self.fine_offset_slider.setValue(0)
        self.fine_offset_label.setText("0.0 mV")
        self.fine_offset_slider.blockSignals(False)
        
        # Set spinbox to the total (which is just base offset now)
        self.offset_spinbox.blockSignals(True)
        self.offset_spinbox.setValue(offset_value)
        self.offset_spinbox.blockSignals(False)
        
        # Set slider to match base offset
        self.offset_slider.blockSignals(True)
        self.offset_slider.setValue(int(offset_value * 100))
        self.offset_slider.blockSignals(False)
        
        # Update status indicators after setting values
        self.output_value_label.setText(f"{offset_value:.3f} V")
        self.update_status_indicators()
        
        # Add slow offset initialization - read current value from device
        self.slow_offset_spinbox.blockSignals(True)
        self.slow_offset_slider.blockSignals(True)
        self.slow_offset_fine_slider.blockSignals(True)
        
        # Read current value directly from device
        try:
            slow_offset_value = self.get_mdrec_slow_offset()
            self.logger.info(f"Initial slow offset value read from device: {slow_offset_value:.3f}V")
        except Exception as e:
            # Default to 4.0V if reading fails
            self.logger.warning(f"Failed to read slow offset from device: {e}")
            slow_offset_value = 4.0
        
        self.slow_offset_base = slow_offset_value  # Initialize base value
        
        # Set controls with actual value from device
        self.slow_offset_spinbox.setValue(slow_offset_value)
        self.start_v_spinbox.setValue(slow_offset_value-0.25)
        self.stop_v_spinbox.setValue(slow_offset_value+0.25)
        self.slow_offset_slider.setValue(int(slow_offset_value * 100))
        self.slow_offset_fine_slider.setValue(0)  # Fine adjustment starts at 0
        self.slow_offset_fine_label.setText("0.0 mV")
        
        self.slow_offset_spinbox.blockSignals(False)
        self.slow_offset_slider.blockSignals(False)
        self.slow_offset_fine_slider.blockSignals(False)
        
        # Unblock signals after setting values
        self.p_gain_spinbox.blockSignals(False)
        self.i_gain_spinbox.blockSignals(False)
        self.bandwidth_spinbox.blockSignals(False)
        self.pid_enable_checkbox.blockSignals(False)
        self.keep_i_checkbox.blockSignals(False)
        self.offset_spinbox.blockSignals(False)
        self.dither_freq_spinbox.blockSignals(False)
        self.dither_strength_spinbox.blockSignals(False)
        self.demod_phase_spinbox.blockSignals(False)
        self.dither_enable_checkbox.blockSignals(False)
        
        # FG initialization
        # Block signals for FG controls
        self.waveform_combo.blockSignals(True)
        self.amplitude_spinbox.blockSignals(True)
        self.freq_spinbox.blockSignals(True)
        self.fg_offset_spinbox.blockSignals(True)
        self.output_checkbox.blockSignals(True)
        self.amplitude_fine_slider.blockSignals(True)
        
        # Set values
        waveform = self.get_fg_waveform()
        self.logger.debug(f"Read waveform from FG: '{waveform}'")
        
        # Find matching waveform in combo box
        index = -1
        for i, wf in enumerate(self.WAVEFORMS):
            if wf.lower() == waveform.lower():
                index = i
                break
        
        if index >= 0:
            self.waveform_combo.setCurrentIndex(index)
        else:
            self.logger.warning(f"Waveform '{waveform}' not found in list, defaulting to first option")
            self.waveform_combo.setCurrentIndex(0)
        
        amplitude_mv = self.get_fg_amplitude()
        self.amplitude_spinbox.setValue(amplitude_mv)
        self.amplitude_fine_slider.setValue(0)  # Reset fine adjustment to 0
        self.amplitude_fine_label.setText("0 mV")
        
        self.freq_spinbox.setValue(self.get_fg_frequency())
        
        # Calculate offset from amplitude (should be amplitude/2)
        if not self.keep_offset_zero:
            offset_mv = amplitude_mv / 2.0
        else:
            offset_mv = 0.0
        self.fg_offset_spinbox.setValue(offset_mv)
        
        self.output_checkbox.setChecked(self.get_fg_output_enabled())
        
        # Unblock signals
        self.waveform_combo.blockSignals(False)
        self.amplitude_spinbox.blockSignals(False)
        self.freq_spinbox.blockSignals(False)
        self.fg_offset_spinbox.blockSignals(False)
        self.output_checkbox.blockSignals(False)
        self.amplitude_fine_slider.blockSignals(False)

        # Initialize fine offset slider to 0
        self.fine_offset_slider.blockSignals(True)
        self.fine_offset_slider.setValue(0)  # Always start at 0
        self.fine_offset_label.setText("0.0 mV")
        self.fine_offset_slider.blockSignals(False)
        
        # Update offset spinbox state based on initial PID enable state
        self.update_offset_spinbox_state()

        # After all signals are unblocked and offset spinbox state is updated, 
        # Start offset monitor thread if PID is enabled
        if self.pid_enable_checkbox.isChecked():
            self.logger.info("Starting offset monitoring during initialization")
            self.start_offset_monitoring()

    def disable_pid(self):
        """Safely disable PID by triggering checkbox state change"""
        if self.pid_enable_checkbox.isChecked():
            self.pid_enable_checkbox.setChecked(False)  # This will trigger on_pid_enable_changed

    def enable_pid(self):
        """Safely enable PID by triggering checkbox state change"""
        if not self.pid_enable_checkbox.isChecked():
            self.pid_enable_checkbox.setChecked(True)  # This will trigger on_pid_enable_changed

    def set_pid_enabled(self, enabled):
        """Set PID enabled state (True to enable, False to disable)"""
        if enabled:
            self.enable_pid()
        else:
            self.disable_pid()

    def set_slow_offset(self, value):
        """
        Programmatically set the slow offset voltage and update all GUI controls
        
        Args:
            value (float): Slow offset voltage in volts (must be between 1.5V and 6.5V)
        """
        # Clamp value to valid range
        value = max(1.5, min(6.5, value))
        
        # Update the device
        with self.mdrec_lock:
            self.mdrec.lock_in.set(f'/{self.device_id}/auxouts/{self.slow_offset}/offset', value)
        
        # Update the base value (assuming fine adjustment is at 0)
        self.slow_offset_base = value
        
        # Block signals to prevent triggering event handlers
        self.slow_offset_spinbox.blockSignals(True)
        self.slow_offset_slider.blockSignals(True)
        self.slow_offset_fine_slider.blockSignals(True)
        
        # Update spinbox (value is already in volts)
        self.slow_offset_spinbox.setValue(value)  # Fixed: removed *1000
        
        # Update slider (convert voltage to slider value)
        self.slow_offset_slider.setValue(int(value * 100))
        
        # Reset fine adjustment to 0
        self.slow_offset_fine_slider.setValue(0)
        self.slow_offset_fine_label.setText("0.0 mV")
        
        # Unblock signals
        self.slow_offset_spinbox.blockSignals(False)
        self.slow_offset_slider.blockSignals(False)
        self.slow_offset_fine_slider.blockSignals(False)

    def create_controls_panel(self):
        """Create the main controls panel with tabs"""
        panel = QTabWidget()
        
        # Create tabs for different control groups
        pid_tab = self.create_pid_controls()
        fg_tab = self.create_fg_controls()
        demod_tab = self.create_demod_controls()
        log_tab = self.create_log_tab()
        
        # Add tabs to panel
        panel.addTab(pid_tab, "Lock Controls")
        panel.addTab(fg_tab, "Function Generator")
        panel.addTab(demod_tab, "Demodulation Settings")
        panel.addTab(log_tab, "Log")
        
        return panel
    
    def create_log_tab(self):
        """Create the log tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Create text edit for log messages (read-only)
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                font-family: Consolas, Monaco, monospace;
                font-size: 8pt;
            }
        """)
        
        layout.addWidget(self.log_text_edit)
        
        # Add a clear button
        clear_button = QPushButton("Clear Log")
        clear_button.clicked.connect(self.clear_log)
        clear_button.setMaximumWidth(100)
        layout.addWidget(clear_button)
        
        return widget
    
    def clear_log(self):
        """Clear the log display"""
        self.log_text_edit.clear()
    
    def create_pid_controls(self):
        """Create PID controller controls"""
        widget = QWidget()
        
        # Use a QVBoxLayout to arrange groups vertically
        layout = QVBoxLayout(widget)
        
        # PID Parameters Group
        pid_group = QGroupBox("Lock Control")
        pid_layout = QGridLayout()

        # P Gain
        pid_layout.addWidget(QLabel("P Gain:"), 0, 0)
        self.p_gain_spinbox = QDoubleSpinBox()
        self.p_gain_spinbox.setRange(-1000000, 1000000)
        self.p_gain_spinbox.setValue(0.0)
        self.p_gain_spinbox.setDecimals(3)
        self.p_gain_spinbox.setSingleStep(0.1)
        self.p_gain_spinbox.setKeyboardTracking(False)  # Only update when Enter is pressed
        self.p_gain_spinbox.valueChanged.connect(self.on_p_gain_changed)
        pid_layout.addWidget(self.p_gain_spinbox, 0, 1)

        # Start V (for mode finding) - next to P Gain
        pid_layout.addWidget(QLabel("Mode Finding Start (V):"), 0, 2)
        self.start_v_spinbox = QDoubleSpinBox()
        self.start_v_spinbox.setRange(1.5, 6.5)
        self.start_v_spinbox.setValue(2.5)
        self.start_v_spinbox.setDecimals(2)
        self.start_v_spinbox.setSingleStep(0.1)
        self.start_v_spinbox.setKeyboardTracking(False)
        pid_layout.addWidget(self.start_v_spinbox, 0, 3)

        # I Gain
        pid_layout.addWidget(QLabel("I Gain:"), 1, 0)
        self.i_gain_spinbox = QDoubleSpinBox()
        self.i_gain_spinbox.setRange(-1000000, 1000000)
        self.i_gain_spinbox.setValue(0.0)
        self.i_gain_spinbox.setDecimals(3)
        self.i_gain_spinbox.setSingleStep(0.1)
        self.i_gain_spinbox.setKeyboardTracking(False)  # Only update when Enter is pressed
        self.i_gain_spinbox.valueChanged.connect(self.on_i_gain_changed)
        pid_layout.addWidget(self.i_gain_spinbox, 1, 1)

        # Stop V (for mode finding) - next to I Gain
        pid_layout.addWidget(QLabel("Mode Finding Stop (V):"), 1, 2)
        self.stop_v_spinbox = QDoubleSpinBox()
        self.stop_v_spinbox.setRange(1.5, 6.5)
        self.stop_v_spinbox.setValue(5.5)
        self.stop_v_spinbox.setDecimals(2)
        self.stop_v_spinbox.setSingleStep(0.1)
        self.stop_v_spinbox.setKeyboardTracking(False)
        pid_layout.addWidget(self.stop_v_spinbox, 1, 3)

        # Bandwidth
        pid_layout.addWidget(QLabel("Bandwidth (Hz):"), 2, 0)
        self.bandwidth_spinbox = QDoubleSpinBox()
        self.bandwidth_spinbox.setRange(0.1, 1000000)
        self.bandwidth_spinbox.setValue(100.0)
        self.bandwidth_spinbox.setDecimals(1)
        self.bandwidth_spinbox.setSingleStep(10)
        self.bandwidth_spinbox.setKeyboardTracking(False)  # Only update when Enter is pressed
        self.bandwidth_spinbox.valueChanged.connect(self.on_bandwidth_changed)
        pid_layout.addWidget(self.bandwidth_spinbox, 2, 1)

        # Stop Routine button
        self.stop_mode_button = QPushButton("Stop Routine")
        self.stop_mode_button.clicked.connect(self.on_stop_mode_clicked)
        self.stop_mode_button.setEnabled(False)  # Initially disabled
        self.stop_mode_button.setVisible(False)  # Initially hidden
        pid_layout.addWidget(self.stop_mode_button, 2, 2, 1, 2)

        # PID Enable
        pid_layout.addWidget(QLabel("PID Enable:"), 3, 0)
        pid_enable_layout = QHBoxLayout()
        self.pid_enable_checkbox = QCheckBox()
        self.pid_enable_checkbox.setChecked(False)
        self.pid_enable_checkbox.stateChanged.connect(self.on_pid_enable_changed)
        pid_enable_layout.addWidget(self.pid_enable_checkbox)
        pid_enable_layout.addStretch()
        pid_layout.addLayout(pid_enable_layout, 3, 1)
        
        # Auto Offset Management checkbox
        pid_layout.addWidget(QLabel("Offset adjustment:"), 3, 2)
        self.auto_offset_checkbox = QCheckBox()
        self.auto_offset_checkbox.setChecked(False)
        self.auto_offset_checkbox.stateChanged.connect(self.on_auto_offset_changed)
        pid_layout.addWidget(self.auto_offset_checkbox, 3, 3)
        
        # Keep I Value
        pid_layout.addWidget(QLabel("Keep I Value:"), 4, 0)
        keep_i_layout = QHBoxLayout()
        self.keep_i_checkbox = QCheckBox()
        self.keep_i_checkbox.setChecked(True)
        self.keep_i_checkbox.stateChanged.connect(self.on_keep_i_changed)
        keep_i_layout.addWidget(self.keep_i_checkbox)
        keep_i_layout.addStretch()
        pid_layout.addLayout(keep_i_layout, 4, 1)

        # Monitor reflection checkbox
        pid_layout.addWidget(QLabel("Monitor Reflection:"), 4, 2)
        self.monitor_reflection_checkbox = QCheckBox()
        self.monitor_reflection_checkbox.setChecked(False)
        self.monitor_reflection_checkbox.stateChanged.connect(self.on_monitor_reflection_changed)
        pid_layout.addWidget(self.monitor_reflection_checkbox, 4, 3)

        # Find Mode button
        self.find_mode_button = QPushButton("Find Mode")
        self.find_mode_button.clicked.connect(self.on_find_mode_clicked)
        self.find_mode_button.setMaximumWidth(100)
        pid_layout.addWidget(self.find_mode_button, 5, 0, 1, 2)

        # Auto mode finder checkbox
        pid_layout.addWidget(QLabel("Auto mode finder:"), 5, 2)
        self.auto_mode_finder_checkbox = QCheckBox()
        self.auto_mode_finder_checkbox.setChecked(False)
        self.auto_mode_finder_checkbox.stateChanged.connect(self.on_auto_mode_finder_changed)
        pid_layout.addWidget(self.auto_mode_finder_checkbox, 5, 3)

        pid_group.setLayout(pid_layout)
        layout.addWidget(pid_group)

        # Slow Offset Control Group (moved to top priority)
        slow_offset_group = QGroupBox("Slow Offset Control (Aux3)")
        slow_offset_layout = QGridLayout()
        
        # Slow Offset Voltage - main control
        slow_offset_layout.addWidget(QLabel("Total Offset (V):"), 0, 0)
        self.slow_offset_spinbox = QDoubleSpinBox()
        self.slow_offset_spinbox.setRange(1.5, 6.5)  # Changed range to 1.5V-6.5V
        self.slow_offset_spinbox.setDecimals(3)
        self.slow_offset_spinbox.setSingleStep(0.01)
        self.slow_offset_spinbox.setKeyboardTracking(False)
        self.slow_offset_spinbox.valueChanged.connect(self.on_slow_offset_changed)
        slow_offset_layout.addWidget(self.slow_offset_spinbox, 0, 1)

        # Rough adjustment slider
        self.slow_offset_slider = QSlider(Qt.Horizontal)
        self.slow_offset_slider.setRange(150, 650)  # 1.5V to 6.5V with 0.01V resolution
        self.slow_offset_slider.setTickPosition(QSlider.TicksBelow)
        self.slow_offset_slider.setTickInterval(100)  # Ticks every 1V
        self.slow_offset_slider.valueChanged.connect(self.on_slow_offset_slider_changed)
        slow_offset_layout.addWidget(self.slow_offset_slider, 1, 0, 1, 2)
        
        # Fine adjustment slider
        slow_offset_layout.addWidget(QLabel("Fine Adjustment (mV):"), 2, 0)
        self.slow_offset_fine_slider = QSlider(Qt.Horizontal)
        self.slow_offset_fine_slider.setRange(-25, 25)  # -25mV to +25mV fine adjustment
        self.slow_offset_fine_slider.setValue(0)
        self.slow_offset_fine_slider.setTickPosition(QSlider.TicksBelow)
        self.slow_offset_fine_slider.setTickInterval(5)  # Ticks every 5mV
        self.slow_offset_fine_slider.valueChanged.connect(self.on_slow_offset_fine_changed)
        slow_offset_layout.addWidget(self.slow_offset_fine_slider, 2, 1)
        
        # Fine adjustment value display
        self.slow_offset_fine_label = QLabel("0.0 mV")
        slow_offset_layout.addWidget(self.slow_offset_fine_label, 3, 1)
        
        slow_offset_group.setLayout(slow_offset_layout)
        layout.addWidget(slow_offset_group)
        
        # Output Settings Group (moved below slow offset)
        output_group = QGroupBox("Output Settings")
        output_layout = QGridLayout()

        # Output Signal Offset (now shows total including fine adjustment)
        output_layout.addWidget(QLabel("Total Output (V):"), 0, 0)
        self.offset_spinbox = QDoubleSpinBox()
        self.offset_spinbox.setRange(0, 1.0)  # Changed from 5.0 to 1.0
        # Default to 0.5V instead of 2V for better starting point in new range
        self.offset_spinbox.setValue(0.5)
        self.offset_spinbox.setDecimals(3)
        self.offset_spinbox.setSingleStep(0.01)
        self.offset_spinbox.setKeyboardTracking(False)
        self.offset_spinbox.valueChanged.connect(self.on_offset_changed)
        output_layout.addWidget(self.offset_spinbox, 0, 1)

        # Add a slider for visual control of offset 
        self.offset_slider = QSlider(Qt.Horizontal)
        self.offset_slider.setRange(0, 100)  # Changed from 500 to 100 for 0-1V with 0.01V resolution
        self.offset_slider.setValue(50)  # Default to 0.5V (matching spinbox)
        self.offset_slider.valueChanged.connect(self.on_offset_slider_changed)
        output_layout.addWidget(self.offset_slider, 1, 0, 1, 2)

        # Fine offset adjustment
        output_layout.addWidget(QLabel("Fine Adjustment (mV):"), 2, 0)
        self.fine_offset_slider = QSlider(Qt.Horizontal)
        self.fine_offset_slider.setRange(-25, 25)  # -25mV to +25mV (each step is 1mV)
        self.fine_offset_slider.setValue(0)
        self.fine_offset_slider.setTickPosition(QSlider.TicksBelow)
        self.fine_offset_slider.setTickInterval(5)  # Ticks at -25, -20, ..., 20, 25 mV
        self.fine_offset_slider.valueChanged.connect(self.on_fine_offset_slider_changed)
        output_layout.addWidget(self.fine_offset_slider, 2, 1)
        
        # Fine offset value display
        self.fine_offset_label = QLabel("0.0 mV")
        output_layout.addWidget(self.fine_offset_label, 3, 1)
        
        # Store the base offset (without fine adjustment) as an instance variable
        self.base_offset = 0.5  # Default to 0.5V instead of 2.0V

        # Remove the separate Total Output display since spinbox now shows total

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Add stretch to push controls to the top
        layout.addStretch()

        # Set initial state of offset based on PID enable
        self.update_offset_spinbox_state()

        return widget
    
    def create_pid_controls(self):
        """Create PID controller controls"""
        widget = QWidget()
        
        # Use a QVBoxLayout to arrange groups vertically
        layout = QVBoxLayout(widget)
        
        # PID Parameters Group
        pid_group = QGroupBox("PID Parameters")
        pid_layout = QGridLayout()

        # P Gain
        pid_layout.addWidget(QLabel("P Gain:"), 0, 0)
        self.p_gain_spinbox = QDoubleSpinBox()
        self.p_gain_spinbox.setRange(-1000000, 1000000)
        self.p_gain_spinbox.setValue(0.0)
        self.p_gain_spinbox.setDecimals(3)
        self.p_gain_spinbox.setSingleStep(0.1)
        self.p_gain_spinbox.setKeyboardTracking(False)  # Only update when Enter is pressed
        self.p_gain_spinbox.valueChanged.connect(self.on_p_gain_changed)
        pid_layout.addWidget(self.p_gain_spinbox, 0, 1)

        # Start V (for mode finding) - next to P Gain
        pid_layout.addWidget(QLabel("Mode Finding Start (V):"), 0, 2)
        self.start_v_spinbox = QDoubleSpinBox()
        self.start_v_spinbox.setRange(1.5, 6.5)
        self.start_v_spinbox.setValue(2.5)
        self.start_v_spinbox.setDecimals(2)
        self.start_v_spinbox.setSingleStep(0.1)
        self.start_v_spinbox.setKeyboardTracking(False)
        pid_layout.addWidget(self.start_v_spinbox, 0, 3)

        # I Gain
        pid_layout.addWidget(QLabel("I Gain:"), 1, 0)
        self.i_gain_spinbox = QDoubleSpinBox()
        self.i_gain_spinbox.setRange(-1000000, 1000000)
        self.i_gain_spinbox.setValue(0.0)
        self.i_gain_spinbox.setDecimals(3)
        self.i_gain_spinbox.setSingleStep(0.1)
        self.i_gain_spinbox.setKeyboardTracking(False)  # Only update when Enter is pressed
        self.i_gain_spinbox.valueChanged.connect(self.on_i_gain_changed)
        pid_layout.addWidget(self.i_gain_spinbox, 1, 1)

        # Stop V (for mode finding) - next to I Gain
        pid_layout.addWidget(QLabel("Mode Finding Stop (V):"), 1, 2)
        self.stop_v_spinbox = QDoubleSpinBox()
        self.stop_v_spinbox.setRange(1.5, 6.5)
        self.stop_v_spinbox.setValue(5.5)
        self.stop_v_spinbox.setDecimals(2)
        self.stop_v_spinbox.setSingleStep(0.1)
        self.stop_v_spinbox.setKeyboardTracking(False)
        pid_layout.addWidget(self.stop_v_spinbox, 1, 3)

        # Bandwidth
        pid_layout.addWidget(QLabel("Bandwidth (Hz):"), 2, 0)
        self.bandwidth_spinbox = QDoubleSpinBox()
        self.bandwidth_spinbox.setRange(0.1, 1000000)
        self.bandwidth_spinbox.setValue(100.0)
        self.bandwidth_spinbox.setDecimals(1)
        self.bandwidth_spinbox.setSingleStep(10)
        self.bandwidth_spinbox.setKeyboardTracking(False)  # Only update when Enter is pressed
        self.bandwidth_spinbox.valueChanged.connect(self.on_bandwidth_changed)
        pid_layout.addWidget(self.bandwidth_spinbox, 2, 1)

        # Stop Routine button
        self.stop_mode_button = QPushButton("Stop Routine")
        self.stop_mode_button.clicked.connect(self.on_stop_mode_clicked)
        self.stop_mode_button.setEnabled(False)  # Initially disabled
        self.stop_mode_button.setVisible(False)  # Initially hidden
        pid_layout.addWidget(self.stop_mode_button, 2, 2, 1, 2)

        # PID Enable
        pid_layout.addWidget(QLabel("PID Enable:"), 3, 0)
        pid_enable_layout = QHBoxLayout()
        self.pid_enable_checkbox = QCheckBox()
        self.pid_enable_checkbox.setChecked(False)
        self.pid_enable_checkbox.stateChanged.connect(self.on_pid_enable_changed)
        pid_enable_layout.addWidget(self.pid_enable_checkbox)
        pid_enable_layout.addStretch()
        pid_layout.addLayout(pid_enable_layout, 3, 1)
        
        # Auto Offset Management checkbox
        pid_layout.addWidget(QLabel("Offset adjustment:"), 3, 2)
        self.auto_offset_checkbox = QCheckBox()
        self.auto_offset_checkbox.setChecked(False)
        self.auto_offset_checkbox.stateChanged.connect(self.on_auto_offset_changed)
        pid_layout.addWidget(self.auto_offset_checkbox, 3, 3)
        
        # Keep I Value
        pid_layout.addWidget(QLabel("Keep I Value:"), 4, 0)
        keep_i_layout = QHBoxLayout()
        self.keep_i_checkbox = QCheckBox()
        self.keep_i_checkbox.setChecked(True)
        self.keep_i_checkbox.stateChanged.connect(self.on_keep_i_changed)
        keep_i_layout.addWidget(self.keep_i_checkbox)
        keep_i_layout.addStretch()
        pid_layout.addLayout(keep_i_layout, 4, 1)

        # Monitor reflection checkbox
        pid_layout.addWidget(QLabel("Monitor Reflection:"), 4, 2)
        self.monitor_reflection_checkbox = QCheckBox()
        self.monitor_reflection_checkbox.setChecked(False)
        self.monitor_reflection_checkbox.stateChanged.connect(self.on_monitor_reflection_changed)
        pid_layout.addWidget(self.monitor_reflection_checkbox, 4, 3)

        # Find Mode button
        self.find_mode_button = QPushButton("Find Mode")
        self.find_mode_button.clicked.connect(self.on_find_mode_clicked)
        self.find_mode_button.setMaximumWidth(100)
        pid_layout.addWidget(self.find_mode_button, 5, 0, 1, 2)

        # Auto mode finder checkbox
        pid_layout.addWidget(QLabel("Auto mode finder:"), 5, 2)
        self.auto_mode_finder_checkbox = QCheckBox()
        self.auto_mode_finder_checkbox.setChecked(False)
        self.auto_mode_finder_checkbox.stateChanged.connect(self.on_auto_mode_finder_changed)
        pid_layout.addWidget(self.auto_mode_finder_checkbox, 5, 3)

        pid_group.setLayout(pid_layout)
        layout.addWidget(pid_group)

        # Slow Offset Control Group (moved to top priority)
        slow_offset_group = QGroupBox("Slow Offset Control (Aux3)")
        slow_offset_layout = QGridLayout()
        
        # Slow Offset Voltage - main control
        slow_offset_layout.addWidget(QLabel("Total Offset (V):"), 0, 0)
        self.slow_offset_spinbox = QDoubleSpinBox()
        self.slow_offset_spinbox.setRange(1.5, 6.5)  # Changed range to 1.5V-6.5V
        self.slow_offset_spinbox.setDecimals(3)
        self.slow_offset_spinbox.setSingleStep(0.01)
        self.slow_offset_spinbox.setKeyboardTracking(False)
        self.slow_offset_spinbox.valueChanged.connect(self.on_slow_offset_changed)
        slow_offset_layout.addWidget(self.slow_offset_spinbox, 0, 1)

        # Rough adjustment slider
        self.slow_offset_slider = QSlider(Qt.Horizontal)
        self.slow_offset_slider.setRange(150, 650)  # 1.5V to 6.5V with 0.01V resolution
        self.slow_offset_slider.setTickPosition(QSlider.TicksBelow)
        self.slow_offset_slider.setTickInterval(100)  # Ticks every 1V
        self.slow_offset_slider.valueChanged.connect(self.on_slow_offset_slider_changed)
        slow_offset_layout.addWidget(self.slow_offset_slider, 1, 0, 1, 2)
        
        # Fine adjustment slider
        slow_offset_layout.addWidget(QLabel("Fine Adjustment (mV):"), 2, 0)
        self.slow_offset_fine_slider = QSlider(Qt.Horizontal)
        self.slow_offset_fine_slider.setRange(-25, 25)  # -25mV to +25mV fine adjustment
        self.slow_offset_fine_slider.setValue(0)
        self.slow_offset_fine_slider.setTickPosition(QSlider.TicksBelow)
        self.slow_offset_fine_slider.setTickInterval(5)  # Ticks every 5mV
        self.slow_offset_fine_slider.valueChanged.connect(self.on_slow_offset_fine_changed)
        slow_offset_layout.addWidget(self.slow_offset_fine_slider, 2, 1)
        
        # Fine adjustment value display
        self.slow_offset_fine_label = QLabel("0.0 mV")
        slow_offset_layout.addWidget(self.slow_offset_fine_label, 3, 1)
        
        slow_offset_group.setLayout(slow_offset_layout)
        layout.addWidget(slow_offset_group)
        
        # Output Settings Group (moved below slow offset)
        output_group = QGroupBox("Output Settings")
        output_layout = QGridLayout()

        # Output Signal Offset (now shows total including fine adjustment)
        output_layout.addWidget(QLabel("Total Output (V):"), 0, 0)
        self.offset_spinbox = QDoubleSpinBox()
        self.offset_spinbox.setRange(0, 1.0)  # Changed from 5.0 to 1.0
        # Default to 0.5V instead of 2V for better starting point in new range
        self.offset_spinbox.setValue(0.5)
        self.offset_spinbox.setDecimals(3)
        self.offset_spinbox.setSingleStep(0.01)
        self.offset_spinbox.setKeyboardTracking(False)
        self.offset_spinbox.valueChanged.connect(self.on_offset_changed)
        output_layout.addWidget(self.offset_spinbox, 0, 1)

        # Add a slider for visual control of offset 
        self.offset_slider = QSlider(Qt.Horizontal)
        self.offset_slider.setRange(0, 100)  # Changed from 500 to 100 for 0-1V with 0.01V resolution
        self.offset_slider.setValue(50)  # Default to 0.5V (matching spinbox)
        self.offset_slider.valueChanged.connect(self.on_offset_slider_changed)
        output_layout.addWidget(self.offset_slider, 1, 0, 1, 2)

        # Fine offset adjustment
        output_layout.addWidget(QLabel("Fine Adjustment (mV):"), 2, 0)
        self.fine_offset_slider = QSlider(Qt.Horizontal)
        self.fine_offset_slider.setRange(-25, 25)  # -25mV to +25mV (each step is 1mV)
        self.fine_offset_slider.setValue(0)
        self.fine_offset_slider.setTickPosition(QSlider.TicksBelow)
        self.fine_offset_slider.setTickInterval(5)  # Ticks at -25, -20, ..., 20, 25 mV
        self.fine_offset_slider.valueChanged.connect(self.on_fine_offset_slider_changed)
        output_layout.addWidget(self.fine_offset_slider, 2, 1)
        
        # Fine offset value display
        self.fine_offset_label = QLabel("0.0 mV")
        output_layout.addWidget(self.fine_offset_label, 3, 1)
        
        # Store the base offset (without fine adjustment) as an instance variable
        self.base_offset = 0.5  # Default to 0.5V instead of 2.0V

        # Remove the separate Total Output display since spinbox now shows total

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Add stretch to push controls to the top
        layout.addStretch()

        # Set initial state of offset based on PID enable
        self.update_offset_spinbox_state()

        return widget
    
    def create_fg_controls(self):
        """Create function generator controls"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Function Generator Settings Group
        fg_group = QGroupBox("Function Generator Settings")
        fg_layout = QGridLayout()

        # Waveform Selection
        fg_layout.addWidget(QLabel("Waveform:"), 0, 0)
        self.waveform_combo = QComboBox()
        # Modify the waveform combo box initialization to use class attribute
        self.waveform_combo.addItems(self.WAVEFORMS)
        self.waveform_combo.currentIndexChanged.connect(self.on_waveform_changed)
        fg_layout.addWidget(self.waveform_combo, 0, 1)

        # Amplitude (in mV)
        fg_layout.addWidget(QLabel("Amplitude (mV):"), 1, 0)
        self.amplitude_spinbox = QDoubleSpinBox()
        self.amplitude_spinbox.setRange(0.0, 10000.0)  # 0-10V in mV
        self.amplitude_spinbox.setValue(1000.0)  # Default 1V = 1000mV
        self.amplitude_spinbox.setDecimals(1)
        self.amplitude_spinbox.setSingleStep(1.0)
        self.amplitude_spinbox.setKeyboardTracking(False)
        self.amplitude_spinbox.valueChanged.connect(self.on_amplitude_changed)
        fg_layout.addWidget(self.amplitude_spinbox, 1, 1)

        # Amplitude fine adjustment
        fg_layout.addWidget(QLabel("Fine Adjustment (mV):"), 2, 0)
        self.amplitude_fine_slider = QSlider(Qt.Horizontal)
        self.amplitude_fine_slider.setRange(-50, 50)  # ±50mV adjustment
        self.amplitude_fine_slider.setValue(0)
        self.amplitude_fine_slider.setTickPosition(QSlider.TicksBelow)
        self.amplitude_fine_slider.setTickInterval(10)
        self.amplitude_fine_slider.valueChanged.connect(self.on_amplitude_fine_changed)
        fg_layout.addWidget(self.amplitude_fine_slider, 2, 1)
        
        # Fine adjustment value display
        self.amplitude_fine_label = QLabel("0 mV")
        fg_layout.addWidget(self.amplitude_fine_label, 3, 1)

        # Frequency
        fg_layout.addWidget(QLabel("Frequency (Hz):"), 4, 0)
        self.freq_spinbox = QDoubleSpinBox()
        self.freq_spinbox.setRange(0.01, 1000000)
        self.freq_spinbox.setValue(1000.0)
        self.freq_spinbox.setDecimals(1)
        self.freq_spinbox.setSingleStep(100)
        self.freq_spinbox.setKeyboardTracking(False)  # Only update when Enter is pressed
        self.freq_spinbox.valueChanged.connect(self.on_freq_changed)
        fg_layout.addWidget(self.freq_spinbox, 4, 1)

        # Auto-calculated offset display
        fg_layout.addWidget(QLabel("Total Offset (mV):"), 5, 0)
        self.fg_offset_spinbox = QDoubleSpinBox()
        self.fg_offset_spinbox.setRange(-5000.0, 5000.0)  # ±5V in mV
        self.fg_offset_spinbox.setValue(0.0)
        self.fg_offset_spinbox.setDecimals(1)
        self.fg_offset_spinbox.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.fg_offset_spinbox.setReadOnly(True)
        self.fg_offset_spinbox.setStyleSheet("background-color: #f0f0f0;")
        fg_layout.addWidget(self.fg_offset_spinbox, 5, 1)

        # Output Toggle
        fg_layout.addWidget(QLabel("Output:"), 6, 0)
        self.output_checkbox = QCheckBox("Enabled")
        self.output_checkbox.setChecked(False)
        self.output_checkbox.stateChanged.connect(self.on_output_toggled)
        fg_layout.addWidget(self.output_checkbox, 6, 1)

        fg_group.setLayout(fg_layout)
        layout.addWidget(fg_group)

        # Add stretch to push controls to the top
        layout.addStretch()

        return widget
    
    def create_demod_controls(self):
        """Create demodulation controls"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Dither Tone Group
        dither_group = QGroupBox("Dither Tone")
        dither_layout = QGridLayout()
        
        # Dither Frequency (now in kHz)
        dither_layout.addWidget(QLabel("Frequency (kHz):"), 0, 0)
        self.dither_freq_spinbox = QDoubleSpinBox()
        self.dither_freq_spinbox.setRange(0.0001, 510.0)  # 0.1 Hz to 100 kHz
        self.dither_freq_spinbox.setValue(0.1)  # Default 100 Hz = 0.1 kHz
        self.dither_freq_spinbox.setDecimals(3)
        self.dither_freq_spinbox.setSingleStep(0.1)
        self.dither_freq_spinbox.setKeyboardTracking(False)
        self.dither_freq_spinbox.valueChanged.connect(self.on_dither_freq_changed)
        dither_layout.addWidget(self.dither_freq_spinbox, 0, 1)
        
        # Dither Drive Strength
        dither_layout.addWidget(QLabel("Drive Strength (mV):"), 1, 0)
        self.dither_strength_spinbox = QDoubleSpinBox()
        self.dither_strength_spinbox.setRange(0.0, 1000.0)  # 0-1000 mV range
        self.dither_strength_spinbox.setValue(100.0)  # Default 100 mV
        self.dither_strength_spinbox.setDecimals(3)
        self.dither_strength_spinbox.setSingleStep(1)
        self.dither_strength_spinbox.setKeyboardTracking(False)  # Only update when Enter is pressed
        self.dither_strength_spinbox.valueChanged.connect(self.on_dither_strength_changed)
        dither_layout.addWidget(self.dither_strength_spinbox, 1, 1)
        
        # Enable dither (checkbox)
        dither_layout.addWidget(QLabel("Enable Dither:"), 2, 0)
        self.dither_enable_checkbox = QCheckBox()
        self.dither_enable_checkbox.setChecked(True)
        self.dither_enable_checkbox.stateChanged.connect(self.on_dither_enable_changed)
        dither_layout.addWidget(self.dither_enable_checkbox, 2, 1)

        dither_group.setLayout(dither_layout)
        layout.addWidget(dither_group)
        
        # Demodulation Phase Group
        demod_group = QGroupBox("Demodulation")
        demod_layout = QGridLayout()
        
        # Demodulation Phase
        demod_layout.addWidget(QLabel("Phase (deg):"), 0, 0)
        self.demod_phase_spinbox = QDoubleSpinBox()
        self.demod_phase_spinbox.setRange(-180.0, 180.0)
        self.demod_phase_spinbox.setValue(0.0)
        self.demod_phase_spinbox.setDecimals(1)
        self.demod_phase_spinbox.setSingleStep(1.0)
        self.demod_phase_spinbox.setKeyboardTracking(False)  # Only update when Enter is pressed
        self.demod_phase_spinbox.valueChanged.connect(self.on_demod_phase_changed)
        demod_layout.addWidget(self.demod_phase_spinbox, 0, 1)
        
        # Phase adjustment slider
        self.phase_slider = QSlider(Qt.Horizontal)
        self.phase_slider.setRange(-180, 180)
        self.phase_slider.setValue(0)
        self.phase_slider.valueChanged.connect(self.on_phase_slider_changed)
        demod_layout.addWidget(self.phase_slider, 1, 0, 1, 2)
        
        demod_group.setLayout(demod_layout)
        layout.addWidget(demod_group)
        
        # Add stretch to push controls to the top
        layout.addStretch()
        
        return widget
    
    def create_monitoring_panel(self):
        """Create the monitoring panel"""
        panel = QGroupBox("Status")
        layout = QGridLayout(panel)
        
        # Status labels
        layout.addWidget(QLabel("PID Status:"), 0, 0)
        self.pid_status_label = QLabel("Unlocked")
        self.pid_status_label.setStyleSheet("color: red; font-weight: bold;")
        layout.addWidget(self.pid_status_label, 0, 1)
        
        layout.addWidget(QLabel("Output Value:"), 1, 0)
        self.output_value_label = QLabel("0.000 V")
        layout.addWidget(self.output_value_label, 1, 1)
        
        layout.addWidget(QLabel("Function Generator:"), 0, 2)
        self.fg_status_label = QLabel("Inactive")
        layout.addWidget(self.fg_status_label, 0, 3)
        
        layout.addWidget(QLabel("Dither Status:"), 1, 2)
        self.dither_status_label = QLabel("Enabled")
        self.dither_status_label.setStyleSheet("color: green;")
        layout.addWidget(self.dither_status_label, 1, 3)
        
        # Add reflection signal monitoring
        layout.addWidget(QLabel("Reflection Signal:"), 2, 0)
        self.reflection_label = QLabel("--- V")
        layout.addWidget(self.reflection_label, 2, 1)
        
        # Move offset adjustment status to next row and span two columns for better visibility
        layout.addWidget(QLabel("Offset Adjuster:"), 3, 0)
        self.auto_offset_status_label = QLabel("Idle")
        layout.addWidget(self.auto_offset_status_label, 3, 1, 1, 3)  # Span 3 columns
        
        # Add spacer item to push status labels to the left
        layout.setColumnStretch(4, 1)
        
        return panel
    
    def update_offset_spinbox_state(self):
        """Update the state of the offset spinbox based on PID enable"""
        is_pid_enabled = self.pid_enable_checkbox.isChecked()
        self.offset_spinbox.setEnabled(not is_pid_enabled)
        self.offset_slider.setEnabled(not is_pid_enabled)
        self.fine_offset_slider.setEnabled(not is_pid_enabled)  # Also disable fine adjustment

    @pyqtSlot()
    def update_offset_controls_slot(self):
        """Slot wrapper for thread-safe offset control updates"""
        self.update_offset_spinbox_state()

    def update_status_indicators(self):
        """Update status indicators based on current state"""
        # PID status indicator
        if self.pid_enable_checkbox.isChecked():
            self.pid_status_label.setText("Locked")
            self.pid_status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.pid_status_label.setText("Unlocked")
            self.pid_status_label.setStyleSheet("color: red; font-weight: bold;")
            
        # FG status indicator
        if self.output_checkbox.isChecked():
            self.fg_status_label.setText(f"Active ({self.waveform_combo.currentText()})")
            self.fg_status_label.setStyleSheet("color: green;")
        else:
            self.fg_status_label.setText("Inactive")
            self.fg_status_label.setStyleSheet("color: gray;")
            
        # Dither status indicator
        if self.dither_enable_checkbox.isChecked():
            self.dither_status_label.setText("Enabled")
            self.dither_status_label.setStyleSheet("color: green;")
        else:
            self.dither_status_label.setText("Disabled")
            self.dither_status_label.setStyleSheet("color: gray;")
    
    # Event handlers for slider controls
    @pyqtSlot(int)
    def on_offset_slider_changed(self, value):
        """Handle offset slider change"""
        # Convert slider value (0-100) to voltage (0-1V) - this is the new base offset
        self.base_offset = value / 100.0
        
        # Get current fine adjustment in volts
        fine_offset_v = (self.fine_offset_slider.value() * 0.5) / 1000.0
        
        # Calculate total offset
        total_offset_v = self.base_offset + fine_offset_v
        
        # Update spinbox with total value (without triggering valueChanged signal)
        self.offset_spinbox.blockSignals(True)
        self.offset_spinbox.setValue(total_offset_v)
        self.offset_spinbox.blockSignals(False)
        
        # Apply to device if PID is disabled
        if not self.pid_enable_checkbox.isChecked():
            with self.mdrec_lock:
                self.mdrec.lock_in.set(f'/{self.device_id}/sigouts/0/offset', total_offset_v)
            self.output_value_label.setText(f"{total_offset_v:.3f} V")

    @pyqtSlot(int)
    def on_phase_slider_changed(self, value):
        """Handle phase slider change"""
        self.demod_phase_spinbox.setValue(float(value))
        # No need to call on_demod_phase_changed as the spinbox valueChanged signal will trigger it
    
    # Event handlers for PID controls
    @pyqtSlot(float)
    def on_p_gain_changed(self, value):
        """Handle P gain changed event"""
        self.logger.info(f"P gain changed to {value}")
        with self.mdrec_lock:
            self.mdrec.lock_in.set(f'/{self.device_id}/pids/{self.dither_pid}/p', value)

    @pyqtSlot(float)
    def on_i_gain_changed(self, value):
        """Handle I gain changed event"""
        self.logger.info(f"I gain changed to {value}")
        with self.mdrec_lock:
            self.mdrec.lock_in.set(f'/{self.device_id}/pids/{self.dither_pid}/i', value)

    @pyqtSlot(float)
    def on_bandwidth_changed(self, value):
        """Handle bandwidth changed event"""
        self.logger.info(f"Bandwidth changed to {value}")
        with self.mdrec_lock:
            self.mdrec.lock_in.set(f'/{self.device_id}/pids/{self.dither_pid}/demod/timeconstant', df2tc(value))

    @pyqtSlot(int)
    def on_pid_enable_changed(self, state):
        """Handle PID enable changed event"""
        enabled = state == Qt.Checked
        self.logger.info(f"PID enable changed to {enabled}")
        
        # If enabling PID, recenter the PID output first
        if enabled:
            self.logger.info("Recentering PID output before enabling PID")
            self.recenter_PID_output()
        
        with self.mdrec_lock:
            self.mdrec.lock_in.set(f'/{self.device_id}/pids/{self.dither_pid}/enable', int(enabled))
            
        # Update all offset-related controls
        self.update_offset_spinbox_state()  # Use the existing method for consistent behavior
        
        self.update_status_indicators()
        
        if enabled:
            # Start monitoring offset when PID is enabled
            self.start_offset_monitoring()
        else:
            # Stop monitoring when PID is disabled
            self.stop_offset_monitoring()
            
            # When disabling PID, read current offset from device and update controls
            with self.mdrec_lock:
                response = self.mdrec.lock_in.get(f'/{self.device_id}/sigouts/0/offset')
                offset_value = float(response[self.device_id]['sigouts']['0']['offset']['value'][0])
            
            self.logger.info(f"Setting output offset to {offset_value:.3f} V on PID disable")
            
            # Reset fine offset slider to 0
            self.fine_offset_slider.blockSignals(True)
            self.fine_offset_slider.setValue(0)
            self.fine_offset_label.setText("0.0 mV")
            self.fine_offset_slider.blockSignals(False)
            
            # Set base offset to the current device value
            self.base_offset = offset_value
            
            # Update spinbox to show current offset
            self.offset_spinbox.blockSignals(True)
            self.offset_spinbox.setValue(offset_value)
            self.offset_spinbox.blockSignals(False)
            
            # Update slider to match base offset
            self.offset_slider.blockSignals(True)
            self.offset_slider.setValue(int(offset_value * 100))
            self.offset_slider.blockSignals(False)
            
            # Update status display
            self.output_value_label.setText(f"{offset_value:.3f} V")

    @pyqtSlot(int)
    def on_keep_i_changed(self, state):
        """Handle keep I value changed event"""
        enabled = state == Qt.Checked
        self.logger.info(f"Keep I value changed to {enabled}")
        with self.mdrec_lock:
            self.mdrec.lock_in.set(f'/{self.device_id}/pids/{self.dither_pid}/keepint', int(enabled))

    @pyqtSlot(float)
    def on_offset_changed(self, value):
        """Handle offset changed event - now treats input as total value"""
        if not self.pid_enable_checkbox.isChecked():
            # Apply the total offset directly to the device
            self.logger.info(f"Total offset changed to {value:.3f} V")
            with self.mdrec_lock:
                self.mdrec.lock_in.set(f'/{self.device_id}/sigouts/0/offset', value)
            
            # Reset fine adjustment to 0
            self.fine_offset_slider.blockSignals(True)
            self.fine_offset_slider.setValue(0)
            self.fine_offset_label.setText("0.0 mV")
            self.fine_offset_slider.blockSignals(False)
            
            # The spinbox value becomes the new base offset
            self.base_offset = value
            
            # Update slider to match the base offset
            self.offset_slider.blockSignals(True)
            self.offset_slider.setValue(int(self.base_offset * 100))
            self.offset_slider.blockSignals(False)
            
            # Update status display
            self.output_value_label.setText(f"{value:.3f} V")

    @pyqtSlot(bool)
    def on_lock_toggled(self, checked):
        """Handle lock button toggle"""
        if self.pid_enable_checkbox.isChecked() != checked:
            self.pid_enable_checkbox.setChecked(checked)
        # No need to handle the actual locking as it's done in on_pid_enable_changed
    
    # Event handlers for dither and demod controls
    @pyqtSlot(float)
    def on_dither_freq_changed(self, value):
        """Handle dither frequency changed event (value in kHz)"""
        self.logger.info(f"Dither frequency changed to {value:.3f} kHz")
        # Convert kHz to Hz for device setting
        with self.mdrec_lock:
            self.mdrec.lock_in.set(f'/{self.device_id}/oscs/{self.dither_drive_demod}/freq', value * 1000.0)

    @pyqtSlot(float)
    def on_dither_strength_changed(self, value):
        """Handle dither strength changed event (value in mV)"""
        self.logger.info(f"Dither strength changed to {value:.3f} mV")
        # Convert mV to V for device setting
        with self.mdrec_lock:
            self.mdrec.lock_in.set(f'/{self.device_id}/sigouts/0/amplitudes/{self.dither_drive_demod}', value/1000.0)

    @pyqtSlot(int)
    def on_dither_enable_changed(self, state):
        """Handle dither enable changed event"""
        enabled = state == Qt.Checked
        self.logger.info(f"Dither enable changed to {enabled}")
        with self.mdrec_lock:
            self.mdrec.lock_in.set(f'/{self.device_id}/sigouts/0/enables/{self.dither_drive_demod}', int(enabled))
        self.update_status_indicators()

    @pyqtSlot(float)
    def on_demod_phase_changed(self, value):
        """Handle demodulation phase changed event"""
        self.logger.info(f"Demodulation phase changed to {value:.1f} deg")
        with self.mdrec_lock:
            self.mdrec.lock_in.set(f'/{self.device_id}/demods/{self.dither_in_demod}/phaseshift', value)
            # Fix the phase slider update by extracting the value from the response
            response = self.mdrec.lock_in.get(f'/{self.device_id}/demods/{self.dither_in_demod}/phaseshift')
            phase_value = float(response[self.device_id]['demods'][str(self.dither_in_demod)]['phaseshift']['value'][0])
                
        self.phase_slider.blockSignals(True)
        self.phase_slider.setValue(int(phase_value))
        self.phase_slider.blockSignals(False)
    
    # Event handlers for function generator controls
    @pyqtSlot(int)
    def on_waveform_changed(self, index):
        """Handle waveform selection changed event"""
        waveform = self.WAVEFORMS[index]
        self.logger.info(f"Waveform changed to {waveform}")
        with self.fg_lock:
            self.fg.out_waveform = waveform  # Set waveform using fg's property
        self.update_status_indicators()
    
    @pyqtSlot(float)
    def on_amplitude_changed(self, value_mv):
        """Handle base amplitude changed event (value in mV)"""
        total_amplitude_mv = value_mv + self.amplitude_fine_slider.value()
        value_v = total_amplitude_mv / 1000.0
        self.logger.info(f"Amplitude changed to {total_amplitude_mv:.1f} mV")
        
        with self.fg_lock:
            self.fg.out_amplitude = value_v
            
            # Always set offset regardless of keep_offset_zero value
            offset_mv = 0.0 if self.keep_offset_zero else total_amplitude_mv / 2.0
            offset_v = offset_mv / 1000.0
            self.fg.out_offset = offset_v
                
        self.fg_offset_spinbox.setValue(offset_mv)  # Update display in mV

    @pyqtSlot(int)
    def on_amplitude_fine_changed(self, fine_mv):
        """Handle fine amplitude adjustment changed event"""
        self.amplitude_fine_label.setText(f"{fine_mv:+d} mV")
        total_amplitude_mv = self.amplitude_spinbox.value() + fine_mv
        value_v = total_amplitude_mv / 1000.0
        self.logger.info(f"Fine adjustment: {fine_mv:+d} mV, total amplitude: {total_amplitude_mv:.1f} mV")
        
        with self.fg_lock:
            self.fg.out_amplitude = value_v
            
            # Always set offset regardless of keep_offset_zero value
            offset_mv = 0.0 if self.keep_offset_zero else total_amplitude_mv / 2.0
            offset_v = offset_mv / 1000.0
            self.fg.out_offset = offset_v
                
        self.fg_offset_spinbox.setValue(offset_mv)  # Update display in mV

    @pyqtSlot(float)
    def on_freq_changed(self, value):
        """Handle frequency changed event"""
        self.logger.info(f"Frequency changed to {value:.1f} Hz")
        with self.fg_lock:
            self.fg.out_frequency = value

    @pyqtSlot(int)
    def on_output_toggled(self, state):
        """Handle output toggled event"""
        enabled = state == Qt.Checked
        self.logger.info(f"Output toggled to {enabled}")
        with self.fg_lock:
            self.fg.out = enabled
            self.mdrec.lock_in.set(f'/{self.device_id}/sigouts/0/add', 1 if enabled else 0)
        self.update_status_indicators()

    @pyqtSlot(int)
    def on_fine_offset_slider_changed(self, value):
        """Handle fine offset slider change"""
        fine_offset_mv = value * 0.5
        self.fine_offset_label.setText(f"{fine_offset_mv:+.1f} mV")
        
        # Calculate total offset
        fine_offset_v = fine_offset_mv / 1000.0  # Convert mV to V
        total_offset_v = self.base_offset + fine_offset_v
        
        # Update spinbox with total value (without triggering valueChanged signal)
        self.offset_spinbox.blockSignals(True)
        self.offset_spinbox.setValue(total_offset_v)
        self.offset_spinbox.blockSignals(False)
        
        # Apply to device if PID is disabled
        if not self.pid_enable_checkbox.isChecked():
            self.logger.info(f"Fine adjustment: {fine_offset_mv:+.1f} mV, total offset: {total_offset_v:.3f} V")
            with self.mdrec_lock:
                self.mdrec.lock_in.set(f'/{self.device_id}/sigouts/0/offset', total_offset_v)
            self.output_value_label.setText(f"{total_offset_v:.3f} V")

    # Add event handlers for slow offset control
    @pyqtSlot(float)
    def on_slow_offset_changed(self, value):
        """Handle slow offset value changed event"""
        self.logger.info(f"Slow offset changed to {value:.3f} V")
        with self.mdrec_lock:
            self.mdrec.lock_in.set(f'/{self.device_id}/auxouts/{self.slow_offset}/offset', value)
        
        # Calculate the new base offset by removing the fine adjustment
        fine_offset_v = (self.slow_offset_fine_slider.value() * 0.5) / 1000.0
        self.slow_offset_base = value - fine_offset_v
        
        # Update slider to match new base offset
        self.slow_offset_slider.blockSignals(True)
        self.slow_offset_slider.setValue(int(self.slow_offset_base * 100))
        self.slow_offset_slider.blockSignals(False)
    
    @pyqtSlot(int)
    def on_slow_offset_slider_changed(self, value):
        """Handle slow offset slider change"""
        # Convert slider value (150-650) to voltage (1.5V-6.5V)
        self.slow_offset_base = value / 100.0
        
        # Get current fine adjustment in volts
        fine_offset_v = (self.slow_offset_fine_slider.value() * 0.5) / 1000.0
        
        # Calculate total offset
        total_offset_v = self.slow_offset_base + fine_offset_v
        
        # Update spinbox with total value (without triggering valueChanged signal)
        self.slow_offset_spinbox.blockSignals(True)
        self.slow_offset_spinbox.setValue(total_offset_v)
        self.slow_offset_spinbox.blockSignals(False)
        
        # Apply to device
        self.logger.info(f"Slow offset slider changed to {total_offset_v:.3f} V")
        with self.mdrec_lock:
            self.mdrec.lock_in.set(f'/{self.device_id}/auxouts/{self.slow_offset}/offset', total_offset_v)
    
    @pyqtSlot(int)
    def on_slow_offset_fine_changed(self, value):
        """Handle slow offset fine slider change"""
        # Each step is 0.5mV
        fine_offset_mv = value * 0.5
        self.slow_offset_fine_label.setText(f"{fine_offset_mv:+.1f} mV")
        
        # Calculate total offset
        fine_offset_v = fine_offset_mv / 1000.0  # Convert mV to V
        total_offset_v = self.slow_offset_base + fine_offset_v
        
        # Update spinbox with total value
        self.slow_offset_spinbox.blockSignals(True)
        self.slow_offset_spinbox.setValue(total_offset_v)
        self.slow_offset_spinbox.blockSignals(False)
        
        # Apply to device
        self.logger.info(f"Fine adjustment: {fine_offset_mv:+.1f} mV, total slow offset: {total_offset_v:.3f} V")
        with self.mdrec_lock:
            self.mdrec.lock_in.set(f'/{self.device_id}/auxouts/{self.slow_offset}/offset', total_offset_v)

    @pyqtSlot(int)
    def on_monitor_reflection_changed(self, state):
        """Handle monitor reflection checkbox state change"""
        enabled = state == Qt.Checked
        if enabled:
            self.start_reflection_monitoring()
        else:
            self.stop_reflection_monitoring()
    
    @pyqtSlot(int)
    def on_auto_offset_changed(self, state):
        """Handle auto offset management checkbox state change"""
        enabled = state == Qt.Checked
        if enabled:
            self.start_auto_offset_management()
        else:
            self.stop_auto_offset_management()
    
    @pyqtSlot(int)
    def on_auto_mode_finder_changed(self, state):
        """Handle auto mode finder checkbox state change"""
        enabled = state == Qt.Checked
        if enabled:
            self.start_auto_mode_finder()
        else:
            self.stop_auto_mode_finder()
    
    @pyqtSlot()
    def on_find_mode_clicked(self):
        """Handle find mode button click"""
        self.logger.info("Manual mode finding triggered")
        mode_finding_thread = threading.Thread(target=self.mode_finding_routine, daemon=True)
        mode_finding_thread.start()

    @pyqtSlot()
    def on_stop_mode_clicked(self):
        """Handle stop mode finding button click"""
        self.logger.info("Routine stop requested")
        self.mode_finding_stop_requested = True
        # Use thread-safe method to disable button
        QMetaObject.invokeMethod(self.stop_mode_button, "setEnabled", Qt.QueuedConnection, Q_ARG(bool, False))

    def _update_button_from_thread(self, text=None, visible=None, enabled=None, style=None):
        """Helper method to safely update button properties from background threads"""
        def update():
            """Perform the actual GUI updates in the main thread"""
            if text is not None:
                self.stop_mode_button.setText(text)
            if visible is not None:
                self.stop_mode_button.setVisible(visible)
            if enabled is not None:
                self.stop_mode_button.setEnabled(enabled)
            if style is not None:
                self.stop_mode_button.setStyleSheet(style)
            # Force layout update
            if self.stop_mode_button.parent():
                self.stop_mode_button.parent().layout().activate()

        # Use QTimer to schedule the update in the main thread
        QTimer.singleShot(0, update)

    def start_offset_monitoring(self):
        """Start the background thread for offset monitoring when PID is enabled"""
        if not self.offset_monitor_thread_running:
            self.offset_monitor_thread_running = True
            self.offset_monitor_thread = threading.Thread(target=self._offset_monitor_loop, daemon=True)
            self.offset_monitor_thread.start()
            self.logger.info("Offset monitoring started")
    
    def stop_offset_monitoring(self):
        """Stop the background thread for offset monitoring"""
        if self.offset_monitor_thread_running:
            self.offset_monitor_thread_running = False
            if self.offset_monitor_thread:
                self.offset_monitor_thread.join(timeout=2.0)
            self.logger.info("Offset monitoring stopped")
    
    def _offset_monitor_loop(self):
        """Background thread loop to monitor and update offset spinbox when PID is enabled"""
        while self.offset_monitor_thread_running:
            try:
                # Check PID state at the start of each iteration
                pid_is_enabled = self.pid_enable_checkbox.isChecked()
                
                if pid_is_enabled:
                    # Get current offset from device
                    offset_value = self.get_mdrec_output_offset()
                    
                    # Thread-safe GUI updates
                    QMetaObject.invokeMethod(
                        self.offset_spinbox,
                        "setValue",
                        Qt.QueuedConnection,
                        Q_ARG(float, offset_value)
                    )
                    
                    # Reset fine adjustment to 0
                    QMetaObject.invokeMethod(
                        self.fine_offset_slider,
                        "setValue",
                        Qt.QueuedConnection,
                        Q_ARG(int, 0)
                    )
                    
                    QMetaObject.invokeMethod(
                        self.fine_offset_label,
                        "setText",
                        Qt.QueuedConnection,
                        Q_ARG(str, "0.0 mV")
                    )
                    
                    # Update base offset (this is safe - it's just a float)
                    self.base_offset = offset_value
                    
                    # Update slider
                    QMetaObject.invokeMethod(
                        self.offset_slider,
                        "setValue",
                        Qt.QueuedConnection,
                        Q_ARG(int, int(offset_value * 100))
                    )
                    
                    # Update status display
                    QMetaObject.invokeMethod(
                        self.output_value_label,
                        "setText",
                        Qt.QueuedConnection,
                        Q_ARG(str, f"{offset_value:.3f} V")
                    )
                    
            except Exception as e:
                self.logger.error(f"Error in offset monitor: {e}")
            
            # Sleep for 0.5 seconds, but check frequently if we should stop
            for _ in range(5):  # Check every 0.1s for 0.5 seconds total
                if not self.offset_monitor_thread_running:
                    break
                time.sleep(0.1)

    def start_auto_mode_finder(self):
        """Start the background thread for automatic mode finding"""
        if not self.auto_mode_finder_thread_running:
            self.auto_mode_finder_thread_running = True
            self.auto_mode_finder_thread = threading.Thread(target=self._auto_mode_finder_loop, daemon=True)
            self.auto_mode_finder_thread.start()
            self.logger.info("Auto mode finder started")
    
    def stop_auto_mode_finder(self):
        """Stop the background thread for automatic mode finding"""
        if self.auto_mode_finder_thread_running:
            self.auto_mode_finder_thread_running = False
            if self.auto_mode_finder_thread:
                self.auto_mode_finder_thread.join(timeout=5.0)
            self.logger.info("Auto mode finder stopped")
    
    def _auto_mode_finder_loop(self):
        """Background thread loop to monitor lock status and re-find mode if lost"""
        while self.auto_mode_finder_thread_running:
            try:
                # Check if PID is enabled (should be locked)
                if self.pid_enable_checkbox.isChecked():
                    # Check if cavity is actually locked
                    if not self.is_cavity_locked():
                        for _ in range(10):  # Double-check over 1 second
                            if not self.auto_mode_finder_thread_running:
                                break
                            time.sleep(0.1)
                        if not self.is_cavity_locked():
                            # Check if another routine is already running
                            if self.routine_lock.acquire(blocking=False):
                                # We got the lock, release it and start mode finding
                                self.routine_lock.release()
                                try:
                                    self.logger.info("Lock lost! Starting mode finding routine...")
                                    self.mode_finding_routine()
                                except Exception as e:
                                    self.logger.error(f"Error during mode finding: {str(e)}")
                            else:
                                self.logger.warning("Lock lost but another routine is in progress, will retry later")
            
            except Exception as e:
                self.logger.error(f"Error in auto mode finder loop: {str(e)}")
            
            # Sleep for 5 seconds, but check frequently if we should stop
            for _ in range(50):  # Check every 0.1s for 5 seconds total
                if not self.auto_mode_finder_thread_running:
                    break
                time.sleep(0.1)
    
    def start_auto_offset_management(self):
        """Start the background thread for automatic offset management"""
        if not self.auto_offset_thread_running:
            self.auto_offset_thread_running = True
            self.auto_offset_thread = threading.Thread(target=self._auto_offset_loop, daemon=True)
            self.auto_offset_thread.start()
            self.logger.info("Auto offset management started")
    
    def stop_auto_offset_management(self):
        """Stop the background thread for automatic offset management"""
        if self.auto_offset_thread_running:
            self.auto_offset_thread_running = False
            if self.auto_offset_thread:
                self.auto_offset_thread.join(timeout=3.0)
            self.auto_offset_status_label.setText("Idle")
            self.logger.info("Auto offset management stopped")
    
    def _auto_offset_loop(self):
        """Background thread loop to monitor and adjust offset"""
        while self.auto_offset_thread_running:
            try:
                # Get current output offset
                current_offset = self.get_mdrec_output_offset()
                
                # Check if ramping is needed
                if current_offset < 0.05:
                    self.logger.info(f"Output offset {current_offset:.3f}V below threshold, starting ramp up")
                    self._ramp_slow_offset(direction='up')
                elif current_offset > 0.95:
                    self.logger.info(f"Output offset {current_offset:.3f}V above threshold, starting ramp down")
                    self._ramp_slow_offset(direction='down')
                else:
                    self.auto_offset_status_label.setText(f"Monitoring")
                
            except Exception as e:
                self.logger.error(f"Error in auto offset management: {e}")
                self.auto_offset_status_label.setText("Error")
            
            # Sleep for 1 second, but check frequently if we should stop
            for _ in range(10):
                if not self.auto_offset_thread_running:
                    break
               
                time.sleep(0.1)
    
    
    def _ramp_slow_offset(self, direction='up'):
        """Ramp the slow offset up or down by 15mV in 1mV steps"""
        # Use helper method for thread-safe button updates
        self._update_button_from_thread(
            text="Stop Offset Adjustment",
            visible=True,
            enabled=True,
            style="background-color: #FF6666;"
        )
        
        # Try to acquire the routine lock without blocking
        if not self.routine_lock.acquire(blocking=False):
            self.logger.warning(f"Cannot ramp slow offset - another routine is in progress")
            # Clean up button state
            self._update_button_from_thread(enabled=False, visible=False, style="")
            return
        
        try:
            # Disable slow offset controls during ramping - thread-safe
            QMetaObject.invokeMethod(self.slow_offset_spinbox, "setEnabled", Qt.QueuedConnection, Q_ARG(bool, False))
            QMetaObject.invokeMethod(self.slow_offset_slider, "setEnabled", Qt.QueuedConnection, Q_ARG(bool, False))
            QMetaObject.invokeMethod(self.slow_offset_fine_slider, "setEnabled", Qt.QueuedConnection, Q_ARG(bool, False))

            step = 0.0005  # 0.5mV step
            if direction == 'down':
                step = -step
            
            # Perform 30 steps
            for i in range(30):
                if self.mode_finding_stop_requested or not self.auto_offset_thread_running:
                    self.logger.info("Ramping stopped by user")
                    break
                    
                # Get current slow offset and calculate new value
                current_slow = self.get_mdrec_slow_offset()
                new_slow = max(1.5, min(6.5, current_slow + step))
                
                # Temporarily re-enable controls for set_slow_offset to update GUI
                QMetaObject.invokeMethod(self.slow_offset_spinbox, "setEnabled", Qt.QueuedConnection, Q_ARG(bool, True))
                QMetaObject.invokeMethod(self.slow_offset_slider, "setEnabled", Qt.QueuedConnection, Q_ARG(bool, True))
                QMetaObject.invokeMethod(self.slow_offset_fine_slider, "setEnabled", Qt.QueuedConnection, Q_ARG(bool, True))
                
                self.set_slow_offset(new_slow)
                
                # Re-disable controls
                QMetaObject.invokeMethod(self.slow_offset_spinbox, "setEnabled", Qt.QueuedConnection, Q_ARG(bool, False))
                QMetaObject.invokeMethod(self.slow_offset_slider, "setEnabled", Qt.QueuedConnection, Q_ARG(bool, False))
                QMetaObject.invokeMethod(self.slow_offset_fine_slider, "setEnabled", Qt.QueuedConnection, Q_ARG(bool, False))
                
                # Update status - thread-safe
                direction_text = "up" if direction == 'up' else "down"
                status_text = f"Ramping {direction_text}: {new_slow:.3f}V (step {i+1}/30)"
                QMetaObject.invokeMethod(self.auto_offset_status_label, "setText", Qt.QueuedConnection, Q_ARG(str, status_text))
                self.logger.info(f"Ramping {direction_text}: step {i+1}/30, slow_offset = {new_slow:.3f}V")

                # Wait 3 seconds before next step
                time.sleep(3.0)
            
            # Re-enable slow offset controls after ramping - thread-safe
            QMetaObject.invokeMethod(self.slow_offset_spinbox, "setEnabled", Qt.QueuedConnection, Q_ARG(bool, True))
            QMetaObject.invokeMethod(self.slow_offset_slider, "setEnabled", Qt.QueuedConnection, Q_ARG(bool, True))
            QMetaObject.invokeMethod(self.slow_offset_fine_slider, "setEnabled", Qt.QueuedConnection, Q_ARG(bool, True))
            
            if self.mode_finding_stop_requested:
                self.logger.info("Ramp routine aborted by user")
            else:
                QMetaObject.invokeMethod(self.auto_offset_status_label, "setText", Qt.QueuedConnection, Q_ARG(str, "Ramp complete, monitoring..."))
                self.logger.info(f"Ramping {direction} complete")
        finally:
            self.routine_lock.release()
            # Thread-safe button cleanup - always disable and hide
            self._update_button_from_thread(enabled=False, visible=False, style="")
            self.mode_finding_stop_requested = False

    def start_reflection_monitoring(self):
        """Start the background thread for reflection monitoring"""
        if not self.reflection_thread_running:
            self.reflection_thread_running = True
            self.reflection_thread = threading.Thread(target=self._reflection_monitor_loop, daemon=True)
            self.reflection_thread.start()
            self.logger.info("Reflection monitoring started")
    
    def stop_reflection_monitoring(self):
        """Stop the background thread for reflection monitoring"""
        if self.reflection_thread_running:
            self.reflection_thread_running = False
            if self.reflection_thread:
                self.reflection_thread.join(timeout=3.0)
            self.logger.info("Reflection monitoring stopped")
    
    def _reflection_monitor_loop(self):
        """Background thread loop to monitor reflection signal"""
        while self.reflection_thread_running:
            try:
                mean_val, std_val = self.get_average_reflection()
                # Format message
                if np.abs(mean_val) < 1:
                    message = f"{mean_val/1e-3:.3f} ± {std_val/1e-3:.3f} mV"
                else: 
                    message = f"{mean_val:.3f} ± {std_val:.3f} V"
                self.logger.info(message)
                # Use thread-safe GUI update
                QMetaObject.invokeMethod(
                    self.reflection_label,
                    "setText",
                    Qt.QueuedConnection,
                    Q_ARG(str, message)
                )
            except Exception as e:
                self.logger.error(f"Error reading reflection: {e}")
                # Use thread-safe GUI update for error message too
                QMetaObject.invokeMethod(
                    self.reflection_label,
                    "setText",
                    Qt.QueuedConnection,
                    Q_ARG(str, "Error")
                )
            
            # Sleep for 2 seconds, but check frequently if we should stop
            for _ in range(20):  # Check every 0.1s for 2 seconds total
                if not self.reflection_thread_running:
                    break
                time.sleep(0.1)
    
    def closeEvent(self, event):
        """Handle window close event to clean up threads"""
        self.stop_reflection_monitoring()
        self.stop_auto_offset_management()
        self.stop_auto_mode_finder()
        self.stop_offset_monitoring()
        self.logger.info("Cavity control GUI closed")
        event.accept()

    def number_of_peaks(self, wave):
        """Count number of peaks in the waveform"""
        if (np.max(wave) - np.min(wave)) < self.mid_baseline_threshold:
            return 0  # No signal detected
        
        idxs = peakutils.indexes(-wave, thres=0.5, min_dist=50)
        num_peaks = len(idxs)
        # self.log(f'Number of peaks found: {num_peaks}')
        return num_peaks

    def find_peak_spacing_regularity(self, wave):
        """Find peak spacing using scope data"""
        if (np.max(wave) - np.min(wave)) < self.mid_baseline_threshold:
            return np.inf  # No signal detected
        
        idxs = peakutils.indexes(-wave, thres=0.5, min_dist=50)
        
        # Check if we have enough peaks to calculate spacing
        if len(idxs) < 5:
            # self.log(f'Not enough peaks found: {len(idxs)}')
            return np.inf  # Not enough peaks to calculate regularity
        
        spacings = idxs[1:] - idxs[:-1]
        
        # Check if we have valid spacings
        if len(spacings) == 0 or np.mean(spacings) == 0:
            return np.inf
        
        # self.log(f'Peak spacings (samples): {spacings}')
        return np.std(spacings) / np.mean(spacings)

    def mode_finding_routine(self, step_v=0.01, delay_s=0.1, regularity_threshold=0.25, 
                           fine_step=0.01, fine_regularity_threshold=0.2):
        """Finding the cavity mode"""
        self.mode_finding_stop_requested = False  # Reset flag
        # Use helper method for thread-safe button updates
        self._update_button_from_thread(
            text="Stop Mode Finding",
            visible=True,
            enabled=True,
            style="background-color: #FF6666;"
        )

        # Try to acquire the routine lock without blocking
        if not self.routine_lock.acquire(blocking=False):
            self.logger.warning("Cannot start mode finding - another routine is in progress")
            # Clean up button state properly
            self._update_button_from_thread(enabled=False, visible=False, style="")
            return
        
        try:
            start_v = self.start_v_spinbox.value()
            current_offset = self.offset_spinbox.value()
            stop_v = self.stop_v_spinbox.value()
            self.logger.info('Reading current function generator settings.')
            with self.fg_lock:
                prev_amplitude = self.fg.out_amplitude
                prev_frequency = self.fg.out_frequency

            self.logger.info(f'Function generator current amplitude: {prev_amplitude*1000.0:.1f} mV, frequency: {prev_frequency:.1f} Hz')

            is_pid_enabled = self.pid_enable_checkbox.isChecked()
            is_dither_enabled = self.dither_enable_checkbox.isChecked()
            is_offset_adjust_enabled = self.auto_offset_checkbox.isChecked()
            is_reflection_monitor_enabled = self.monitor_reflection_checkbox.isChecked()
            is_fg_output_enabled = self.output_checkbox.isChecked()
            is_mode_finding_enabled = self.auto_mode_finder_checkbox.isChecked()

            self.logger.info('Temporarily disabling active routines.')
            
            # Disable PID first (uses its own thread-safe method)
            if is_pid_enabled:
                self.disable_pid()
            
            # Use thread-safe GUI updates for other checkboxes
            QMetaObject.invokeMethod(self.dither_enable_checkbox, "setChecked", Qt.QueuedConnection, Q_ARG(bool, False))
            QMetaObject.invokeMethod(self.auto_offset_checkbox, "setChecked", Qt.QueuedConnection, Q_ARG(bool, False))
            QMetaObject.invokeMethod(self.monitor_reflection_checkbox, "setChecked", Qt.QueuedConnection, Q_ARG(bool, False))
            QMetaObject.invokeMethod(self.auto_mode_finder_checkbox, "setChecked", Qt.QueuedConnection, Q_ARG(bool, False))

            self.logger.info('Setting function generator for mode finding.')
            
            # Thread-safe GUI updates for FG settings
            QMetaObject.invokeMethod(self.amplitude_fine_slider, "setValue", Qt.QueuedConnection, Q_ARG(int, 0))
            QMetaObject.invokeMethod(self.amplitude_spinbox, "setValue", Qt.QueuedConnection, Q_ARG(float, self.mode_finding_settings['fg_amplitude_mv']))
            QMetaObject.invokeMethod(self.freq_spinbox, "setValue", Qt.QueuedConnection, Q_ARG(float, self.mode_finding_settings['fg_amplitude_frequency_hz']))
            QMetaObject.invokeMethod(self.output_checkbox, "setChecked", Qt.QueuedConnection, Q_ARG(bool, True))

            # Disable controls during mode finding - thread-safe
            for widget in [self.auto_mode_finder_checkbox, self.find_mode_button, self.dither_enable_checkbox,
                        self.auto_offset_checkbox, self.pid_enable_checkbox, self.monitor_reflection_checkbox,
                        self.amplitude_spinbox, self.amplitude_fine_slider, self.freq_spinbox, self.output_checkbox,
                        self.slow_offset_slider, self.slow_offset_fine_slider, self.slow_offset_spinbox,
                        self.offset_slider, self.fine_offset_slider, self.offset_spinbox]:
                QMetaObject.invokeMethod(widget, "setEnabled", Qt.QueuedConnection, Q_ARG(bool, False))

            self.logger.info('Starting rough alignment phase...\n\n')
            
            # Reset fine adjustment
            QMetaObject.invokeMethod(self.slow_offset_fine_slider, "setValue", Qt.QueuedConnection, Q_ARG(int, 0))
            
            found_mode = False
            wave, dt = self.read_scope_data(length=16384)
            num_peaks = self.number_of_peaks(wave=wave)
            if num_peaks >= 5:
                self.logger.info(f'Initial number of peaks at start offset {current_offset:.3f} V is {num_peaks}, starting regularity check.')
                regularity = self.find_peak_spacing_regularity(wave=wave)
                if regularity < regularity_threshold:
                    self.logger.info(f'Initial regularity threshold met at offset {current_offset:.3f} V (regularity={regularity:.4f}).')
                    found_mode = True
                else:
                    prev_regularity = regularity
                    current_offset = self.slow_offset_spinbox.value()
                    dir = 1 
                    new_offset = current_offset + dir*step_v
                    QMetaObject.invokeMethod(self.slow_offset_spinbox, "setValue", Qt.QueuedConnection, Q_ARG(float, new_offset))
                    wave, dt = self.read_scope_data(length=16384)
                    regularity = self.find_peak_spacing_regularity(wave=wave)
                    if regularity > prev_regularity:
                        dir = -1  # Reverse direction
                    elif regularity < regularity_threshold:
                        self.logger.info(f'Regularity threshold met at offset {current_offset:.3f} V (regularity={regularity:.4f}).')
                        found_mode = True
                    attempts = 0
                    while regularity > regularity_threshold and attempts < 10:
                        if self.mode_finding_stop_requested:
                            self.logger.info("Mode finding stopped by user")
                            break
                        current_offset += dir*step_v
                        QMetaObject.invokeMethod(self.slow_offset_spinbox, "setValue", Qt.QueuedConnection, Q_ARG(float, current_offset))
                        time.sleep(delay_s)
                        wave, dt = self.read_scope_data(length=16384)
                        regularity = self.find_peak_spacing_regularity(wave=wave)
                        if regularity < regularity_threshold:
                            self.logger.info(f'Regularity threshold met at offset {current_offset:.3f} V (regularity={regularity:.4f}).')
                            found_mode = True
                            break
                        attempts += 1

            if not found_mode:
                # Set initial slow offset
                QMetaObject.invokeMethod(self.slow_offset_spinbox, "setValue", Qt.QueuedConnection, Q_ARG(float, start_v))
                current_offset = start_v
                time.sleep(1.0)  # Wait for offset to settle
                while current_offset <= stop_v:
                    if self.mode_finding_stop_requested:
                        self.logger.info("Mode finding stopped by user")
                        break
                        
                    wave, dt = self.read_scope_data(length=16384)
                    regularity = self.find_peak_spacing_regularity(wave=wave)
                    if regularity < regularity_threshold:
                        self.logger.info(f'Regularity threshold met at offset {current_offset:.3f} V (regularity={regularity:.4f}).')
                        found_mode = True
                        break
                    current_offset += step_v
                    QMetaObject.invokeMethod(self.slow_offset_spinbox, "setValue", Qt.QueuedConnection, Q_ARG(float, current_offset))
                    time.sleep(delay_s)

            # Restore amplitude
            QMetaObject.invokeMethod(self.amplitude_spinbox, "setValue", Qt.QueuedConnection, Q_ARG(float, prev_amplitude*1000.0))

            if found_mode:
                self.logger.info(f'Found mode at offset {current_offset:.3f} V')
                self.logger.info('Starting fine alignment phase...\n\n')
                
                initial_offset = self.offset_spinbox.value()
                current_offset = max(0, initial_offset - 0.15)
                QMetaObject.invokeMethod(self.offset_spinbox, "setValue", Qt.QueuedConnection, Q_ARG(float, current_offset))
                time.sleep(0.2)
                
                while current_offset <= min(1.0, initial_offset + 0.15):
                    if self.mode_finding_stop_requested:
                        self.logger.info("Mode finding stopped by user")
                        break
                    wave, dt = self.read_scope_data(length=16384)
                    regularity = self.find_peak_spacing_regularity(wave=wave)
                    if regularity < fine_regularity_threshold:
                        self.logger.info(f'Fine regularity threshold met at offset {current_offset:.3f} V (regularity={regularity:.4f}).')
                        break
                    current_offset += fine_step
                    QMetaObject.invokeMethod(self.offset_spinbox, "setValue", Qt.QueuedConnection, Q_ARG(float, current_offset))
                    time.sleep(delay_s)
            else:
                self.logger.info(f'No mode found between {start_v:.3f} V and {stop_v:.3f} V')
                self.logger.info('Restoring previous settings and re-enabling routines.')

            # Restore settings - thread-safe
            QMetaObject.invokeMethod(self.freq_spinbox, "setValue", Qt.QueuedConnection, Q_ARG(float, prev_frequency))
            
            if not is_fg_output_enabled:
                QMetaObject.invokeMethod(self.output_checkbox, "setChecked", Qt.QueuedConnection, Q_ARG(bool, False))
            if is_dither_enabled:
                QMetaObject.invokeMethod(self.dither_enable_checkbox, "setChecked", Qt.QueuedConnection, Q_ARG(bool, True))
            if is_offset_adjust_enabled:
                QMetaObject.invokeMethod(self.auto_offset_checkbox, "setChecked", Qt.QueuedConnection, Q_ARG(bool, True))
            if is_reflection_monitor_enabled:
                QMetaObject.invokeMethod(self.monitor_reflection_checkbox, "setChecked", Qt.QueuedConnection, Q_ARG(bool, True))
            if is_mode_finding_enabled:
                QMetaObject.invokeMethod(self.auto_mode_finder_checkbox, "setChecked", Qt.QueuedConnection, Q_ARG(bool, True))
            
            # Re-enable PID last (uses its own thread-safe method)
            if is_pid_enabled:
                self.enable_pid()
                # Explicitly start offset monitoring after enabling PID
                self.start_offset_monitoring()

        finally:
            self.routine_lock.release()
            # Thread-safe button cleanup - always disable and hide
            self._update_button_from_thread(enabled=False, visible=False, style="")
            self.mode_finding_stop_requested = False

            # Enable back controls - thread-safe
            for widget in [self.auto_mode_finder_checkbox, self.find_mode_button, self.dither_enable_checkbox,
                        self.auto_offset_checkbox, self.pid_enable_checkbox, self.monitor_reflection_checkbox,
                        self.amplitude_spinbox, self.amplitude_fine_slider, self.freq_spinbox, self.output_checkbox,
                        self.slow_offset_slider, self.slow_offset_fine_slider, self.slow_offset_spinbox,
                        self.offset_slider, self.fine_offset_slider, self.offset_spinbox]:
                QMetaObject.invokeMethod(widget, "setEnabled", Qt.QueuedConnection, Q_ARG(bool, True))

            # Create a function to update controls after re-enabling everything
            def final_state_update():
                # Update offset control state based on final PID status
                self.update_offset_spinbox_state()
                # Force update of enabled states if PID is enabled
                if self.pid_enable_checkbox.isChecked():
                    self.offset_spinbox.setEnabled(False)
                    self.offset_slider.setEnabled(False)
                    self.fine_offset_slider.setEnabled(False)
            
            # Queue the final state update to run after all other GUI updates
            QTimer.singleShot(100, final_state_update)

    def is_cavity_locked(self):
        """Check if the cavity is locked based on reflection signal"""
        mean_val, std_val = self.get_average_reflection(length=16384)
        return (np.abs(mean_val) < self.locked_reflection_threshold)

    def get_average_reflection(self, length=4096, inputselect=9, sampling=9):
        """Get average reflection signal from the device"""
        # Don't acquire lock here - read_scope_data will handle it
        wave, dt = self.read_scope_data(length=length, inputselect=inputselect, sampling=sampling)
        return np.mean(wave), np.std(wave)

    def read_scope_data(self, length=4096, inputselect=9, sampling=9):
        """Read and log current scope data from the device"""
        settings = self.read_scope_settings()  # Save current settings
        with self.mdrec_lock:
            self.mdrec.lock_in.set(f'/{self.device_id}/scopes/0/time', sampling)
            self.mdrec.lock_in.set(f'/{self.device_id}/scopes/0/length', length)
            self.mdrec.lock_in.set(f'/{self.device_id}/scopes/0/channels/0/inputselect', inputselect)
            data = get_data_scope(self.mdrec, self.device_id)
            dt = data[f'/{self.device_id}/scopes/0/wave'][-1][0]['dt']
            wave = data[f'/{self.device_id}/scopes/0/wave'][-1][0]['wave'][0]
        # Restore previous settings
        self.set_scope_settings(settings)
        return wave, dt

    def read_scope_settings(self):
        """Read and log current scope settings from the device"""
        with self.mdrec_lock:
            sampling = self.mdrec.lock_in.getInt(f'/{self.device_id}/scopes/0/time')
            length = self.mdrec.lock_in.getInt(f'/{self.device_id}/scopes/0/length')
            inputselect = self.mdrec.lock_in.getInt(f'/{self.device_id}/scopes/0/channels/0/inputselect')
            settings = {
                'sampling': sampling,
                'length': length,
                'inputselect': inputselect
            }
            #self.log(f"Scope settings: {settings}")
            return settings

    def set_scope_settings(self, settings):
        """Set scope settings on the device"""
        with self.mdrec_lock:
            if 'sampling' in settings.keys():
                self.mdrec.lock_in.setInt(f'/{self.device_id}/scopes/0/time', settings['sampling'])
            if 'length' in settings.keys():
                self.mdrec.lock_in.setInt(f'/{self.device_id}/scopes/0/length', settings['length'])
            if 'inputselect' in settings.keys():
                self.mdrec.lock_in.setInt(f'/{self.device_id}/scopes/0/channels/0/inputselect', settings['inputselect'])
            #self.log(f"Scope settings updated to: {settings}")

    def log(self, message):
        """Log message if verbose mode is enabled - thread-safe
        
        DEPRECATED: Use self.logger instead for new code. This method is kept for backwards compatibility.
        """
        # Just forward to the logger - it handles everything including file output and GUI updates
        self.logger.info(message)


# Example usage:
def main(mdrec=None, fg=None, device_id=None, dither_pid=None, dither_drive_demod=None, 
         dither_in_demod=None, verbose=False, mdrec_lock=None, fg_lock=None, slow_offset=2, 
         keep_offset_zero=True, mode_finding_settings=None, logfile=None):
    """Main function to run the Cavity Control GUI"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = CavityControlGUI(mdrec=mdrec, fg=fg, device_id=device_id, dither_pid=dither_pid,
                              dither_drive_demod=dither_drive_demod, dither_in_demod=dither_in_demod,
                              verbose=verbose, mdrec_lock=mdrec_lock, fg_lock=fg_lock, logfile=logfile,
                              slow_offset=slow_offset, keep_offset_zero=keep_offset_zero, mode_finding_settings=mode_finding_settings)
    window.show()
    sys.exit(app.exec_())


def main_entry():
    from experiment_interface.zhinst_utils.demodulation_recorder import zhinst_demod_recorder
    from experiment_interface.control.device.function_generator import SingleOutput
    import threading

    verbose = False
    keep_offset_zero = True

    ip_PC = '10.21.217.191'
    ip_fg = '10.21.217.150'
    device_id = 'dev30794'

    pid_dither = 1
    dither_drive_demod = 2
    dither_in_demod = 3
    slow_offset = 2  # auxout2 corresponds to Aux3 on device

    mode_finding_settings = {
        'fg_amplitude_mv': 1000.0, 
        'fg_amplitude_frequency_hz': 120.0
    }
    
    # Use VISA resource string for function generator
    visa_resource_fg = f"TCPIP0::{ip_fg}::inst0::INSTR"

    # Create locks for thread safety
    mdrec_lock = threading.Lock()
    fg_lock = threading.Lock()
    
    mdrec = zhinst_demod_recorder(ip_PC, devtype='MFLI')
    fg = SingleOutput(visa_resource_fg)

    logpath = Path(os.path.join(".", "cavity_logs"))
    # Replace colons in timestamp to avoid invalid filename on Windows
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile = logpath / f"cavity_control_{timestamp}.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)  # Create log directory if it doesn't exist

    try:
        main(mdrec=mdrec, fg=fg, device_id=device_id, dither_pid=pid_dither,    
            dither_drive_demod=dither_drive_demod, dither_in_demod=dither_in_demod,
            verbose=verbose, mdrec_lock=mdrec_lock, fg_lock=fg_lock, logfile=logfile,
            slow_offset=slow_offset, keep_offset_zero=keep_offset_zero, mode_finding_settings=mode_finding_settings)
    except Exception as e:
        # Use logging for error handling too
        logging.error(f"Error running Cavity Control GUI: {e}", exc_info=True)


if __name__ == "__main__":
    main_entry()
