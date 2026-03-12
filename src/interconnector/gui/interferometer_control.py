"""
Interferometer Control GUI
Author: Andrei Militaru with GitHub Copilot
Organization: Institute of Science and Technology Austria (ISTA)
Date: February 2026

Description: GUI for controlling the Mach-Zehnder interferometer lock via
two PIDs (piezo PID 0 and laser PID 3) on the MFLI lock-in amplifier.
Designed to run alongside cavity_control.py sharing the same mdrec device.
"""

import sys
import os
import threading
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

from interconnector.mach_zehnder_utils.mach_zehnder_lock import df2tc
from interconnector.zhinst_utils.scope_settings import get_data_scope
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QDoubleSpinBox, QCheckBox, QPushButton,
    QGroupBox, QGridLayout, QTextEdit
)
from PyQt5.QtCore import Qt, pyqtSlot, QMetaObject, Q_ARG


class QTextEditLogger(logging.Handler):
    """Custom logging handler that emits to a QTextEdit widget"""
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit

    def emit(self, record):
        msg = self.format(record)
        QMetaObject.invokeMethod(
            self.text_edit,
            "append",
            Qt.QueuedConnection,
            Q_ARG(str, msg)
        )
        QMetaObject.invokeMethod(
            self.text_edit.verticalScrollBar(),
            "setValue",
            Qt.QueuedConnection,
            Q_ARG(int, self.text_edit.verticalScrollBar().maximum())
        )


class InterferometerControlGUI(QMainWindow):
    """GUI for interferometer lock control using two PIDs on the MFLI."""

    def __init__(self, mdrec=None, device_id=None, piezo_pid=0, laser_pid=3,
                 mdrec_lock=None, verbose=False, logfile=None, parent=None):
        super().__init__(parent)

        # Logging setup
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                      datefmt='%Y-%m-%d_%H-%M-%S')

        if logfile is not None:
            logfile.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(logfile)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        if verbose:
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        self.verbose = verbose
        self.mdrec = mdrec
        self.device_id = device_id
        self.piezo_pid = piezo_pid
        self.laser_pid = laser_pid
        self.mdrec_lock = mdrec_lock

        # Initialize reflection monitoring thread control
        self.reflection_thread = None
        self.reflection_thread_running = False

        self.init_ui()

        # Attach GUI log handler after log_text_edit is created
        gui_handler = QTextEditLogger(self.log_text_edit)
        gui_handler.setFormatter(formatter)
        self.logger.addHandler(gui_handler)

        # Read initial values from device
        self.set_initial_values_from_device()

    # ------------------------------------------------------------------
    # Device read helpers
    # ------------------------------------------------------------------
    def _get_pid_param(self, pid, param):
        """Generic helper to read a PID parameter."""
        with self.mdrec_lock:
            resp = self.mdrec.lock_in.get(
                f'/{self.device_id}/pids/{pid}/{param}')
            return float(resp[self.device_id]['pids'][str(pid)][param]['value'][0])

    def _set_pid_param(self, pid, param, value):
        """Generic helper to write a PID parameter."""
        with self.mdrec_lock:
            self.mdrec.lock_in.set(
                f'/{self.device_id}/pids/{pid}/{param}', value)

    def get_pid_enabled(self, pid):
        return self._get_pid_param(pid, 'enable') == 1

    def get_pid_p(self, pid):
        return self._get_pid_param(pid, 'p')

    def get_pid_i(self, pid):
        return self._get_pid_param(pid, 'i')

    def get_pid_setpoint(self, pid):
        return self._get_pid_param(pid, 'setpoint')

    def get_pid_bandwidth(self, pid):
        """Return bandwidth in kHz (read timeconstant, convert via df2tc inverse)."""
        with self.mdrec_lock:
            resp = self.mdrec.lock_in.get(
                f'/{self.device_id}/pids/{pid}/demod/timeconstant')
            tc = float(resp[self.device_id]['pids'][str(pid)]['demod']['timeconstant']['value'][0])
        # df2tc converts freq(Hz) -> tc ; inverse: freq = 1/(2*pi*tc)
        freq_hz = 1.0 / (2.0 * np.pi * tc) if tc > 0 else 0.0
        return freq_hz / 1000.0  # return in kHz

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def init_ui(self):
        self.setWindowTitle('Interferometer Control')
        self.setGeometry(600, 50, 380, 520)

        central = QWidget()
        layout = QVBoxLayout(central)

        layout.addWidget(self._create_control_group())
        layout.addWidget(self._create_parameters_group())
        layout.addWidget(self._create_viewing_group())
        layout.addWidget(self._create_calibration_group())
        layout.addWidget(self._create_log_group())
        layout.addStretch()

        self.setCentralWidget(central)

    # ---- Control section ----
    def _create_control_group(self):
        group = QGroupBox("Control")
        grid = QGridLayout()

        # PID Enable (controls both PIDs simultaneously)
        grid.addWidget(QLabel("PID Enable:"), 0, 0)
        self.pid_enable_checkbox = QCheckBox()
        self.pid_enable_checkbox.stateChanged.connect(self.on_pid_enable_changed)
        grid.addWidget(self.pid_enable_checkbox, 0, 1)

        # Setpoint (V) - same value applied to both PIDs
        grid.addWidget(QLabel("Setpoint (V):"), 1, 0)
        self.setpoint_spinbox = QDoubleSpinBox()
        self.setpoint_spinbox.setRange(-10.0, 10.0)
        self.setpoint_spinbox.setDecimals(4)
        self.setpoint_spinbox.setSingleStep(0.001)
        self.setpoint_spinbox.setKeyboardTracking(False)
        self.setpoint_spinbox.valueChanged.connect(self.on_setpoint_changed)
        grid.addWidget(self.setpoint_spinbox, 1, 1)

        # Bandwidth (kHz) - same value applied to both PIDs
        grid.addWidget(QLabel("Bandwidth (kHz):"), 2, 0)
        self.bandwidth_spinbox = QDoubleSpinBox()
        self.bandwidth_spinbox.setRange(0.001, 500.0)
        self.bandwidth_spinbox.setDecimals(3)
        self.bandwidth_spinbox.setSingleStep(0.1)
        self.bandwidth_spinbox.setKeyboardTracking(False)
        self.bandwidth_spinbox.valueChanged.connect(self.on_bandwidth_changed)
        grid.addWidget(self.bandwidth_spinbox, 2, 1)

        # Monitor Reflection checkbox
        grid.addWidget(QLabel("Monitor Reflection:"), 3, 0)
        self.monitor_reflection_checkbox = QCheckBox()
        self.monitor_reflection_checkbox.setChecked(False)
        self.monitor_reflection_checkbox.stateChanged.connect(self.on_monitor_reflection_changed)
        grid.addWidget(self.monitor_reflection_checkbox, 3, 1)

        group.setLayout(grid)
        return group

    # ---- Parameters section ----
    def _create_parameters_group(self):
        group = QGroupBox("PID Parameters")
        grid = QGridLayout()

        # Piezo PID
        grid.addWidget(QLabel("Piezo P:"), 0, 0)
        self.piezo_p_spinbox = QDoubleSpinBox()
        self.piezo_p_spinbox.setRange(-1e6, 1e6)
        self.piezo_p_spinbox.setDecimals(3)
        self.piezo_p_spinbox.setSingleStep(0.1)
        self.piezo_p_spinbox.setKeyboardTracking(False)
        self.piezo_p_spinbox.valueChanged.connect(self.on_piezo_p_changed)
        grid.addWidget(self.piezo_p_spinbox, 0, 1)

        grid.addWidget(QLabel("Piezo I:"), 1, 0)
        self.piezo_i_spinbox = QDoubleSpinBox()
        self.piezo_i_spinbox.setRange(-1e6, 1e6)
        self.piezo_i_spinbox.setDecimals(3)
        self.piezo_i_spinbox.setSingleStep(0.1)
        self.piezo_i_spinbox.setKeyboardTracking(False)
        self.piezo_i_spinbox.valueChanged.connect(self.on_piezo_i_changed)
        grid.addWidget(self.piezo_i_spinbox, 1, 1)

        # Laser PID
        grid.addWidget(QLabel("Laser P:"), 0, 2)
        self.laser_p_spinbox = QDoubleSpinBox()
        self.laser_p_spinbox.setRange(-1e6, 1e6)
        self.laser_p_spinbox.setDecimals(3)
        self.laser_p_spinbox.setSingleStep(0.1)
        self.laser_p_spinbox.setKeyboardTracking(False)
        self.laser_p_spinbox.valueChanged.connect(self.on_laser_p_changed)
        grid.addWidget(self.laser_p_spinbox, 0, 3)

        grid.addWidget(QLabel("Laser I:"), 1, 2)
        self.laser_i_spinbox = QDoubleSpinBox()
        self.laser_i_spinbox.setRange(-1e6, 1e6)
        self.laser_i_spinbox.setDecimals(3)
        self.laser_i_spinbox.setSingleStep(0.1)
        self.laser_i_spinbox.setKeyboardTracking(False)
        self.laser_i_spinbox.valueChanged.connect(self.on_laser_i_changed)
        grid.addWidget(self.laser_i_spinbox, 1, 3)

        self.btn_flip_pi_signs = QPushButton("Flip PI Signs")
        self.btn_flip_pi_signs.setToolTip(
            "Negate P and I gains of both the piezo and laser PIDs")
        self.btn_flip_pi_signs.clicked.connect(self.on_flip_pi_signs_clicked)
        grid.addWidget(self.btn_flip_pi_signs, 2, 0, 1, 4)

        group.setLayout(grid)
        return group

    # ---- Viewing section ----
    def _create_viewing_group(self):
        group = QGroupBox("Scope Input")
        hlay = QHBoxLayout()

        self.btn_interferometer = QPushButton("Interferometer")
        self.btn_interferometer.setCheckable(True)
        self.btn_interferometer.clicked.connect(self.on_interferometer_clicked)
        hlay.addWidget(self.btn_interferometer)

        self.btn_cavity = QPushButton("Cavity")
        self.btn_cavity.setCheckable(True)
        self.btn_cavity.clicked.connect(self.on_cavity_clicked)
        hlay.addWidget(self.btn_cavity)

        group.setLayout(hlay)
        return group

    # ---- Calibration section ----
    def _create_calibration_group(self):
        group = QGroupBox("Phase Calibration")
        grid = QGridLayout()

        grid.addWidget(QLabel("Min V:"), 0, 0)
        self.min_v_spinbox = QDoubleSpinBox()
        self.min_v_spinbox.setRange(-10.0, 10.0)
        self.min_v_spinbox.setDecimals(4)
        self.min_v_spinbox.setSingleStep(0.001)
        self.min_v_spinbox.setKeyboardTracking(False)
        self.min_v_spinbox.valueChanged.connect(self.update_calibration_output)
        grid.addWidget(self.min_v_spinbox, 0, 1)

        grid.addWidget(QLabel("Max V:"), 0, 2)
        self.max_v_spinbox = QDoubleSpinBox()
        self.max_v_spinbox.setRange(-10.0, 10.0)
        self.max_v_spinbox.setDecimals(4)
        self.max_v_spinbox.setSingleStep(0.001)
        self.max_v_spinbox.setKeyboardTracking(False)
        self.max_v_spinbox.valueChanged.connect(self.update_calibration_output)
        grid.addWidget(self.max_v_spinbox, 0, 3)

        grid.addWidget(QLabel("Angle (deg):"), 1, 0)
        self.angle_spinbox = QDoubleSpinBox()
        self.angle_spinbox.setRange(0.0, 360.0)
        self.angle_spinbox.setDecimals(2)
        self.angle_spinbox.setSingleStep(1.0)
        self.angle_spinbox.setKeyboardTracking(False)
        self.angle_spinbox.valueChanged.connect(self.update_calibration_output)
        grid.addWidget(self.angle_spinbox, 1, 1)

        grid.addWidget(QLabel("Voltage:"), 1, 2)
        self.calibration_output = QDoubleSpinBox()
        self.calibration_output.setReadOnly(True)
        self.calibration_output.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.calibration_output.setRange(-10.0, 10.0)
        self.calibration_output.setDecimals(4)
        self.calibration_output.setStyleSheet("background-color: #f0f0f0;")
        grid.addWidget(self.calibration_output, 1, 3)

        group.setLayout(grid)
        return group

    # ---- Log section ----
    def _create_log_group(self):
        group = QGroupBox("Log")
        layout = QVBoxLayout()

        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setMaximumHeight(120)
        self.log_text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                font-family: Consolas, Monaco, monospace;
                font-size: 8pt;
            }
        """)
        layout.addWidget(self.log_text_edit)

        clear_btn = QPushButton("Clear Log")
        clear_btn.setMaximumWidth(80)
        clear_btn.clicked.connect(self.log_text_edit.clear)
        layout.addWidget(clear_btn)

        group.setLayout(layout)
        return group

    # ------------------------------------------------------------------
    # Initialisation from device
    # ------------------------------------------------------------------
    def set_initial_values_from_device(self):
        """Read current values from both PIDs and populate widgets."""
        # Block all signals during init
        widgets = [self.pid_enable_checkbox, self.setpoint_spinbox,
                   self.bandwidth_spinbox, self.piezo_p_spinbox,
                   self.piezo_i_spinbox, self.laser_p_spinbox,
                   self.laser_i_spinbox]
        for w in widgets:
            w.blockSignals(True)

        try:
            # Read from piezo PID (use as reference for shared params)
            pid_enabled = self.get_pid_enabled(self.piezo_pid)
            setpoint = self.get_pid_setpoint(self.piezo_pid)
            bandwidth = self.get_pid_bandwidth(self.piezo_pid)
            piezo_p = self.get_pid_p(self.piezo_pid)
            piezo_i = self.get_pid_i(self.piezo_pid)
            laser_p = self.get_pid_p(self.laser_pid)
            laser_i = self.get_pid_i(self.laser_pid)
            
            self.pid_enable_checkbox.setChecked(pid_enabled)
            self.setpoint_spinbox.setValue(setpoint)
            self.bandwidth_spinbox.setValue(bandwidth)
            self.piezo_p_spinbox.setValue(piezo_p)
            self.piezo_i_spinbox.setValue(piezo_i)
            self.laser_p_spinbox.setValue(laser_p)
            self.laser_i_spinbox.setValue(laser_i)

            # Log initial values
            self.logger.info(f"Initial values loaded from device: PID Enabled={pid_enabled}, Setpoint={setpoint:.4f} V, Bandwidth={bandwidth:.3f} kHz")
            self.logger.info(f"Initial PID parameters: Piezo P={piezo_p}, Piezo I={piezo_i}, Laser P={laser_p}, Laser I={laser_i}")
        except Exception as e:
            self.logger.warning(f"Failed to read initial values: {e}")

        for w in widgets:
            w.blockSignals(False)

        # Set initial scope button highlighting
        self._update_scope_buttons()

    def _update_scope_buttons(self):
        """Highlight the active scope input button."""
        try:
            with self.mdrec_lock:
                current = self.mdrec.lock_in.getInt(
                    f'/{self.device_id}/scopes/0/channels/0/inputselect')
        except Exception:
            current = -1

        self.btn_interferometer.setChecked(current == 0)
        self.btn_cavity.setChecked(current == 9)

    def is_interferometer_locked(self):
        """Check if the interferometer is locked based on reflection signal"""
        mean_val, std_val = self.get_average_reflection()
        is_locked = (np.abs(mean_val) < self.locked_reflection_threshold)
        
        # If locked, update mode finding start/stop values around current slow offset
        if is_locked:
            current_slow_offset = self.slow_offset_spinbox.value()
            # Set start to 50mV (0.05V) before current value
            start_value = max(1.5, current_slow_offset - 0.05)
            # Set stop to 50mV (0.05V) after current value
            stop_value = min(6.5, current_slow_offset + 0.05)
            
            # Update spinboxes (thread-safe)
            QMetaObject.invokeMethod(self.start_v_spinbox, "setValue", Qt.QueuedConnection, Q_ARG(float, start_value))
            QMetaObject.invokeMethod(self.stop_v_spinbox, "setValue", Qt.QueuedConnection, Q_ARG(float, stop_value))
        
        return is_locked

    def get_average_reflection(self, length=None, inputselect=0, sampling=None):
        """Get average reflection signal from the device"""
        # Don't acquire lock here - read_scope_data will handle it
        wave, dt = self.read_scope_data(length=length, inputselect=inputselect, sampling=sampling)
        self.logger.info(f"Reflection waveform: mean={np.mean(wave):.3f} V, std={np.std(wave):.3f} V")
        return np.mean(wave), np.std(wave)

    def read_scope_data(self, length=None, inputselect=0, sampling=None):
        """Read and log current scope data from the device"""
        settings = self.read_scope_settings()  # Save current settings
        with self.mdrec_lock:
            self.mdrec.lock_in.set(f'/{self.device_id}/scopes/0/time', sampling)
            self.mdrec.lock_in.set(f'/{self.device_id}/scopes/0/length', length if length is not None else settings['length'])
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

    # ------------------------------------------------------------------
    # Event handlers - Control
    # ------------------------------------------------------------------
    @pyqtSlot(int)
    def on_pid_enable_changed(self, state):
        enabled = state == Qt.Checked
        self.logger.info(f"PID enable -> {enabled}  (piezo PID {self.piezo_pid} + laser PID {self.laser_pid})")
        self._set_pid_param(self.piezo_pid, 'enable', int(enabled))
        self._set_pid_param(self.laser_pid, 'enable', int(enabled))

    @pyqtSlot(float)
    def on_setpoint_changed(self, value):
        self.logger.info(f"Setpoint -> {value:.4f} V  (both PIDs)")
        self._set_pid_param(self.piezo_pid, 'setpoint', value)
        self._set_pid_param(self.laser_pid, 'setpoint', value)

    @pyqtSlot(float)
    def on_bandwidth_changed(self, value_khz):
        freq_hz = value_khz * 1000.0
        tc = df2tc(freq_hz)
        self.logger.info(f"Bandwidth -> {value_khz:.3f} kHz  (tc={tc:.6e} s, both PIDs)")
        with self.mdrec_lock:
            self.mdrec.lock_in.set(
                f'/{self.device_id}/pids/{self.piezo_pid}/demod/timeconstant', tc)
            self.mdrec.lock_in.set(
                f'/{self.device_id}/pids/{self.laser_pid}/demod/timeconstant', tc)

    # ------------------------------------------------------------------
    # Event handlers - Parameters
    # ------------------------------------------------------------------
    @pyqtSlot(float)
    def on_piezo_p_changed(self, value):
        self.logger.info(f"Piezo P -> {value}")
        self._set_pid_param(self.piezo_pid, 'p', value)

    @pyqtSlot(float)
    def on_piezo_i_changed(self, value):
        self.logger.info(f"Piezo I -> {value}")
        self._set_pid_param(self.piezo_pid, 'i', value)

    @pyqtSlot(float)
    def on_laser_p_changed(self, value):
        self.logger.info(f"Laser P -> {value}")
        self._set_pid_param(self.laser_pid, 'p', value)

    @pyqtSlot(float)
    def on_laser_i_changed(self, value):
        self.logger.info(f"Laser I -> {value}")
        self._set_pid_param(self.laser_pid, 'i', value)

    @pyqtSlot()
    def on_flip_pi_signs_clicked(self):
        """Negate P and I gains of both PIDs in spinboxes and on the device."""
        for spinbox, pid, param in [
            (self.piezo_p_spinbox, self.piezo_pid, 'p'),
            (self.piezo_i_spinbox, self.piezo_pid, 'i'),
            (self.laser_p_spinbox, self.laser_pid, 'p'),
            (self.laser_i_spinbox, self.laser_pid, 'i'),
        ]:
            new_val = -spinbox.value()
            spinbox.blockSignals(True)
            spinbox.setValue(new_val)
            spinbox.blockSignals(False)
            self._set_pid_param(pid, param, new_val)
        self.logger.info(
            f"PI signs flipped: "
            f"Piezo P={self.piezo_p_spinbox.value():.3f}, "
            f"Piezo I={self.piezo_i_spinbox.value():.3f}, "
            f"Laser P={self.laser_p_spinbox.value():.3f}, "
            f"Laser I={self.laser_i_spinbox.value():.3f}"
        )

    @pyqtSlot(int)
    def on_monitor_reflection_changed(self, state):
        """Handle reflection monitoring checkbox state change."""
        if state == Qt.Checked:
            self.start_reflection_monitoring()
        else:
            self.stop_reflection_monitoring()

    # ------------------------------------------------------------------
    # Event handlers - Viewing
    # ------------------------------------------------------------------
    @pyqtSlot()
    def on_interferometer_clicked(self):
        self.logger.info("Scope input -> 0 (Interferometer)")
        with self.mdrec_lock:
            self.mdrec.lock_in.setInt(
                f'/{self.device_id}/scopes/0/channels/0/inputselect', 0)
        self.btn_interferometer.setChecked(True)
        self.btn_cavity.setChecked(False)

    @pyqtSlot()
    def on_cavity_clicked(self):
        self.logger.info("Scope input -> 9 (Cavity)")
        with self.mdrec_lock:
            self.mdrec.lock_in.setInt(
                f'/{self.device_id}/scopes/0/channels/0/inputselect', 9)
        self.btn_interferometer.setChecked(False)
        self.btn_cavity.setChecked(True)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    @pyqtSlot()
    def update_calibration_output(self):
        """Compute voltage from angle using sinusoidal fringe model.

        Model: V(angle) = midpoint - amplitude * cos(angle)
        where midpoint = (maxV + minV) / 2, amplitude = (maxV - minV) / 2.
        0 deg corresponds to minV, 180 deg to maxV.
        """
        min_v = self.min_v_spinbox.value()
        max_v = self.max_v_spinbox.value()
        angle_deg = self.angle_spinbox.value()

        midpoint = (max_v + min_v) / 2.0
        amplitude = (max_v - min_v) / 2.0
        angle_rad = np.radians(angle_deg)

        # cos(0)=1 -> min_v ; cos(180)=-1 -> max_v  ->  V = mid - amp*cos(angle)
        voltage = midpoint - amplitude * np.cos(angle_rad)

        self.calibration_output.setValue(voltage)

    # ------------------------------------------------------------------
    # Reflection Monitoring
    # ------------------------------------------------------------------
    def start_reflection_monitoring(self):
        """Start background thread for reflection monitoring."""
        if not self.reflection_thread_running:
            self.reflection_thread_running = True
            self.reflection_thread = threading.Thread(target=self._reflection_monitor_loop, daemon=True)
            self.reflection_thread.start()
            self.logger.info("Reflection monitoring started")

    def stop_reflection_monitoring(self):
        """Stop reflection monitoring thread."""
        if self.reflection_thread_running:
            self.reflection_thread_running = False
            if self.reflection_thread:
                self.reflection_thread.join(timeout=1.0)
            self.logger.info("Reflection monitoring stopped")

    def _reflection_monitor_loop(self):
        """Background thread loop that monitors reflection signal every 5 seconds."""
        import time
        while self.reflection_thread_running:
            try:
                # Read scope data (using interferometer input)
                wave, dt = self.read_scope_data(inputselect=0)
                mean_val = np.mean(wave)
                std_val = np.std(wave)
                
                # Log the reflection state
                self.logger.info(f"Reflection: mean={mean_val:.4f} V, std={std_val:.4f} V")
                
            except Exception as e:
                self.logger.warning(f"Error reading reflection: {e}")
            
            # Wait 5 seconds (check every 0.1s to allow quick shutdown)
            for _ in range(50):
                if not self.reflection_thread_running:
                    break
                time.sleep(0.1)

    def closeEvent(self, event):
        """Stop monitoring threads on window close."""
        self.stop_reflection_monitoring()
        event.accept()


# ======================================================================
# Entry points
# ======================================================================
def main(mdrec=None, device_id=None, piezo_pid=0, laser_pid=3,
         mdrec_lock=None, verbose=False, logfile=None):
    """Run the Interferometer Control GUI standalone."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = InterferometerControlGUI(
        mdrec=mdrec, device_id=device_id, piezo_pid=piezo_pid, laser_pid=laser_pid,
        mdrec_lock=mdrec_lock, verbose=verbose, logfile=logfile)
    window.show()
    sys.exit(app.exec_())


def main_entry():
    """Standalone entry point - creates its own mdrec connection."""
    from interconnector.zhinst_utils.demodulation_recorder import zhinst_demod_recorder

    verbose = False
    ip_PC = '10.21.217.191'
    device_id = 'dev30794'
    
    # PID indices on the MFLI
    piezo_pid = 0
    laser_pid = 3

    mdrec_lock = threading.Lock()
    mdrec = zhinst_demod_recorder(ip_PC, devtype='MFLI')

    logpath = Path(os.path.join(".", "interferometer_logs"))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile = logpath / f"interferometer_control_{timestamp}.log"

    try:
        main(mdrec=mdrec, device_id=device_id, piezo_pid=piezo_pid, laser_pid=laser_pid,
             mdrec_lock=mdrec_lock, verbose=verbose, logfile=logfile)
    except Exception as e:
        logging.error(f"Error running Interferometer Control GUI: {e}",
                      exc_info=True)


if __name__ == "__main__":
    main_entry()
