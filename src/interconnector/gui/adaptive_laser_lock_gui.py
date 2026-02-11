"""
Adaptive Laser Lock GUI - Compact Version

Author: Andrei Militaru with contributions from Github Copilot
Date: 3rd February 2026
"""

import sys
import threading
import subprocess
import time
import socket
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QCheckBox, QGroupBox, 
                              QDoubleSpinBox, QPushButton, QTabWidget, QGridLayout)
from PyQt5.QtCore import Qt, QTimer
from datetime import datetime
import numpy as np

# Import the lock controller
from interconnector.control.lock.adaptive_laser_lock import (
    LockController, initialize_hardware, initialize_log_file, read_laser_offset,
    interpolate, LASER_SHORT_RANGE, LASER_MID_RANGE, LASER_FAR_RANGE,
    LASER_MIN_STEP, LASER_MED_STEP, LASER_MAX_STEP,
    CAVITY_MIN_RANGE, CAVITY_MAX_RANGE, CAVITY_MIN_STEP, CAVITY_MAX_STEP
)

# Import matplotlib
try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: Matplotlib not available")


# Configuration
PLOT_TIME_WINDOW_SECONDS = 600  # 10 minutes rolling window


class AdaptiveLockGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Adaptive Lock Control")
        self.setGeometry(100, 100, 900, 700)
        
        # Hardware connections
        self.laser_socket = None
        self.oscilloscope = None
        self.cavity_controller = None
        self.lock_controller = None
        self.server_process = None  # Track server subprocess if we start it
        
        # Tracked values
        self.laser_offset_tracked = 0.0
        self.cavity_position_tracked = 0.0
        
        # History for plotting (configurable window at ~0.25s updates)
        self.max_history = int(PLOT_TIME_WINDOW_SECONDS / 0.25)  # 2400 points for 600s
        self.time_history = []
        self.laser_offset_history = []
        self.laser_state_history = []
        self.contrast_history = []
        self.cavity_position_history = []
        self.cavity_state_history = []
        self.cavity_rms_history = []
        
        # Thread control
        self.lock_thread = None
        self.lock_running = False
        
        # Initialize UI
        self.init_ui()
        
        # Connect to hardware
        self.connect_hardware()
        
        # Set up update timer (GUI refresh only)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(100)  # Update GUI every 100ms
        
    def init_ui(self):
        """Set up compact user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Title
        title_label = QLabel("Adaptive Lock Control")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Automatic control tab
        auto_tab = QWidget()
        self.init_automatic_tab(auto_tab)
        self.tab_widget.addTab(auto_tab, "Automatic")
        
        # Manual control tab
        manual_tab = QWidget()
        self.init_manual_tab(manual_tab)
        self.tab_widget.addTab(manual_tab, "Manual")
        
        # Status bar
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
    
    def init_automatic_tab(self, tab):
        """Initialize compact automatic lock control tab."""
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Control checkboxes
        control_layout = QHBoxLayout()
        self.laser_lock_checkbox = QCheckBox("Laser Lock")
        self.laser_lock_checkbox.setChecked(True)
        self.laser_lock_checkbox.stateChanged.connect(self.on_laser_lock_toggled)
        control_layout.addWidget(self.laser_lock_checkbox)
        
        self.cavity_lock_checkbox = QCheckBox("Cavity Lock")
        self.cavity_lock_checkbox.setChecked(True)
        self.cavity_lock_checkbox.stateChanged.connect(self.on_cavity_lock_toggled)
        control_layout.addWidget(self.cavity_lock_checkbox)
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Status panels (compact, side by side)
        status_layout = QHBoxLayout()
        
        # Laser status
        laser_group = QGroupBox("Laser")
        laser_layout = QGridLayout()
        laser_layout.addWidget(QLabel("State:"), 0, 0)
        self.laser_state_label = QLabel("INIT")
        laser_layout.addWidget(self.laser_state_label, 0, 1)
        laser_layout.addWidget(QLabel("Contrast:"), 1, 0)
        self.laser_contrast_label = QLabel("0.000")
        laser_layout.addWidget(self.laser_contrast_label, 1, 1)
        laser_layout.addWidget(QLabel("Voltage:"), 2, 0)
        self.laser_voltage_label = QLabel("0.000 V")
        laser_layout.addWidget(self.laser_voltage_label, 2, 1)
        laser_group.setLayout(laser_layout)
        status_layout.addWidget(laser_group)
        
        # Cavity status
        cavity_group = QGroupBox("Cavity")
        cavity_layout = QGridLayout()
        cavity_layout.addWidget(QLabel("State:"), 0, 0)
        self.cavity_state_label = QLabel("INIT")
        cavity_layout.addWidget(self.cavity_state_label, 0, 1)
        cavity_layout.addWidget(QLabel("RMS:"), 1, 0)
        self.cavity_rms_label = QLabel("0.0 mV")
        cavity_layout.addWidget(self.cavity_rms_label, 1, 1)
        cavity_layout.addWidget(QLabel("Voltage:"), 2, 0)
        self.cavity_voltage_label = QLabel("0.00 V")
        cavity_layout.addWidget(self.cavity_voltage_label, 2, 1)
        cavity_group.setLayout(cavity_layout)
        status_layout.addWidget(cavity_group)
        
        layout.addLayout(status_layout)
        
        # Plotting area (compact, 2 plots)
        if PLOTTING_AVAILABLE:
            self.figure = Figure(figsize=(8, 5))
            self.canvas = FigureCanvas(self.figure)
            
            # Two subplots: Laser (top), Cavity (bottom)
            self.ax_laser = self.figure.add_subplot(211)
            self.ax_laser_contrast = self.ax_laser.twinx()
            
            self.ax_cavity = self.figure.add_subplot(212)
            self.ax_cavity_rms = self.ax_cavity.twinx()
            
            # Configure laser plot (Contrast=blue left, Offset=orange right)
            self.ax_laser.set_ylabel('Contrast', color='blue')
            self.ax_laser.tick_params(axis='y', labelcolor='blue')
            self.ax_laser_contrast.set_ylabel('Laser Offset (V)', color='orange')
            self.ax_laser_contrast.tick_params(axis='y', labelcolor='orange')
            self.ax_laser_contrast.yaxis.set_label_position('right')
            self.ax_laser.set_xlabel('Time (s)')
            self.ax_laser.grid(True, alpha=0.3)
            
            # Configure cavity plot (RMS=blue left, Voltage=orange right)
            self.ax_cavity.set_ylabel('RMS (mV)', color='blue')
            self.ax_cavity.tick_params(axis='y', labelcolor='blue')
            self.ax_cavity_rms.set_ylabel('Cavity Voltage (V)', color='orange')
            self.ax_cavity_rms.tick_params(axis='y', labelcolor='orange')
            self.ax_cavity_rms.yaxis.set_label_position('right')
            self.ax_cavity.set_xlabel('Time (s)')
            self.ax_cavity.grid(True, alpha=0.3)
            
            self.figure.tight_layout()
            
            layout.addWidget(self.canvas)
    
    def init_manual_tab(self, tab):
        """Initialize compact manual control tab."""
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Laser control
        laser_group = QGroupBox("Laser Control")
        laser_layout = QHBoxLayout()
        laser_layout.addWidget(QLabel("Offset (V):"))
        
        self.laser_down_btn = QPushButton("◄")
        self.laser_down_btn.setMaximumWidth(30)
        self.laser_down_btn.pressed.connect(lambda: self.start_laser_adjust(-1))
        self.laser_down_btn.released.connect(self.stop_laser_adjust)
        laser_layout.addWidget(self.laser_down_btn)
        
        self.laser_spinbox = QDoubleSpinBox()
        self.laser_spinbox.setRange(30.0, 90.0)
        self.laser_spinbox.setSingleStep(0.001)
        self.laser_spinbox.setDecimals(3)
        self.laser_spinbox.setSuffix(" V")
        self.laser_spinbox.valueChanged.connect(self.on_laser_manual_change)
        laser_layout.addWidget(self.laser_spinbox)
        
        self.laser_up_btn = QPushButton("►")
        self.laser_up_btn.setMaximumWidth(30)
        self.laser_up_btn.pressed.connect(lambda: self.start_laser_adjust(+1))
        self.laser_up_btn.released.connect(self.stop_laser_adjust)
        laser_layout.addWidget(self.laser_up_btn)
        
        laser_layout.addStretch()
        laser_group.setLayout(laser_layout)
        layout.addWidget(laser_group)
        
        # Cavity control
        cavity_group = QGroupBox("Cavity Control")
        cavity_layout = QHBoxLayout()
        cavity_layout.addWidget(QLabel("Position (V):"))
        
        self.cavity_down_btn = QPushButton("◄")
        self.cavity_down_btn.setMaximumWidth(30)
        self.cavity_down_btn.pressed.connect(lambda: self.start_cavity_adjust(-1))
        self.cavity_down_btn.released.connect(self.stop_cavity_adjust)
        cavity_layout.addWidget(self.cavity_down_btn)
        
        self.cavity_spinbox = QDoubleSpinBox()
        self.cavity_spinbox.setRange(0.0, 110.0)
        self.cavity_spinbox.setSingleStep(0.01)
        self.cavity_spinbox.setDecimals(2)
        self.cavity_spinbox.setSuffix(" V")
        self.cavity_spinbox.valueChanged.connect(self.on_cavity_manual_change)
        cavity_layout.addWidget(self.cavity_spinbox)
        
        self.cavity_up_btn = QPushButton("►")
        self.cavity_up_btn.setMaximumWidth(30)
        self.cavity_up_btn.pressed.connect(lambda: self.start_cavity_adjust(+1))
        self.cavity_up_btn.released.connect(self.stop_cavity_adjust)
        cavity_layout.addWidget(self.cavity_up_btn)
        
        cavity_layout.addStretch()
        cavity_group.setLayout(cavity_layout)
        layout.addWidget(cavity_group)
        
        layout.addStretch()
        
        # Timers for continuous adjustment
        self.laser_adjust_timer = QTimer()
        self.laser_adjust_timer.timeout.connect(self.apply_laser_adjust)
        self.laser_adjust_direction = 0
        
        self.cavity_adjust_timer = QTimer()
        self.cavity_adjust_timer.timeout.connect(self.apply_cavity_adjust)
        self.cavity_adjust_direction = 0
    
    def connect_hardware(self):
        """Connect to hardware and initialize lock controller."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.status_label.setText(f"[{timestamp}] Connecting...")
            QApplication.processEvents()
            
            # Check if piezo control server is running, start it if not
            self.ensure_server_running()
            
            # Initialize hardware
            self.laser_socket, self.oscilloscope, self.cavity_controller = initialize_hardware()
            
            # Read initial values
            self.laser_offset_tracked = read_laser_offset(self.laser_socket)
            self.cavity_position_tracked = self.cavity_controller.get_z_voltage()
            
            # Update spinboxes
            self.laser_spinbox.blockSignals(True)
            self.laser_spinbox.setValue(self.laser_offset_tracked)
            self.laser_spinbox.blockSignals(False)
            
            self.cavity_spinbox.blockSignals(True)
            self.cavity_spinbox.setValue(self.cavity_position_tracked)
            self.cavity_spinbox.blockSignals(False)
            
            # Create log file
            log_filepath = initialize_log_file()
            
            # Create step interpolators
            laser_step_interpolator = interpolate.interp1d(
                [-20, LASER_SHORT_RANGE, LASER_MID_RANGE, LASER_FAR_RANGE, 20],
                [LASER_MIN_STEP, LASER_MIN_STEP, LASER_MED_STEP, LASER_MAX_STEP, LASER_MAX_STEP]
            )
            
            cavity_step_interpolator = interpolate.interp1d(
                [0, CAVITY_MIN_RANGE, CAVITY_MAX_RANGE, 5.0],
                [CAVITY_MIN_STEP, CAVITY_MIN_STEP, CAVITY_MAX_STEP, CAVITY_MAX_STEP],
                bounds_error=False,
                fill_value=(CAVITY_MIN_STEP, CAVITY_MAX_STEP)
            )
            
            # Create lock controller
            self.lock_controller = LockController(
                self.laser_socket, self.oscilloscope, self.cavity_controller,
                log_filepath, laser_step_interpolator, cavity_step_interpolator,
                status_callback=self.on_lock_status_update,
                log_callback=self.on_log_message
            )
            
            # Start lock in separate thread
            self.lock_running = True
            self.lock_thread = threading.Thread(target=self.run_lock_thread, daemon=True)
            self.lock_thread.start()
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.status_label.setText(f"[{timestamp}] Connected")
            
        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.status_label.setText(f"[{timestamp}] Error: {e}")
    
    def run_lock_thread(self):
        """Run lock in separate thread (blocking operations isolated from GUI)."""
        self.lock_controller.start()
        
        while self.lock_running:
            self.lock_controller.run_single_iteration()

    def on_lock_status_update(self, status_dict):
        """Callback from lock controller."""
        if status_dict.get('lock_lost', False):
            return
        
        # Update tracked values
        self.laser_offset_tracked = status_dict['laser_offset']
        self.cavity_position_tracked = status_dict['cavity_position']
        
        # Store history (limit to 5 minutes)
        # Use indices instead of absolute time for rolling window
        self.time_history.append(len(self.time_history))  # Just use index
        self.laser_offset_history.append(status_dict['laser_offset'])
        self.laser_state_history.append(status_dict['laser_behavior'])
        self.contrast_history.append(status_dict['contrast'])
        self.cavity_position_history.append(status_dict['cavity_position'])
        self.cavity_state_history.append(status_dict['cavity_behavior'])
        self.cavity_rms_history.append(status_dict['cavity_rms'])
        
        # Trim to max history and reset indices
        if len(self.time_history) > self.max_history:
            self.time_history.pop(0)
            self.laser_offset_history.pop(0)
            self.laser_state_history.pop(0)
            self.contrast_history.pop(0)
            self.cavity_position_history.pop(0)
            self.cavity_state_history.pop(0)
            self.cavity_rms_history.pop(0)
            
            # Reset time_history to start from 0 again
            self.time_history = list(range(len(self.time_history)))

    def on_log_message(self, message):
        """Callback for log messages."""
        print(message)
    
    def update_display(self):
        """Update all display elements (GUI thread only, non-blocking)."""
        if self.lock_controller is None:
            return
        
        # Just update the GUI with current values (no blocking operations)
        
        # Update laser status
        self.laser_contrast_label.setText(f"{self.lock_controller.current_contrast:.3f}")
        self.laser_voltage_label.setText(f"{self.laser_offset_tracked:.4f} V")
        
        if hasattr(self.lock_controller, 'laser_state'):
            _, laser_behavior = self.lock_controller.laser_state.get_adaptive_step_multiplier()
            self.laser_state_label.setText(laser_behavior)
        
        # Update cavity status
        self.cavity_rms_label.setText(f"{self.lock_controller.current_cavity_rms*1000:.1f} mV")
        self.cavity_voltage_label.setText(f"{self.cavity_position_tracked:.2f} V")
        
        if hasattr(self.lock_controller, 'cavity_state'):
            _, cavity_behavior = self.lock_controller.cavity_state.get_adaptive_step_multiplier()
            self.cavity_state_label.setText(cavity_behavior)
        
        # Update manual spinboxes
        laser_locked = self.laser_lock_checkbox.isChecked()
        cavity_locked = self.cavity_lock_checkbox.isChecked()
        
        if laser_locked:
            self.laser_spinbox.blockSignals(True)
            self.laser_spinbox.setValue(self.laser_offset_tracked)
            self.laser_spinbox.blockSignals(False)
            self.laser_spinbox.setEnabled(False)
            self.laser_down_btn.setEnabled(False)
            self.laser_up_btn.setEnabled(False)
        else:
            self.laser_spinbox.setEnabled(True)
            self.laser_down_btn.setEnabled(True)
            self.laser_up_btn.setEnabled(True)
        
        if cavity_locked:
            self.cavity_spinbox.blockSignals(True)
            self.cavity_spinbox.setValue(self.cavity_position_tracked)
            self.cavity_spinbox.blockSignals(False)
            self.cavity_spinbox.setEnabled(False)
            self.cavity_down_btn.setEnabled(False)
            self.cavity_up_btn.setEnabled(False)
        else:
            self.cavity_spinbox.setEnabled(True)
            self.cavity_down_btn.setEnabled(True)
            self.cavity_up_btn.setEnabled(True)
        
        # Update plots
        if PLOTTING_AVAILABLE and len(self.time_history) > 1:
            self.update_plots()
    
    def update_plots(self):
        """Update plots with simplified colors."""
        # THREAD-SAFE: Take snapshot of all history data at once
        try:
            # Copy all lists atomically to avoid race conditions
            time_history = list(self.time_history)
            contrast_history = list(self.contrast_history)
            laser_offset_history = list(self.laser_offset_history)
            cavity_rms_history = list(self.cavity_rms_history)
            cavity_position_history = list(self.cavity_position_history)
            
            # Check if we have enough data
            if len(time_history) < 2:
                return
            
            # Ensure all lists have the same length (take minimum)
            min_len = min(len(time_history), len(contrast_history), 
                         len(laser_offset_history), len(cavity_rms_history),
                         len(cavity_position_history))
            
            time_history = time_history[:min_len]
            contrast_history = contrast_history[:min_len]
            laser_offset_history = laser_offset_history[:min_len]
            cavity_rms_history = cavity_rms_history[:min_len]
            cavity_position_history = cavity_position_history[:min_len]
            
        except Exception as e:
            # If there's any error during snapshot, just skip this update
            return
        
        # Clear all axes
        self.ax_laser.clear()
        self.ax_laser_contrast.clear()
        self.ax_cavity.clear()
        self.ax_cavity_rms.clear()
        
        # Convert indices to time in seconds (0.25s per point)
        time_in_seconds = [t * 0.25 for t in time_history]
        
        # Laser plot (Contrast=blue left, Offset=orange right)
        self.ax_laser.plot(time_in_seconds, contrast_history, color='blue', linewidth=1)
        self.ax_laser_contrast.plot(time_in_seconds, laser_offset_history, color='orange', linewidth=1)
        
        self.ax_laser.set_ylabel('Contrast', color='blue')
        self.ax_laser.tick_params(axis='y', labelcolor='blue')
        self.ax_laser_contrast.set_ylabel('Laser Offset (V)', color='orange')
        self.ax_laser_contrast.tick_params(axis='y', labelcolor='orange')
        self.ax_laser_contrast.yaxis.set_label_position('right')
        self.ax_laser.set_xlabel('Time (s)')
        self.ax_laser.grid(True, alpha=0.3)
        
        # Cavity plot (RMS=blue left, Voltage=orange right)
        cavity_rms_mv = [v*1000 for v in cavity_rms_history]
        self.ax_cavity.plot(time_in_seconds, cavity_rms_mv, color='blue', linewidth=1)
        self.ax_cavity_rms.plot(time_in_seconds, cavity_position_history, color='orange', linewidth=1)
        
        self.ax_cavity.set_ylabel('RMS (mV)', color='blue')
        self.ax_cavity.tick_params(axis='y', labelcolor='blue')
        self.ax_cavity_rms.set_ylabel('Cavity Voltage (V)', color='orange')
        self.ax_cavity_rms.tick_params(axis='y', labelcolor='orange')
        self.ax_cavity_rms.yaxis.set_label_position('right')
        self.ax_cavity.set_xlabel('Time (s)')
        self.ax_cavity.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def on_laser_manual_change(self, value):
        """Handle manual laser offset change."""
        if not self.laser_lock_checkbox.isChecked():
            try:
                from interconnector.control.lock.adaptive_laser_lock import set_laser_offset
                set_laser_offset(self.laser_socket, value)
                self.laser_offset_tracked = value
            except Exception as e:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Error setting laser: {e}")
    
    def on_cavity_manual_change(self, value):
        """Handle manual cavity position change."""
        if not self.cavity_lock_checkbox.isChecked():
            try:
                self.cavity_controller.set_z_voltage(value)
                self.cavity_position_tracked = value
            except Exception as e:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Error setting cavity: {e}")
    
    def on_laser_lock_toggled(self, state):
        """Handle laser lock checkbox toggle."""
        if self.lock_controller:
            enabled = (state == Qt.Checked)
            self.lock_controller.set_laser_enabled(enabled)
    
    def on_cavity_lock_toggled(self, state):
        """Handle cavity lock checkbox toggle."""
        if self.lock_controller:
            enabled = (state == Qt.Checked)
            self.lock_controller.set_cavity_enabled(enabled)
    
    def start_laser_adjust(self, direction):
        """Start continuous laser adjustment."""
        self.laser_adjust_direction = direction
        self.laser_adjust_timer.start(50)
    
    def stop_laser_adjust(self):
        """Stop continuous laser adjustment."""
        self.laser_adjust_timer.stop()
    
    def apply_laser_adjust(self):
        """Apply continuous laser adjustment."""
        current = self.laser_spinbox.value()
        new_value = current + self.laser_adjust_direction * 0.001
        self.laser_spinbox.setValue(max(30.0, min(90.0, new_value)))
    
    def start_cavity_adjust(self, direction):
        """Start continuous cavity adjustment."""
        self.cavity_adjust_direction = direction
        self.cavity_adjust_timer.start(50)
    
    def stop_cavity_adjust(self):
        """Stop continuous cavity adjustment."""
        self.cavity_adjust_timer.stop()
    
    def apply_cavity_adjust(self):
        """Apply continuous cavity adjustment."""
        current = self.cavity_spinbox.value()
        new_value = current + self.cavity_adjust_direction * 0.01
        self.cavity_spinbox.setValue(max(0.0, min(110.0, new_value)))
    
    def ensure_server_running(self):
        """Check if server is running, start it if not."""
        # Try to connect to see if server is already running
        HOST = '10.21.217.17'
        PORT = 65433
        
        try:
            test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_sock.settimeout(0.5)
            test_sock.connect((HOST, PORT))
            test_sock.close()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Server already running")
            return  # Server is already running
        except (socket.error, socket.timeout):
            pass  # Server not running, we'll start it
        
        # Start the server as a subprocess
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Starting piezo control server...")
        self.status_label.setText(f"[{timestamp}] Starting server...")
        QApplication.processEvents()
        
        try:
            # Use python -m to run the module
            self.server_process = subprocess.Popen(
                [sys.executable, '-m', 'interconnector.control.device.piezo_control_server'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
            )
            
            # Wait for server to start (try connecting a few times)
            for attempt in range(10):
                time.sleep(0.5)
                try:
                    test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    test_sock.settimeout(0.5)
                    test_sock.connect((HOST, PORT))
                    test_sock.close()
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] Server started successfully")
                    return  # Server is now running
                except (socket.error, socket.timeout):
                    if self.server_process.poll() is not None:
                        # Process terminated unexpectedly
                        raise Exception(f"Server process terminated with code {self.server_process.returncode}")
                    continue
            
            raise Exception("Server failed to start after 5 seconds")
            
        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Failed to start server: {e}")
            if self.server_process:
                self.server_process.terminate()
                self.server_process = None
            raise
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop lock thread
        self.lock_running = False
        
        if self.lock_controller:
            self.lock_controller.stop()
        
        # Wait for thread to finish
        if self.lock_thread and self.lock_thread.is_alive():
            self.lock_thread.join(timeout=2.0)
        
        # Terminate server if we started it
        if self.server_process:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Stopping piezo control server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] GUI closed")
        event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    gui = AdaptiveLockGUI()
    gui.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
