"""
Manual Cavity Voltage Control GUI - Simplified

Author: Andrei Militaru with contributions from Github Copilot
Date: 3rd February 2026
"""

import sys
import socket
import subprocess
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QSlider, QDoubleSpinBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from interconnector.control.device.piezo_control_client import ControllerClient
from datetime import datetime


class CavityControlGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cavity Control")
        self.setGeometry(100, 100, 500, 250)
        
        # Connect to hardware
        self.controller = None
        self.server_process = None  # Track server subprocess if we start it
        self.coarse_voltage = 0.0
        self.fine_voltage = 0.0
        
        # Initialize UI
        self.init_ui()
        
        # Connect to controller and read initial position
        self.connect_hardware()
        
    def init_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Status label
        self.status_label = QLabel("Connecting...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 10pt; color: gray;")
        main_layout.addWidget(self.status_label)
        
        # Current voltage display
        self.voltage_display = QLabel("0.000 V")
        display_font = QFont()
        display_font.setPointSize(20)
        display_font.setBold(True)
        self.voltage_display.setFont(display_font)
        self.voltage_display.setAlignment(Qt.AlignCenter)
        self.voltage_display.setStyleSheet("color: #2196F3;")
        main_layout.addWidget(self.voltage_display)
        
        # Coarse slider (0-110V in 0.01V steps)
        coarse_layout = QHBoxLayout()
        coarse_label = QLabel("Coarse (0-110V):")
        coarse_label.setMinimumWidth(120)
        
        self.coarse_slider = QSlider(Qt.Horizontal)
        self.coarse_slider.setMinimum(0)
        self.coarse_slider.setMaximum(11000)  # 0-110V in 0.01V steps
        self.coarse_slider.setValue(0)
        self.coarse_slider.valueChanged.connect(self.on_coarse_changed)
        
        coarse_layout.addWidget(coarse_label)
        coarse_layout.addWidget(self.coarse_slider)
        main_layout.addLayout(coarse_layout)
        
        # Fine slider (-5 to +5V in 1mV steps)
        fine_layout = QHBoxLayout()
        fine_label = QLabel("Fine (+/-5V):")
        fine_label.setMinimumWidth(120)
        
        self.fine_slider = QSlider(Qt.Horizontal)
        self.fine_slider.setMinimum(-5000)  # -5V in 1mV steps
        self.fine_slider.setMaximum(5000)   # +5V in 1mV steps
        self.fine_slider.setValue(0)
        self.fine_slider.valueChanged.connect(self.on_fine_changed)
        
        fine_layout.addWidget(fine_label)
        fine_layout.addWidget(self.fine_slider)
        main_layout.addLayout(fine_layout)
        
        # SpinBox for direct input
        spinbox_layout = QHBoxLayout()
        spinbox_label = QLabel("Set (V):")
        spinbox_label.setMinimumWidth(120)
        
        self.voltage_spinbox = QDoubleSpinBox()
        self.voltage_spinbox.setMinimum(0.0)
        self.voltage_spinbox.setMaximum(110.0)
        self.voltage_spinbox.setSingleStep(0.001)
        self.voltage_spinbox.setDecimals(3)
        self.voltage_spinbox.setValue(0.0)
        self.voltage_spinbox.setSuffix(" V")
        self.voltage_spinbox.setMinimumWidth(150)
        self.voltage_spinbox.valueChanged.connect(self.on_spinbox_changed)
        
        spinbox_layout.addWidget(spinbox_label)
        spinbox_layout.addWidget(self.voltage_spinbox)
        spinbox_layout.addStretch()
        main_layout.addLayout(spinbox_layout)
        
    def ensure_server_running(self):
        """Check if server is running, start it if not."""
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
        self.status_label.setText("Starting server...")
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
    
    def connect_hardware(self):
        """Connect to cavity controller and read initial position."""
        try:
            # Check if piezo control server is running, start it if not
            self.ensure_server_running()
            
            self.controller = ControllerClient()
            
            # Read initial voltage
            initial_voltage = self.controller.get_z_voltage()
            self.coarse_voltage = initial_voltage
            self.fine_voltage = 0.0
            
            # Update UI with initial value
            self.update_display(update_hardware=False)
            
            self.status_label.setText(f"Connected | Initial: {initial_voltage:.3f}V")
            self.status_label.setStyleSheet("color: green; font-weight: bold; font-size: 10pt;")
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Connected to cavity controller")
            print(f"[{timestamp}] Initial voltage: {initial_voltage:.3f}V")
            
        except Exception as e:
            self.status_label.setText(f"Connection Failed: {e}")
            self.status_label.setStyleSheet("color: red; font-weight: bold; font-size: 10pt;")
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] ERROR: Failed to connect: {e}")
    
    def on_coarse_changed(self, value):
        """Handle coarse slider change."""
        self.coarse_voltage = value / 100.0  # Convert to voltage (0.01V resolution)
        self.update_display()
    
    def on_fine_changed(self, value):
        """Handle fine slider change."""
        self.fine_voltage = value / 1000.0  # Convert to voltage (1mV resolution)
        self.update_display()
    
    def on_spinbox_changed(self, value):
        """Handle spinbox value change."""
        # Set coarse to the main value, reset fine to 0
        self.coarse_voltage = value
        self.fine_voltage = 0.0
        
        # Update sliders
        self.coarse_slider.blockSignals(True)
        self.coarse_slider.setValue(int(self.coarse_voltage * 100))
        self.coarse_slider.blockSignals(False)
        
        self.fine_slider.blockSignals(True)
        self.fine_slider.setValue(0)
        self.fine_slider.blockSignals(False)
        
        self.update_display()
    
    def update_display(self, update_hardware=True):
        """Update voltage display and hardware."""
        # Calculate total voltage
        total_voltage = self.coarse_voltage + self.fine_voltage
        
        # Clamp to valid range
        total_voltage = max(0.0, min(110.0, total_voltage))
        
        # Update hardware if requested
        if update_hardware and self.controller is not None:
            try:
                self.controller.set_z_voltage(total_voltage)
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Set: {total_voltage:.3f}V (Coarse: {self.coarse_voltage:.3f}V, Fine: {self.fine_voltage:+.3f}V)")
                
            except Exception as e:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] ERROR: Failed to set voltage: {e}")
                self.status_label.setText(f"Set Failed: {e}")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
                return
        
        # Update display
        self.voltage_display.setText(f"{total_voltage:.3f} V")
        
        # Update spinbox
        self.voltage_spinbox.blockSignals(True)
        self.voltage_spinbox.setValue(total_voltage)
        self.voltage_spinbox.blockSignals(False)
        
        # Update coarse slider if needed (in case fine pushed it out of bounds)
        if not update_hardware:
            self.coarse_slider.blockSignals(True)
            self.coarse_slider.setValue(int(self.coarse_voltage * 100))
            self.coarse_slider.blockSignals(False)
    
    def closeEvent(self, event):
        """Handle window close event."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_voltage = self.coarse_voltage + self.fine_voltage
        print(f"[{timestamp}] Closing GUI - final voltage: {total_voltage:.3f}V")
        
        # Terminate server if we started it
        if self.server_process:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Stopping piezo control server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
        
        event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    gui = CavityControlGUI()
    gui.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
