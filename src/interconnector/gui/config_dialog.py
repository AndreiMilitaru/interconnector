"""
Configuration Dialog for Mach-Zehnder Control
Author: GitHub Copilot (based on requirements by Andrei Militaru)
Date: October 2025
Description: PyQt5 configuration dialog for Mach-Zehnder interferometer control.
"""

from PyQt5.QtWidgets import (QDialog, QLabel, QLineEdit, QPushButton, 
                            QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, 
                            QFormLayout, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
from pathlib import Path

class ConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MZ Configuration")
        self.result = None
        
        # Default values
        self.default_ip = "10.21.217.191"
        self.default_device = "MFLI"
        self.default_config = "./config/mach_zehnder/"
        self.default_interval = "0.1"
        
        self._create_widgets()
        self._center_window()
        
    def _create_widgets(self):
        layout = QVBoxLayout(self)
        
        # Add dummy mode checkbox at the top
        self.dummy_checkbox = QCheckBox("Dummy Mode")
        layout.addWidget(self.dummy_checkbox)
        
        # Create form for input fields
        form_layout = QFormLayout()
        
        # IP Address
        self.ip_entry = QLineEdit(self.default_ip)
        form_layout.addRow("IP Address:", self.ip_entry)
        
        # Device Type
        self.device_combo = QComboBox()
        self.device_combo.addItems(["MFLI", "HF2LI"])
        self.device_combo.setCurrentText(self.default_device)
        form_layout.addRow("Device Type:", self.device_combo)
        
        # Config File with browse button
        config_layout = QHBoxLayout()
        self.config_entry = QLineEdit(self.default_config)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._browse_config)
        config_layout.addWidget(self.config_entry)
        config_layout.addWidget(browse_button)
        form_layout.addRow("Config File:", config_layout)
        
        # Lock Check Interval
        self.interval_entry = QLineEdit(self.default_interval)
        form_layout.addRow("Check Interval (s):", self.interval_entry)
        
        layout.addLayout(form_layout)
        
        # OK/Cancel buttons
        buttons_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self._on_ok)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self._on_cancel)
        
        buttons_layout.addWidget(ok_button)
        buttons_layout.addWidget(cancel_button)
        layout.addLayout(buttons_layout)
        
    def _browse_config(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Config File",
            "",
            "YAML files (*.yaml);;All Files (*.*)"
        )
        if filename:
            self.config_entry.setText(filename)
    
    def _center_window(self):
        # Center the window on the screen
        self.setFixedSize(self.sizeHint())
        geometry = self.frameGeometry()
        center_point = QDialog().screen().availableGeometry().center()
        geometry.moveCenter(center_point)
        self.move(geometry.topLeft())
    
    def _on_ok(self):
        try:
            # Only validate IP and device if not in dummy mode
            if not self.dummy_checkbox.isChecked():
                if not self.ip_entry.text().strip():
                    raise ValueError("IP address cannot be empty")
                if self.device_combo.currentText() not in ["MFLI", "HF2LI"]:
                    raise ValueError("Invalid device type")
            
            try:
                interval = float(self.interval_entry.text())
                if interval <= 0:
                    raise ValueError("Interval must be positive")
            except ValueError:
                raise ValueError("Interval must be a valid number")
            
            # Store validated results
            self.result = {
                'dummy_mode': self.dummy_checkbox.isChecked(),
                'ip': self.ip_entry.text().strip(),
                'device_type': self.device_combo.currentText(),
                'config_path': self.config_entry.text().strip(),
                'interval': interval
            }
            self.accept()
            
        except ValueError as e:
            QMessageBox.critical(self, "Input Error", str(e))

    def _on_cancel(self):
        self.reject()
