"""
Independent Mach-Zehnder Control GUI
Author: GitHub Copilot (based on requirements by Andrei Militaru)
Date: October 2025
Description: Self-contained GUI interface for controlling Mach-Zehnder interferometer,
handling its own configuration and initialization of hardware resources.
"""

import sys
import os
import yaml
from datetime import datetime
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton, 
                            QCheckBox, QLineEdit, QGroupBox, QGridLayout, QVBoxLayout, 
                            QHBoxLayout, QFrame, QMessageBox, QToolTip)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QFont

from mach_zehnder_utils.dummy_manager import DummyMZManager
from visualization.mach_zehnder_visualizer import MachZehnderVisualizer

# Try to import hardware-dependent modules
HARDWARE_AVAILABLE = False
try:
    from zhinst_utils.demodulation_recorder import zhinst_demod_recorder
    from control.mach_zehnder_stabilization import MachZehnderManager
    HARDWARE_AVAILABLE = True
except ImportError:
    print("Hardware modules not available. Running in dummy mode.")

class ToolTip:
    """Simple tooltip class for PyQt widgets"""
    def __init__(self, widget, text, delay=1000):
        self.widget = widget
        self.text = text
        self.widget.setToolTip(text)

class MZControlIndyGUI(QMainWindow):
    """Self-contained Mach-Zehnder interferometer control GUI that manages its own
    configuration and hardware initialization."""
    
    def __init__(self):
        """Initialize the GUI with self-contained configuration dialog"""
        super().__init__()
        self.setWindowTitle("Mach-Zehnder Phase Control (Independent)")
        
        # Hide main window initially
        self.hide()
        
        # Initialize manager to None first
        self.manager = None
        self.visualizer = None  # Will be initialized in _check_config_and_continue
        
        # Get configuration
        self._get_configuration()
        
    def _get_configuration(self):
        """Self-contained configuration retrieval and processing"""
        # Default configuration
        config = {
            'dummy_mode': True,
            'interval': 1.0,  # Default interval
            'ip': '',
            'device_type': '',
            'config_path': ''
        }

        # Get the project root directory (parent of gui directory)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Resolve config path
        if not config.get('config_path'):
            # Use default config path relative to project root
            config['config_path'] = os.path.join(project_root, 'config', 'mach_zehnder', 'default_config.yaml')
        elif not os.path.isabs(config['config_path']):
            # If path is relative, make it relative to project root
            config['config_path'] = os.path.join(project_root, config['config_path'])
        
        # Normalize the path for the current OS
        config['config_path'] = os.path.normpath(config['config_path'])
        
        print(f"Resolved config path: {config['config_path']}")

        # Initialize manager based on mode
        try:
            if not config['dummy_mode'] and HARDWARE_AVAILABLE: 
                # Initialize hardware
                self.mdrec = zhinst_demod_recorder(
                    config['ip'],
                    devtype=config['device_type']
                )
                
                # Initialize real manager
                self.manager = MachZehnderManager(
                    self.mdrec,
                    config_path=config['config_path'],
                    lock_check_interval=config['interval']
                )
            else:
                self.manager = DummyMZManager(
                    lock_check_interval=config['interval']
                )
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", f"Failed to initialize manager: {str(e)}")
            sys.exit(1)
        
        if self.manager is None:
            QMessageBox.critical(self, "Error", "Failed to create manager")
            sys.exit(1)
        
        # Initialize visualizer with config path
        config_path = config.get('config_path')
        self.visualizer = MachZehnderVisualizer(config_path)

        self._create_widgets()
        self._center_window()
        
        # Show the main window
        self.show()
        self.raise_()
        self.activateWindow()

    def _create_widgets(self):
        """Create all GUI widgets"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QGridLayout(central_widget)
        
        # Range Calibration Frame (now includes visibility)
        range_frame = QGroupBox("Range Calibration")
        range_layout = QVBoxLayout(range_frame)
        
        range_calib_btn = QPushButton("Range Calibration")
        range_calib_btn.clicked.connect(self._range_calibration)
        range_layout.addWidget(range_calib_btn)
        ToolTip(range_calib_btn, "Calibrate the voltage range of the Mach-Zehnder interferometer\nby scanning and finding minimum and maximum transmission points")
        
        vis_btn = QPushButton("Measure Visibility")
        vis_btn.clicked.connect(self._measure_visibility)
        range_layout.addWidget(vis_btn)
        ToolTip(vis_btn, "Measure the visibility (fringe contrast) of the interferometer\nHigher visibility indicates better interference quality")
        
        # Range calibration results
        self.range_label = QLabel("No range calibration")
        range_layout.addWidget(self.range_label)
        ToolTip(self.range_label, "Shows the calibrated voltage range (Vmin - Vmax)\nThese values define the operating range of the interferometer")
        
        self.range_time = QLabel("")
        range_layout.addWidget(self.range_time)
        
        # Visibility results
        self.vis_label = QLabel("No visibility measurement")
        range_layout.addWidget(self.vis_label)
        ToolTip(self.vis_label, "Visibility value between 0 and 1\nHigher values indicate better fringe contrast and interferometer quality")
        
        self.vis_time = QLabel("")
        range_layout.addWidget(self.vis_time)
        
        # PID Configuration Frame
        pid_frame = QGroupBox("PID Configuration")
        pid_layout = QVBoxLayout(pid_frame)
        
        save_pid_btn = QPushButton("Save PID Config")
        save_pid_btn.clicked.connect(self.manager.save_current_pid_config)
        pid_layout.addWidget(save_pid_btn)
        ToolTip(save_pid_btn, "Save the current PID controller parameters to file\nThis preserves your tuned settings for future use")
        
        load_pid_btn = QPushButton("Load PID Config")
        load_pid_btn.clicked.connect(self._load_pid_config)
        pid_layout.addWidget(load_pid_btn)
        ToolTip(load_pid_btn, "Load previously saved PID parameters\nThis will overwrite current controller settings")
        
        # Lock Quality Frame
        lock_frame = QGroupBox("Lock Quality")
        lock_layout = QVBoxLayout(lock_frame)
        
        eval_lock_btn = QPushButton("Evaluate Lock")
        eval_lock_btn.clicked.connect(self._evaluate_lock)
        lock_layout.addWidget(eval_lock_btn)
        ToolTip(eval_lock_btn, "Evaluate the current lock stability and quality\nLower values indicate more stable phase locking")
        
        self.lock_label = QLabel("No measurement")
        lock_layout.addWidget(self.lock_label)
        ToolTip(self.lock_label, "Lock quality metric: phase standard deviation")
        
        self.lock_time = QLabel("")
        lock_layout.addWidget(self.lock_time)
        
        # Control Frame
        ctrl_frame = QGroupBox("Control")
        ctrl_layout = QVBoxLayout(ctrl_frame)
        
        # Setpoint control
        sp_layout = QHBoxLayout()
        sp_label = QLabel("Setpoint:")
        sp_layout.addWidget(sp_label)
        ToolTip(sp_label, "Target phase setpoint for the PID controller\nThis is the desired phase value to maintain")
        
        # Safe setpoint initialization
        initial_setpoint = getattr(self.manager, 'setpoint', 0.0)
        self.sp_entry = QLineEdit(str(initial_setpoint))
        self.sp_entry.returnPressed.connect(self._update_setpoint)
        sp_layout.addWidget(self.sp_entry)
        ToolTip(self.sp_entry, "Enter the desired phase setpoint value\nPress Enter to apply the new setpoint")
        
        # Auto setpoint button
        auto_sp_btn = QPushButton("Auto")
        auto_sp_btn.clicked.connect(self._auto_setpoint)
        sp_layout.addWidget(auto_sp_btn)
        ToolTip(auto_sp_btn, "Automatically set setpoint to the middle value\nbetween Vmin and Vmax from range calibration")
        
        ctrl_layout.addLayout(sp_layout)
        
        # Create a layout for checkboxes to place them side by side
        check_layout = QHBoxLayout()
        
        # Lock enable control
        self.lock_check = QCheckBox("Enable Lock")
        self.lock_check.stateChanged.connect(self._toggle_lock)
        check_layout.addWidget(self.lock_check)
        ToolTip(self.lock_check, "Enable/disable the PID lock\nWhen disabled, the phase drifts freely.")
        
        # Monitoring control
        self.monitor_check = QCheckBox("Monitor Locks")
        self.monitor_check.stateChanged.connect(self._toggle_monitoring)
        check_layout.addWidget(self.monitor_check)
        ToolTip(self.monitor_check, "Enable/disable continuous monitoring of phase locks\nWhen enabled, the system will automatically check and maintain lock stability")
        
        ctrl_layout.addLayout(check_layout)
        
        # Add Visualization Frame
        vis_frame = QGroupBox("Visualization")
        vis_layout = QHBoxLayout(vis_frame)
        
        plot_range_btn = QPushButton("Plot Range Calibration")
        plot_range_btn.clicked.connect(self._plot_range_calibration)
        vis_layout.addWidget(plot_range_btn)
        ToolTip(plot_range_btn, "Display the latest range calibration data and fit")
        
        plot_lock_btn = QPushButton("Plot Lock Performance")
        plot_lock_btn.clicked.connect(self._plot_lock_performance)
        vis_layout.addWidget(plot_lock_btn)
        ToolTip(plot_lock_btn, "Display the latest lock performance data and fit")
        
        plot_combined_btn = QPushButton("Plot Combined Analysis")
        plot_combined_btn.clicked.connect(self._plot_combined_analysis)
        vis_layout.addWidget(plot_combined_btn)
        ToolTip(plot_combined_btn, "Display both calibration and lock performance plots")
        
        # Add all frames to the main layout
        main_layout.addWidget(range_frame, 0, 0)
        main_layout.addWidget(pid_frame, 0, 1)
        main_layout.addWidget(lock_frame, 1, 0)
        main_layout.addWidget(ctrl_frame, 1, 1)
        main_layout.addWidget(vis_frame, 2, 0, 1, 2)
        
        # Auto-load latest results
        self._load_latest_results()

    def _load_latest_results(self):
        """Automatically load and display the latest available results"""
        try:
            # Try to get latest range calibration
            if hasattr(self.manager, 'get_latest_range_calibration'):
                range_result = self.manager.get_latest_range_calibration()
                if range_result:
                    # Extract vmin and vmax from par array (indices 1 and 2)
                    if 'par' in range_result and len(range_result['par']) >= 3:
                        vmin = range_result['par'][1]
                        vmax = range_result['par'][2]
                        self.range_label.setText(f"Range: {vmin:.3f} - {vmax:.3f}V")
                        self.range_time.setText(f"Calibrated: {self._format_timestamp(range_result['timestamp'])}")
                    else:
                        # Fallback to direct keys if par array not available
                        vmin = range_result.get('vmin', 'N/A')
                        vmax = range_result.get('vmax', 'N/A')
                        if isinstance(vmin, (int, float)):
                            self.range_label.setText(f"Range: {vmin:.3f} - {vmax:.3f}V")
                        else:
                            self.range_label.setText(f"Range: {vmin} - {vmax}V")
                        if 'timestamp' in range_result:
                            self.range_time.setText(f"Calibrated: {self._format_timestamp(range_result['timestamp'])}")
        except Exception as e:
            print(f"Could not load latest range calibration: {e}")
        
        try:
            # Try to get latest visibility
            if hasattr(self.manager, 'get_latest_visibility'):
                vis_result = self.manager.get_latest_visibility()
                if vis_result:
                    self.vis_label.setText(f"Visibility: {vis_result['visibility']:.3f}")
                    self.vis_time.setText(f"Measured: {self._format_timestamp(vis_result['timestamp'])}")
        except Exception as e:
            print(f"Could not load latest visibility: {e}")
        
        try:
            # Try to get latest lock quality
            if hasattr(self.manager, 'latest_lock_quality') and self.manager.latest_lock_quality is not None:
                if hasattr(self.manager, 'get_latest_lock_evaluation'):
                    lock_result = self.manager.get_latest_lock_evaluation()
                    if lock_result:
                        self.lock_label.setText(f"Lock Quality: {self.manager.latest_lock_quality:.3f}")
                        self.lock_time.setText(f"Measured: {self._format_timestamp(lock_result['timestamp'])}")
        except Exception as e:
            print(f"Could not load latest lock quality: {e}")

    def _format_timestamp(self, timestamp_str: str) -> str:
        """Convert ISO timestamp to readable format"""
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%d.%m.%Y at %H:%M:%S")

    def _load_pid_config(self):
        """Load PID config with confirmation dialog"""
        reply = QMessageBox.question(self, 'Confirm', 
                                     "Load the latest PID configuration? This will overwrite current settings.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.manager.load_latest_pid_config()

    def _range_calibration(self):
        """Run range calibration with confirmation"""
        reply = QMessageBox.question(self, 'Confirm', 
                                     "Run range calibration? This will temporarily disable the locks\nand drive the piezo.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            result = self.manager.perform_range_calibration()
            if result:
                # Extract vmin and vmax from par array (indices 1 and 2)
                if 'par' in result and len(result['par']) >= 3:
                    vmin = result['par'][1]
                    vmax = result['par'][2]
                    self.range_label.setText(f"Range: {vmin:.3f} - {vmax:.3f}V")
                    self.range_time.setText(f"Calibrated: {self._format_timestamp(result['timestamp'])}")
                else:
                    # Fallback to direct keys if par array not available
                    vmin = result.get('vmin', 'N/A')
                    vmax = result.get('vmax', 'N/A')
                    if isinstance(vmin, (int, float)):
                        self.range_label.setText(f"Range: {vmin:.3f} - {vmax:.3f}V")
                    else:
                        self.range_label.setText(f"Range: {vmin} - {vmax}V")
                    if 'timestamp' in result:
                        self.range_time.setText(f"Calibrated: {self._format_timestamp(result['timestamp'])}")

    def _measure_visibility(self):
        """Measure fringe visibility"""
        result = self.manager.perform_visibility_calibration()
        self.vis_label.setText(f"Visibility: {result['visibility']:.3f}")
        self.vis_time.setText(f"Measured: {self._format_timestamp(result['timestamp'])}")
    
    def _evaluate_lock(self):
        """Evaluate lock quality"""
        result = self.manager.evaluate_current_lock()
        self.lock_label.setText(f"Lock Quality: {self.manager.latest_lock_quality:.3f}")
        self.lock_time.setText(f"Measured: {self._format_timestamp(result['timestamp'])}")
    
    @pyqtSlot()
    def _update_setpoint(self):
        """Update setpoint from text entry"""
        try:
            value = float(self.sp_entry.text())
            self.manager.setpoint = value
        except ValueError:
            self.sp_entry.setText(str(self.manager.setpoint))
    
    @pyqtSlot(int)
    def _toggle_lock(self, state):
        """Toggle the PID lock on/off"""
        try:
            self.manager.toggle_locks(state == Qt.Checked)
        except AttributeError:
            QMessageBox.warning(self, "Feature Unavailable", 
                               "Lock control not available with current manager.")
            self.lock_check.setChecked(False)
    
    @pyqtSlot(int)
    def _toggle_monitoring(self, state):
        """Toggle continuous lock monitoring"""
        if state == Qt.Checked:
            self.manager.start_monitoring()
        else:
            self.manager.stop_monitoring()
    
    def _center_window(self):
        """Center the window on screen"""
        frame_geometry = self.frameGeometry()
        screen_center = QApplication.desktop().screenGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.move(frame_geometry.topLeft())
    
    def _auto_setpoint(self):
        """Automatically set setpoint to middle of calibrated range"""
        try:
            if hasattr(self.manager, 'get_latest_range_calibration'):
                range_result = self.manager.get_latest_range_calibration()
                if range_result:
                    # Extract vmin and vmax from par array (indices 1 and 2)
                    if 'par' in range_result and len(range_result['par']) >= 3:
                        vmin = range_result['par'][1]
                        vmax = range_result['par'][2]
                        middle_value = (vmin + vmax) / 2.0
                        self.sp_entry.setText(f"{middle_value:.3f}")
                        self.manager.setpoint = middle_value
                        print(f"Auto-set setpoint to middle value: {middle_value:.3f}V")
                    elif 'vmin' in range_result and 'vmax' in range_result:
                        # Fallback to direct keys
                        vmin = range_result['vmin']
                        vmax = range_result['vmax']
                        middle_value = (vmin + vmax) / 2.0
                        self.sp_entry.setText(f"{middle_value:.3f}")
                        self.manager.setpoint = middle_value
                        print(f"Auto-set setpoint to middle value: {middle_value:.3f}V")
                    else:
                        QMessageBox.warning(self, "Invalid Data", "Range calibration data format not recognized.")
                else:
                    QMessageBox.warning(self, "No Range Data", 
                                       "No range calibration data available.\nPlease run range calibration first.")
            else:
                QMessageBox.warning(self, "Feature Unavailable", 
                                   "Auto setpoint feature not available with current manager.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to auto-set setpoint: {str(e)}")
    
    def _plot_range_calibration(self):
        """Display range calibration plot"""
        try:
            fig, _ = self.visualizer.plot_range_calibration()
            plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Failed to plot range calibration: {str(e)}")
    
    def _plot_lock_performance(self):
        """Display lock performance plot"""
        try:
            fig, _ = self.visualizer.plot_lock_performance()
            plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Failed to plot lock performance: {str(e)}")
    
    def _plot_combined_analysis(self):
        """Display combined analysis plots"""
        try:
            fig, _ = self.visualizer.plot_combined_analysis()
            plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Failed to plot combined analysis: {str(e)}")
            print(f"Debug info - Error details: {str(e)}")  # Added debug info

# Application entry point
if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MZControlIndyGUI()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()