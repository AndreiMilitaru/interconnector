from datetime import datetime
import numpy as np
from typing import Optional, Dict
from .manager_interface import MZManagerInterface

class DummyMZManager(MZManagerInterface):
    def __init__(self, lock_check_interval: float = 0.1):
        self._monitoring = False
        self._setpoint = 0.0
        self._lock_check_interval = lock_check_interval
        self._locks_enabled = False
        self._latest_lock_quality = 0.9  # Store as private attribute
        
    @property
    def setpoint(self) -> float:
        return self._setpoint
    
    @setpoint.setter
    def setpoint(self, value: float):
        self._setpoint = value
    
    def start_monitoring(self):
        self._monitoring = True
    
    def stop_monitoring(self):
        self._monitoring = False
    
    def save_current_pid_config(self):
        """Save current PID configuration"""
        print("Dummy: Saving PID configuration")
        
    def load_latest_pid_config(self):
        """Load latest PID configuration"""
        print("Dummy: Loading PID configuration")
        
    def perform_range_calibration(self) -> Dict:
        """Perform range calibration"""
        print("Dummy: Performing range calibration")
        return {
            'par': np.array([1.0, 0.0, 5.0]),  # [amplitude, vmin, vmax]
            'timestamp': datetime.now().isoformat()
        }
        
    def perform_visibility_calibration(self) -> Dict:
        """Perform visibility calibration"""
        print("Dummy: Measuring visibility")
        return {
            'visibility': 0.95,
            'timestamp': datetime.now().isoformat()
        }
        
    def evaluate_current_lock(self) -> Dict:
        """Evaluate current lock quality"""
        print("Dummy: Evaluating lock quality")
        # Update private attribute to match interface requirements
        self._latest_lock_quality = 0.9
        return {
            'timestamp': datetime.now().isoformat()
        }
    
    @property
    def latest_lock_quality(self) -> Optional[float]:
        """Return latest lock quality - implemented as property to satisfy abstract method"""
        return self._latest_lock_quality
    
    def get_latest_range_calibration(self) -> Optional[Dict]:
        """Get latest range calibration results"""
        return {
            'par': np.array([1.0, 0.0, 5.0]),  # [amplitude, vmin, vmax]
            'timestamp': datetime.now().isoformat()
        }
    
    def get_latest_visibility(self) -> Optional[Dict]:
        """Get latest visibility measurement"""
        return {
            'visibility': 0.95,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_latest_lock_evaluation(self) -> Optional[Dict]:
        """Get latest lock evaluation results"""
        return {
            'quality': 0.9,
            'timestamp': datetime.now().isoformat()
        }
        
    def toggle_locks(self, enable: bool):
        """Toggle locks on/off"""
        self._locks_enabled = enable
        print(f"Dummy: {'Enabling' if enable else 'Disabling'} locks")
