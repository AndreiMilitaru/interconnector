"""
author: Andrei Militaru
organization: Institute of Science and Technology Austria (ISTA)
date: October 2025
Description: Main class to handle the Mach Zehnder stabilization system.
"""

import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import threading
import time
from mach_zehnder_utils.phase_calibration import (
    calibrate_range, evaluate_visibility, evaluate_lock_precision, toggle_locks
)
from mach_zehnder_utils.mach_zehnder_lock import (
    set_demodulators, set_aux_limits, set_pid_params, set_setpoint, check_locks
)
from mach_zehnder_utils.manager_interface import MZManagerInterface

class MachZehnderManager(MZManagerInterface):
    """Class to manage Mach Zehnder stabilization system."""

    def __init__(
        self,
        mdrec,
        config_path: Optional[str] = None,
        load_latest_pid_config: bool = False,
        lock_check_interval: float = 0.1,
    ):
        """Initialize MZ stabilization system.
        
        Args:
            mdrec: Measurement device record instance
            config_path: Path to configuration files folder
            load_latest_pid_config: Whether to load most recent PID config
            lock_check_interval: Interval for lock monitoring thread
        """
        print("Initializing manager...")
        self._mdrec = mdrec
        self._config_path = Path(config_path or "../config/mach_zehnder")
        self._lock_check_interval = lock_check_interval
        self._monitoring_active = False
        self._monitor_thread = None
        
        self._load_config()
        self._setup_calibration_folders()
        
        # Initialize demodulators
        self._setup_demodulators()
        self.set_aux_limits()
        self.set_pid_params() 
        
        if load_latest_pid_config:
            self.load_latest_pid_config()
    
    def _load_config(self, config_name: Optional[str] = 'default_config.yaml'):
        """Load YAML configuration"""
        print(f"Loading configuration from {self._config_path / config_name}")
        with open(self._config_path / config_name, 'r') as f:
            self._config = yaml.safe_load(f)
        self._device_id = self._config['device']['id']
        print("Loading complete.")
    
    def _setup_calibration_folders(self):
        """Create folders for storing calibration data"""
        print("Creating calibration folders if not already existing...")
        calib_base = self._config_path / "calibrations"
        calib_base.mkdir(exist_ok=True)
        for calib_type in ['range', 'visibility', 'lock_precision', 'pid_config']:
            (calib_base / calib_type).mkdir(exist_ok=True)
        print("Done.")

    def _setup_demodulators(self):
        """Set up demodulators based on config file"""
        print("Setting up demodulators...")
        demod_config = self._config.get('demodulators', {})
        if not demod_config:
            raise ValueError("No demodulator configuration found in config file")
        
        # Setup main demodulator
        main_config = demod_config['input']
        set_demodulators(
            self._mdrec,
            dev=self._device_id,
            oscillator=main_config['oscillator'],
            demodulator=main_config['demodulator'],
            order=main_config['order'],
            rate=main_config['rate'],
            bandwidth=main_config['bandwidth']
        )
        print("Done.")
        
        self._demod_config = demod_config
    
    def toggle_locks(self, value: bool):
        toggle_locks(self._mdrec, value, dev=self._device_id)

    def perform_range_calibration(self, reset_pids: Optional[bool] = True) -> Dict:
        """Perform range calibration and save results"""

        restart_monitor = False
        if self._monitoring_active:
            restart_monitor = True
            self.stop_monitoring()

        par, cov, hist, edges = calibrate_range(
            self._mdrec,
            dev=self._device_id,
            reset_pids=reset_pids,
            **self._config['demodulators']['phase_drive']
        )
        
        timestamp = datetime.now().isoformat()
        data = {
            'parameters': par,
            'covariance': cov,
            'histogram': hist,
            'edges': edges,
            'timestamp': timestamp
        }
        path = self._config_path / self._config['calibration_paths']['range']
        self._save_calibration_data(path, data)
        if restart_monitor:
            self.start_monitoring()

        return data
    
    def save_current_pid_config(self):
        """Save current PID configuration to file"""
        piezo_config = self._config['pid']['piezo']
        laser_config = self._config['pid']['laser']
        
        piezo_params = {
            'p': self._mdrec.lock_in.get(f'/{self._device_id}/pids/{piezo_config["pid_number"]}/p'),
            'i': self._mdrec.lock_in.get(f'/{self._device_id}/pids/{piezo_config["pid_number"]}/i'),
            'd': self._mdrec.lock_in.get(f'/{self._device_id}/pids/{piezo_config["pid_number"]}/d'),
            'setpoint': self._mdrec.lock_in.get(f'/{self._device_id}/pids/{piezo_config["pid_number"]}/setpoint')
        }
        
        laser_params = {
            'p': self._mdrec.lock_in.get(f'/{self._device_id}/pids/{laser_config["pid_number"]}/p'),
            'i': self._mdrec.lock_in.get(f'/{self._device_id}/pids/{laser_config["pid_number"]}/i'),
            'd': self._mdrec.lock_in.get(f'/{self._device_id}/pids/{laser_config["pid_number"]}/d'),
            'setpoint': self._mdrec.lock_in.get(f'/{self._device_id}/pids/{laser_config["pid_number"]}/setpoint')
        }
        
        data = {
            'piezo_params': piezo_params,
            'laser_params': laser_params,
            'timestamp': datetime.now().isoformat()
        }
        
        path = self._config_path / self._config['calibration_paths']['pid_config']
        self._save_calibration_data(path, data)

    def load_latest_pid_config(self) -> Optional[Dict]:
        """Load the most recent PID configuration"""
        path = self._config_path / self._config['calibration_paths']['pid_config']
        if not path.exists():
            return None
        
        pid_data = np.load(str(path), allow_pickle=True).item()
        piezo_config = self._config['pid']['piezo']
        laser_config = self._config['pid']['laser']
        
        set_pid_params(
            self._mdrec,
            dev=self._device_id,
            piezo_params=pid_data['piezo_params'],
            laser_params=pid_data['laser_params'],
            piezo_aux=piezo_config['pid_number'],
            laser_aux=laser_config['pid_number'],
            demodulator=self._config['demodulators']['main']['demodulator'],
            piezo_out=piezo_config['aux'],
            laser_out=laser_config['aux'],
            piezo_center=piezo_config['center'],
            laser_range=laser_config['limit_upper']
        )
        return pid_data

    def perform_visibility_calibration(self, range_parameters: Optional[np.ndarray] = None) -> Dict:
        """Perform visibility calibration and save results"""
        if range_parameters is None:
            range_calib = self._load_latest_calibration('range')
            if range_calib is None:
                raise ValueError("No range calibration found. Run range calibration first.")
            range_parameters = range_calib['parameters']
        
        visibility = evaluate_visibility(range_parameters)
        timestamp = datetime.now().isoformat()
        data = {
            'visibility': visibility,
            'timestamp': timestamp
        }
        
        path = self._config_path / self._config['calibration_paths']['visibility']
        self._save_calibration_data(path, data)
        return data

    def evaluate_current_lock(self, use_latest_calibration: bool = True) -> Dict:
        """Evaluate current lock precision"""
        if use_latest_calibration:
            range_calib = self._load_latest_calibration('range')
            if range_calib is None:
                raise ValueError("No range calibration found. Run calibration first.")
        
        par_lock, cov_lock, hist, edges = evaluate_lock_precision(
            self._mdrec,
            dev=self._device_id,
            par=range_calib['parameters']
        )
        
        data = {
            'lock_parameters': par_lock,
            'lock_covariance': cov_lock,
            'histogram': hist,
            'edges': edges,
            'timestamp': datetime.now().isoformat()
        }
        
        path = self._config_path / self._config['calibration_paths']['lock_precision']
        self._save_calibration_data(path, data)
        return data
    
    @staticmethod
    def _create_timestamped_filename(base_path: Path, timestamp: str) -> Path:
        """Create a filename with timestamp"""
        # Convert timestamp to a filename-friendly format
        clean_timestamp = timestamp.replace(':', '-').replace('.', '-')
        return base_path / f"data_{clean_timestamp}.npy"

    def _save_calibration_data(self, path: Path, data: Dict):
        """Save calibration data with timestamp in filename"""
        timestamp = data.get('timestamp', datetime.now().isoformat())
        filepath = self._create_timestamped_filename(path, timestamp)
        np.save(str(filepath), data, allow_pickle=True)

    def _load_latest_calibration(self, calib_type: str) -> Optional[Dict]:
        """Load most recent calibration data"""
        path = self._config_path / self._config['calibration_paths'][calib_type]
        if not path.exists():
            return None
        
        # Find all calibration files
        calib_files = list(path.glob("data_*.npy"))
        if not calib_files:
            return None
            
        # Sort by modification time and get the most recent
        latest_file = max(calib_files, key=lambda x: x.stat().st_mtime)
        return np.load(str(latest_file), allow_pickle=True).item()
    
    @property
    def latest_lock_quality(self) -> Optional[float]:
        """Get the quality metric from the most recent lock evaluation"""
        lock_data = self._load_latest_calibration('lock_precision')
        if lock_data is None:
            return None
        # Calculate quality metric based on lock parameters
        sigma = np.sqrt(lock_data['lock_parameters'][1])  # standard deviation
        return sigma
    
    def set_aux_limits(self):
        """Set auxiliary output limits for piezo and laser channels"""
        piezo_limits = self._config['aux_limits']['piezo']
        laser_limits = self._config['aux_limits']['laser']

        set_aux_limits(
            self._mdrec,
            dev=self._device_id,
            aux_lim=[piezo_limits['min'], piezo_limits['max']],
            laser_lim=[laser_limits['min'], laser_limits['max']]
        )
    
    def set_pid_params(self):
        """Configure PID parameters for both piezo and laser channels"""
        piezo_config = self._config['pid']['piezo']
        laser_config = self._config['pid']['laser']
        set_pid_params(
            self._mdrec,
            dev=self._device_id,
            piezo_params=piezo_config['params'],
            laser_params=laser_config['params'],
            piezo_pid=piezo_config['pid_number'],
            laser_pid=laser_config['pid_number'],
            demodulator=self._config['demodulators']['input']['demodulator'],
            piezo_out=piezo_config['aux'],
            laser_out=laser_config['aux'],
            piezo_center=piezo_config['center'],
            laser_range=laser_config['limit_upper']
        )
    
    @property
    def setpoint(self) -> float:
        """Get the current PID setpoint value"""
        return self._mdrec.lock_in.get(f'/{self._device_id}/pids/0/setpoint')
    
    @setpoint.setter
    def setpoint(self, value: float):
        """Set the PID setpoint for both piezo and laser channels"""
        set_setpoint(self._mdrec, value, dev=self._device_id)
    
    def _monitor_locks(self):
        """Background thread function to monitor lock status"""
        piezo_config = self._config['pid']['piezo']
        laser_config = self._config['pid']['laser']
        
        while self._monitoring_active:
            check_locks(
                self._mdrec,
                dev=self._device_id,
                piezo_pid=piezo_config['pid_number'],
                piezo_aux=piezo_config['aux'],
                laser_pid=laser_config['pid_number'],
                laser_aux=laser_config['aux']
            )
            time.sleep(self._lock_check_interval)

    def start_monitoring(self):
        """Start the lock monitoring thread"""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_locks,
                daemon=True
            )
            self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop the lock monitoring thread"""
        if self._monitor_thread is not None:
            self._monitoring_active = False
            # Wait for thread to finish
            self._monitor_thread.join(timeout=2*self._lock_check_interval)
            
            # Check if thread actually stopped
            if self._monitor_thread.is_alive():
                raise RuntimeWarning("Monitor thread did not stop cleanly")
            
            self._monitor_thread = None

    @property
    def is_monitoring(self) -> bool:
        """Check if lock monitoring is active"""
        return self._monitoring_active

    def __del__(self):
        """Ensure monitoring thread is stopped when object is destroyed"""
        self.stop_monitoring()
