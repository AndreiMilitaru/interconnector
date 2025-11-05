from abc import ABC, abstractmethod
from typing import Optional, Dict
from datetime import datetime

class MZManagerInterface(ABC):
    @abstractmethod
    def perform_range_calibration(self) -> Dict:
        pass
    
    @abstractmethod
    def save_current_pid_config(self):
        pass
    
    @abstractmethod
    def load_latest_pid_config(self):
        pass
    
    @abstractmethod
    def perform_visibility_calibration(self) -> Dict:
        pass
    
    @abstractmethod
    def evaluate_current_lock(self) -> Dict:
        pass
    
    @property
    @abstractmethod
    def latest_lock_quality(self) -> Optional[float]:
        pass
    
    @property
    @abstractmethod
    def setpoint(self) -> float:
        pass
    
    @setpoint.setter
    @abstractmethod
    def setpoint(self, value: float):
        pass
    
    @abstractmethod
    def start_monitoring(self):
        pass
    
    @abstractmethod
    def stop_monitoring(self):
        pass
