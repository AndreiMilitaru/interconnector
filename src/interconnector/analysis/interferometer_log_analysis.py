"""
Interferometer Control Log Analysis
Author: Andrei Militaru with GitHub Copilot
Organization: Institute of Science and Technology Austria (ISTA)
Date: February 2026

Description: Object-oriented data analysis tool for interferometer control log files.
Extracts timestamps, parameter changes, and system states from log files.
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
import numpy as np


class InterferometerLogAnalyzer:
    """Analyzes interferometer control log files to extract system events and parameters."""
    
    def __init__(self, input_path: Union[str, List[str], Path]):
        """
        Initialize the analyzer with log files.
        
        Args:
            input_path: Either a single filename, list of filenames, or a folder path.
                       If folder, all log files in it will be analyzed.
        """
        self.log_files = []
        self.parsed_data = {}
        
        # Handle input path
        if isinstance(input_path, (str, Path)):
            path = Path(input_path)
            if path.is_dir():
                # Get all .log files in directory
                self.log_files = sorted(path.glob("interferometer_control_*.log"))
            elif path.is_file():
                self.log_files = [path]
        elif isinstance(input_path, list):
            self.log_files = [Path(f) for f in input_path]
        
        if not self.log_files:
            raise ValueError(f"No log files found at: {input_path}")
        
        # Parse all files
        self._parse_all_files()
    
    def _parse_timestamp(self, line: str) -> Optional[datetime]:
        """Extract timestamp from log line."""
        match = re.match(r'^\[(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\]', line)
        if match:
            return datetime.strptime(match.group(1), '%Y-%m-%d_%H-%M-%S')
        return None
    
    def _parse_all_files(self):
        """Parse all log files and extract relevant information."""
        for log_file in self.log_files:
            self.parsed_data[str(log_file)] = self._parse_file(log_file)
    
    def _parse_file(self, log_file: Path) -> Dict:
        """Parse a single log file and extract all relevant data."""
        data = {
            'file_start': None,
            'file_end': None,
            'pid_enabled': [],  # List of (time, state) tuples
            'reflection': [],  # List of (time, mean, std) tuples
            'setpoint': [],  # List of (time, value) tuples
            'bandwidth': [],  # List of (time, value) tuples
            'piezo_p': [],  # List of (time, value) tuples
            'piezo_i': [],  # List of (time, value) tuples
            'laser_p': [],  # List of (time, value) tuples
            'laser_i': [],  # List of (time, value) tuples
            'initial_values': {}
        }
        
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            timestamp = self._parse_timestamp(line)
            if not timestamp:
                continue
            
            # Track file start and end
            if data['file_start'] is None:
                data['file_start'] = timestamp
            data['file_end'] = timestamp
            
            # Parse initial values
            if 'Initial values loaded from device:' in line:
                match = re.search(r'PID Enabled=(True|False), Setpoint=([\d.\-]+) V, Bandwidth=([\d.]+) kHz', line)
                if match:
                    data['initial_values']['pid_enabled'] = match.group(1) == 'True'
                    data['initial_values']['setpoint'] = float(match.group(2))
                    data['initial_values']['bandwidth'] = float(match.group(3))
            
            if 'Initial PID parameters:' in line:
                match = re.search(r'Piezo P=([\d.\-]+), Piezo I=([\d.\-]+), Laser P=([\d.\-]+), Laser I=([\d.\-]+)', line)
                if match:
                    data['initial_values']['piezo_p'] = float(match.group(1))
                    data['initial_values']['piezo_i'] = float(match.group(2))
                    data['initial_values']['laser_p'] = float(match.group(3))
                    data['initial_values']['laser_i'] = float(match.group(4))
            
            # Parse PID enable state changes
            if 'PID enable ->' in line:
                match = re.search(r'PID enable -> (True|False)', line)
                if match:
                    state = match.group(1) == 'True'
                    data['pid_enabled'].append((timestamp, state))
            
            # Parse reflection measurements (from monitoring)
            if 'Reflection: mean=' in line:
                match = re.search(r'mean=([\d.\-]+) V, std=([\d.\-]+) V', line)
                if match:
                    mean = float(match.group(1))
                    std = float(match.group(2))
                    data['reflection'].append((timestamp, mean, std))
            
            # Parse setpoint changes
            if 'Setpoint ->' in line:
                match = re.search(r'Setpoint -> ([\d.\-]+) V', line)
                if match:
                    value = float(match.group(1))
                    data['setpoint'].append((timestamp, value))
            
            # Parse bandwidth changes
            if 'Bandwidth ->' in line:
                match = re.search(r'Bandwidth -> ([\d.]+) kHz', line)
                if match:
                    value = float(match.group(1))
                    data['bandwidth'].append((timestamp, value))
            
            # Parse PID parameter changes
            if 'Piezo P ->' in line:
                match = re.search(r'Piezo P -> ([\d.\-]+)', line)
                if match:
                    value = float(match.group(1))
                    data['piezo_p'].append((timestamp, value))
            
            if 'Piezo I ->' in line:
                match = re.search(r'Piezo I -> ([\d.\-]+)', line)
                if match:
                    value = float(match.group(1))
                    data['piezo_i'].append((timestamp, value))
            
            if 'Laser P ->' in line:
                match = re.search(r'Laser P -> ([\d.\-]+)', line)
                if match:
                    value = float(match.group(1))
                    data['laser_p'].append((timestamp, value))
            
            if 'Laser I ->' in line:
                match = re.search(r'Laser I -> ([\d.\-]+)', line)
                if match:
                    value = float(match.group(1))
                    data['laser_i'].append((timestamp, value))
        
        return data
    
    def _filter_by_date(self, timestamps: List[Tuple], 
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> List[Tuple]:
        """Filter timestamp tuples by date range."""
        if not timestamps:
            return []
        
        start_dt = None
        end_dt = None
        
        if start_date:
            start_dt = datetime.strptime(start_date, '%y%m%d:%H%M')
        if end_date:
            end_dt = datetime.strptime(end_date, '%y%m%d:%H%M')
        
        filtered = []
        for item in timestamps:
            # Get the timestamp (first element of tuple)
            ts = item[0]
            
            if start_dt and ts < start_dt:
                continue
            if end_dt and ts > end_dt:
                continue
            
            filtered.append(item)
        
        return filtered
    
    def inspect_pid_enabled(self,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> Dict:
        """
        Return periods when PID was enabled.
        
        Args:
            start_date: Optional start date in format 'yymmdd:hhmm'
            end_date: Optional end date in format 'yymmdd:hhmm'
        
        Returns:
            Dictionary with 'enabled_periods' (list of start/end times) and
            'file_boundaries' (list of file start/end times)
        """
        periods = []
        file_boundaries = []
        
        for log_file, data in self.parsed_data.items():
            file_boundaries.append((log_file, data['file_start'], data['file_end']))
            states = self._filter_by_date(data['pid_enabled'], start_date, end_date)
            
            if not states:
                continue
            
            # Build enabled periods
            current_state = data['initial_values'].get('pid_enabled', False)
            period_start = data['file_start'] if current_state else None
            
            for timestamp, state in states:
                if state and not current_state:  # Enabling
                    period_start = timestamp
                    current_state = True
                elif not state and current_state:  # Disabling
                    if period_start:
                        periods.append((period_start, timestamp))
                    current_state = False
                    period_start = None
            
            # If ended in enabled state
            if current_state and period_start:
                periods.append((period_start, data['file_end']))
        
        return {
            'enabled_periods': periods,
            'file_boundaries': file_boundaries
        }
    
    def inspect_pid_disabled(self,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> List[Tuple]:
        """
        Return periods when PID was disabled.
        
        Args:
            start_date: Optional start date in format 'yymmdd:hhmm'
            end_date: Optional end date in format 'yymmdd:hhmm'
        
        Returns:
            List of (start_time, end_time) tuples when PID was disabled
        """
        periods = []
        
        for log_file, data in self.parsed_data.items():
            states = self._filter_by_date(data['pid_enabled'], start_date, end_date)
            
            if not states:
                # If no state changes and initial state was disabled
                if not data['initial_values'].get('pid_enabled', False):
                    periods.append((data['file_start'], data['file_end']))
                continue
            
            # Build disabled periods
            current_state = data['initial_values'].get('pid_enabled', False)
            period_start = data['file_start'] if not current_state else None
            
            for timestamp, state in states:
                if not state and current_state:  # Disabling
                    period_start = timestamp
                    current_state = False
                elif state and not current_state:  # Enabling
                    if period_start:
                        periods.append((period_start, timestamp))
                    current_state = True
                    period_start = None
            
            # If ended in disabled state
            if not current_state and period_start:
                periods.append((period_start, data['file_end']))
        
        return periods
    
    def inspect_interferometer_locked(self,
                                     start_date: Optional[str] = None,
                                     end_date: Optional[str] = None) -> Dict:
        """
        Inspect when interferometer was locked (PID enabled periods).
        
        Args:
            start_date: Optional start date in format 'yymmdd:hhmm'
            end_date: Optional end date in format 'yymmdd:hhmm'
        
        Returns:
            Dictionary with 'locked_periods' when interferometer was locked
        """
        return self.inspect_pid_enabled(start_date, end_date)
    
    def get_reflection_vs_time(self,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> Dict:
        """
        Get average reflection and standard deviation as a function of time.
        
        Args:
            start_date: Optional start date in format 'yymmdd:hhmm'
            end_date: Optional end date in format 'yymmdd:hhmm'
        
        Returns:
            Dictionary with 'timestamps', 'mean', and 'std' arrays
        """
        all_data = []
        
        for log_file, data in self.parsed_data.items():
            reflections = self._filter_by_date(data['reflection'], start_date, end_date)
            all_data.extend(reflections)
        
        if not all_data:
            return {'timestamps': np.array([]), 'mean': np.array([]), 'std': np.array([])}
        
        # Sort by timestamp
        all_data.sort(key=lambda x: x[0])
        
        timestamps = np.array([item[0] for item in all_data])
        means = np.array([item[1] for item in all_data])
        stds = np.array([item[2] for item in all_data])
        
        return {
            'timestamps': timestamps,
            'mean': means,
            'std': stds
        }
    
    def get_pid_parameter_changes(self,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None) -> Dict:
        """
        Get all changes in PID parameters over time.
        
        Args:
            start_date: Optional start date in format 'yymmdd:hhmm'
            end_date: Optional end date in format 'yymmdd:hhmm'
        
        Returns:
            Dictionary with parameter names as keys, each containing
            'timestamps' and 'values' arrays
        """
        result = {}
        
        for param_name in ['setpoint', 'bandwidth', 'piezo_p', 'piezo_i', 'laser_p', 'laser_i']:
            all_changes = []
            
            for log_file, data in self.parsed_data.items():
                changes = self._filter_by_date(data[param_name], start_date, end_date)
                all_changes.extend(changes)
            
            # Sort by timestamp
            all_changes.sort(key=lambda x: x[0])
            
            result[param_name] = {
                'timestamps': np.array([item[0] for item in all_changes]) if all_changes else np.array([]),
                'values': np.array([item[1] for item in all_changes]) if all_changes else np.array([])
            }
        
        return result
    
    def initial_values(self) -> List[Dict]:
        """
        Return initial values from all log files.
        
        Returns:
            List of dictionaries containing initial values, one per file
        """
        result = []
        
        for log_file, data in self.parsed_data.items():
            entry = {
                'filename': str(log_file),
                'file_start': data['file_start'],
                'values': data['initial_values']
            }
            result.append(entry)
        
        return result


# Example usage
if __name__ == "__main__":
    # Example: analyze logs from a folder
    analyzer = InterferometerLogAnalyzer("./interferometer_logs")
    
    # Get PID enabled periods
    pid_enabled = analyzer.inspect_pid_enabled()
    print(f"Found {len(pid_enabled['enabled_periods'])} PID enabled periods")
    
    # Get locked periods
    locked = analyzer.inspect_interferometer_locked()
    print(f"Found {len(locked['locked_periods'])} locked periods")
    
    # Get initial values
    initial = analyzer.initial_values()
    print(f"Initial values from {len(initial)} files")
    
    # Get PID parameter changes
    params = analyzer.get_pid_parameter_changes()
    print(f"Setpoint changes: {len(params['setpoint']['timestamps'])}")
