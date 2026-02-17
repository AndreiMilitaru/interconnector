"""
Adaptive Laser Lock Log Analysis
Author: Andrei Militaru with GitHub Copilot
Organization: Institute of Science and Technology Austria (ISTA)
Date: February 2026

Description: Object-oriented data analysis tool for adaptive laser lock log files.
Extracts timestamps, parameter changes, and system states from CSV log files.
"""

import os
import re
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
import numpy as np


class AdaptiveLockLogAnalyzer:
    """Analyzes adaptive laser lock log files to extract system events and parameters."""
    
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
                # Get all _log.txt files in directory
                self.log_files = sorted(path.glob("*_log.txt"))
            elif path.is_file():
                self.log_files = [path]
        elif isinstance(input_path, list):
            self.log_files = [Path(f) for f in input_path]
        
        if not self.log_files:
            raise ValueError(f"No log files found at: {input_path}")
        
        # Parse all files
        self._parse_all_files()
    
    def _parse_all_files(self):
        """Parse all log files and extract relevant information."""
        for log_file in self.log_files:
            self.parsed_data[str(log_file)] = self._parse_file(log_file)
    
    def _parse_file(self, log_file: Path) -> Dict:
        """Parse a single CSV log file and extract all relevant data."""
        data = {
            'file_start': None,
            'file_end': None,
            'timestamps': [],  # List of datetime objects
            'in_pulse': [],    # List of voltages (V)
            'peak': [],        # List of voltages (V)
            'laser_offset': [],  # List of voltages (V)
            'cavity_rms': [],    # List of voltages (V)
            'cavity_position': [],  # List of voltages (V)
            'status': [],      # List of status strings
            'laser_locked': [],  # List of (start, end) tuples when laser was locked
            'cavity_locked': [],  # List of (start, end) tuples when cavity was locked
            'initial_values': {}
        }
        
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip header lines
        data_lines = []
        for line in lines:
            if line.startswith('Laser and Cavity Lock') or line.startswith('timestamp,'):
                continue
            data_lines.append(line.strip())
        
        if not data_lines:
            return data
        
        # Parse CSV data
        reader = csv.reader(data_lines)
        
        laser_lock_start = None
        cavity_lock_start = None
        
        for row in reader:
            if len(row) < 7:
                continue
            
            try:
                # Parse timestamp (Unix timestamp as float)
                timestamp = datetime.fromtimestamp(float(row[0]))
                in_pulse = float(row[1])
                peak = float(row[2])
                laser_offset = float(row[3])
                cavity_rms = float(row[4])
                cavity_position = float(row[5])
                status = row[6] if len(row) > 6 else ""
                
                # Store data
                data['timestamps'].append(timestamp)
                data['in_pulse'].append(in_pulse)
                data['peak'].append(peak)
                data['laser_offset'].append(laser_offset)
                data['cavity_rms'].append(cavity_rms)
                data['cavity_position'].append(cavity_position)
                data['status'].append(status)
                
                # Track file start and end
                if data['file_start'] is None:
                    data['file_start'] = timestamp
                    # Initial values are from first data point
                    data['initial_values'] = {
                        'in_pulse': in_pulse,
                        'peak': peak,
                        'laser_offset': laser_offset,
                        'cavity_rms': cavity_rms,
                        'cavity_position': cavity_position
                    }
                data['file_end'] = timestamp
                
                # Detect lock states from status string
                # Status contains things like "LOCKED", "SEARCH", "OSC", "DRIFT", etc.
                status_upper = status.upper()
                
                # Laser lock detection
                laser_locked = 'LASER' in status_upper and 'LOCKED' in status_upper
                if laser_locked and laser_lock_start is None:
                    laser_lock_start = timestamp
                elif not laser_locked and laser_lock_start is not None:
                    data['laser_locked'].append((laser_lock_start, timestamp))
                    laser_lock_start = None
                
                # Cavity lock detection
                cavity_locked = 'CAVITY' in status_upper and 'LOCKED' in status_upper
                if cavity_locked and cavity_lock_start is None:
                    cavity_lock_start = timestamp
                elif not cavity_locked and cavity_lock_start is not None:
                    data['cavity_locked'].append((cavity_lock_start, timestamp))
                    cavity_lock_start = None
                    
            except (ValueError, IndexError) as e:
                # Skip malformed lines
                continue
        
        # Close any open lock periods
        if laser_lock_start is not None and data['file_end'] is not None:
            data['laser_locked'].append((laser_lock_start, data['file_end']))
        if cavity_lock_start is not None and data['file_end'] is not None:
            data['cavity_locked'].append((cavity_lock_start, data['file_end']))
        
        return data
    
    def _filter_by_date(self, timestamps: List, values: List,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> Tuple[List, List]:
        """Filter timestamp and value arrays by date range."""
        if not timestamps:
            return [], []
        
        start_dt = None
        end_dt = None
        
        if start_date:
            start_dt = datetime.strptime(start_date, '%y%m%d:%H%M')
        if end_date:
            end_dt = datetime.strptime(end_date, '%y%m%d:%H%M')
        
        filtered_timestamps = []
        filtered_values = []
        
        for ts, val in zip(timestamps, values):
            if start_dt and ts < start_dt:
                continue
            if end_dt and ts > end_dt:
                continue
            
            filtered_timestamps.append(ts)
            filtered_values.append(val)
        
        return filtered_timestamps, filtered_values
    
    def get_time_series(self,
                       parameter: str,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> Dict:
        """
        Get time series data for a specific parameter.
        
        Args:
            parameter: One of 'in_pulse', 'peak', 'laser_offset', 'cavity_rms', 'cavity_position'
            start_date: Optional start date in format 'yymmdd:hhmm'
            end_date: Optional end date in format 'yymmdd:hhmm'
        
        Returns:
            Dictionary with 'timestamps' and 'values' arrays
        """
        valid_params = ['in_pulse', 'peak', 'laser_offset', 'cavity_rms', 'cavity_position']
        if parameter not in valid_params:
            raise ValueError(f"Parameter must be one of {valid_params}")
        
        all_timestamps = []
        all_values = []
        
        for log_file, data in self.parsed_data.items():
            filtered_ts, filtered_vals = self._filter_by_date(
                data['timestamps'], data[parameter], start_date, end_date
            )
            all_timestamps.extend(filtered_ts)
            all_values.extend(filtered_vals)
        
        if not all_timestamps:
            return {'timestamps': np.array([]), 'values': np.array([])}
        
        # Sort by timestamp
        sorted_pairs = sorted(zip(all_timestamps, all_values), key=lambda x: x[0])
        timestamps, values = zip(*sorted_pairs)
        
        return {
            'timestamps': np.array(timestamps),
            'values': np.array(values)
        }
    
    def get_all_parameters_vs_time(self,
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None) -> Dict:
        """
        Get time series data for all parameters.
        
        Args:
            start_date: Optional start date in format 'yymmdd:hhmm'
            end_date: Optional end date in format 'yymmdd:hhmm'
        
        Returns:
            Dictionary with keys for each parameter, each containing 'timestamps' and 'values'
        """
        result = {}
        
        for param in ['in_pulse', 'peak', 'laser_offset', 'cavity_rms', 'cavity_position']:
            result[param] = self.get_time_series(param, start_date, end_date)
        
        return result
    
    def get_contrast_vs_time(self,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> Dict:
        """
        Calculate and return contrast (ratio of in_pulse to peak) over time.
        
        Contrast = in_pulse / peak
        
        Args:
            start_date: Optional start date in format 'yymmdd:hhmm'
            end_date: Optional end date in format 'yymmdd:hhmm'
        
        Returns:
            Dictionary with 'timestamps' and 'contrast' arrays
        """
        all_timestamps = []
        all_contrasts = []
        
        for log_file, data in self.parsed_data.items():
            timestamps = data['timestamps']
            in_pulse = data['in_pulse']
            peak = data['peak']
            
            # Calculate contrast (avoid division by zero)
            contrasts = [ip / p if p != 0 else 0.0 for ip, p in zip(in_pulse, peak)]
            
            filtered_ts, filtered_contrasts = self._filter_by_date(
                timestamps, contrasts, start_date, end_date
            )
            all_timestamps.extend(filtered_ts)
            all_contrasts.extend(filtered_contrasts)
        
        if not all_timestamps:
            return {'timestamps': np.array([]), 'contrast': np.array([])}
        
        # Sort by timestamp
        sorted_pairs = sorted(zip(all_timestamps, all_contrasts), key=lambda x: x[0])
        timestamps, contrasts = zip(*sorted_pairs)
        
        return {
            'timestamps': np.array(timestamps),
            'contrast': np.array(contrasts)
        }
    
    def inspect_laser_locked(self,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> Dict:
        """
        Get periods when laser was locked.
        
        Args:
            start_date: Optional start date in format 'yymmdd:hhmm'
            end_date: Optional end date in format 'yymmdd:hhmm'
        
        Returns:
            Dictionary with 'locked_periods' and 'file_boundaries'
        """
        all_periods = []
        file_boundaries = []
        
        for log_file, data in self.parsed_data.items():
            file_boundaries.append((log_file, data['file_start'], data['file_end']))
            
            # Filter periods by date
            for start, end in data['laser_locked']:
                if start_date:
                    start_dt = datetime.strptime(start_date, '%y%m%d:%H%M')
                    if end < start_dt:
                        continue
                if end_date:
                    end_dt = datetime.strptime(end_date, '%y%m%d:%H%M')
                    if start > end_dt:
                        continue
                
                all_periods.append((start, end))
        
        return {
            'locked_periods': all_periods,
            'file_boundaries': file_boundaries
        }
    
    def inspect_cavity_locked(self,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> Dict:
        """
        Get periods when cavity was locked.
        
        Args:
            start_date: Optional start date in format 'yymmdd:hhmm'
            end_date: Optional end date in format 'yymmdd:hhmm'
        
        Returns:
            Dictionary with 'locked_periods' and 'file_boundaries'
        """
        all_periods = []
        file_boundaries = []
        
        for log_file, data in self.parsed_data.items():
            file_boundaries.append((log_file, data['file_start'], data['file_end']))
            
            # Filter periods by date
            for start, end in data['cavity_locked']:
                if start_date:
                    start_dt = datetime.strptime(start_date, '%y%m%d:%H%M')
                    if end < start_dt:
                        continue
                if end_date:
                    end_dt = datetime.strptime(end_date, '%y%m%d:%H%M')
                    if start > end_dt:
                        continue
                
                all_periods.append((start, end))
        
        return {
            'locked_periods': all_periods,
            'file_boundaries': file_boundaries
        }
    
    def get_lock_stability_statistics(self,
                                     parameter: str = 'laser_offset',
                                     start_date: Optional[str] = None,
                                     end_date: Optional[str] = None) -> Dict:
        """
        Calculate stability statistics (mean, std, min, max) for a parameter.
        
        Args:
            parameter: One of 'in_pulse', 'peak', 'laser_offset', 'cavity_rms', 'cavity_position'
            start_date: Optional start date in format 'yymmdd:hhmm'
            end_date: Optional end date in format 'yymmdd:hhmm'
        
        Returns:
            Dictionary with 'mean', 'std', 'min', 'max' values
        """
        time_series = self.get_time_series(parameter, start_date, end_date)
        values = time_series['values']
        
        if len(values) == 0:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}
        
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'count': len(values)
        }
    
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
    analyzer = AdaptiveLockLogAnalyzer("./adaptive_laser_lock_logs")
    
    # Get laser offset over time
    laser_offset = analyzer.get_time_series('laser_offset')
    print(f"Laser offset: {len(laser_offset['timestamps'])} points")
    
    # Get contrast over time
    contrast = analyzer.get_contrast_vs_time()
    print(f"Contrast: {len(contrast['timestamps'])} points")
    
    # Get lock periods
    laser_locked = analyzer.inspect_laser_locked()
    print(f"Laser locked: {len(laser_locked['locked_periods'])} periods")
    
    cavity_locked = analyzer.inspect_cavity_locked()
    print(f"Cavity locked: {len(cavity_locked['locked_periods'])} periods")
    
    # Get stability statistics
    stats = analyzer.get_lock_stability_statistics('laser_offset')
    print(f"Laser offset stability: mean={stats['mean']:.4f}V, std={stats['std']:.4f}V")
    
    # Get initial values
    initial = analyzer.initial_values()
    print(f"Initial values from {len(initial)} files")
