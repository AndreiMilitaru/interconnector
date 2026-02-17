"""
Cavity Control Log Analysis
Author: Andrei Militaru with GitHub Copilot
Organization: Institute of Science and Technology Austria (ISTA)
Date: February 2026

Description: Object-oriented data analysis tool for cavity control log files.
Extracts timestamps, parameter changes, and system states from log files.
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
import numpy as np


class CavityLogAnalyzer:
    """Analyzes cavity control log files to extract system events and parameters."""
    
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
                self.log_files = sorted(path.glob("cavity_control_*.log"))
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
            'offset_adjustment': [],  # List of (start, end) tuples
            'mode_finding': [],  # List of (start, end) tuples
            'pid_enabled': [],  # List of (time, state) tuples
            'offset_adjustment_state': [],  # List of (time, state) tuples
            'mode_finding_state': [],  # List of (time, state) tuples (not the routine, the checkbox)
            'reflection': [],  # List of (time, mean, std) tuples
            'fg_frequency': [],  # List of (time, value) tuples
            'fg_amplitude': [],  # List of (time, value) tuples
            'initial_values': {}
        }
        
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        offset_adj_start = None
        mode_finding_start = None
        
        for line in lines:
            timestamp = self._parse_timestamp(line)
            if not timestamp:
                continue
            
            # Track file start and end
            if data['file_start'] is None:
                data['file_start'] = timestamp
            data['file_end'] = timestamp
            
            # Parse initial values
            if 'Initial PID values:' in line:
                match = re.search(r'P=([\d.\-]+), I=([\d.\-]+), Bandwidth=([\d.]+) Hz, Enabled=(True|False), Keep I=(True|False)', line)
                if match:
                    data['initial_values']['pid_p'] = float(match.group(1))
                    data['initial_values']['pid_i'] = float(match.group(2))
                    data['initial_values']['pid_bandwidth'] = float(match.group(3))
                    data['initial_values']['pid_enabled'] = match.group(4) == 'True'
                    data['initial_values']['pid_keep_i'] = match.group(5) == 'True'
            
            if 'Initial dither values:' in line:
                match = re.search(r'Freq=([\d.]+) Hz, Strength=([\d.]+) mV, Phase=([\d.\-]+) deg, Enabled=(True|False)', line)
                if match:
                    data['initial_values']['dither_freq'] = float(match.group(1))
                    data['initial_values']['dither_strength'] = float(match.group(2))
                    data['initial_values']['dither_phase'] = float(match.group(3))
                    data['initial_values']['dither_enabled'] = match.group(4) == 'True'
            
            if 'Initial PID output offset:' in line:
                match = re.search(r'Initial PID output offset: ([\d.\-]+) V', line)
                if match:
                    data['initial_values']['pid_offset'] = float(match.group(1))
            
            if 'Initial slow offset:' in line:
                match = re.search(r'Initial slow offset: ([\d.\-]+) V', line)
                if match:
                    data['initial_values']['slow_offset'] = float(match.group(1))
            
            if 'Initial function generator values:' in line:
                match = re.search(r'Waveform=(\w+), Amplitude=([\d.]+) mV, Frequency=([\d.]+) Hz, Offset=([\d.\-]+) mV, Output=(True|False)', line)
                if match:
                    data['initial_values']['fg_waveform'] = match.group(1)
                    data['initial_values']['fg_amplitude'] = float(match.group(2))
                    data['initial_values']['fg_frequency'] = float(match.group(3))
                    data['initial_values']['fg_offset'] = float(match.group(4))
                    data['initial_values']['fg_output'] = match.group(5) == 'True'
            
            # Parse offset adjustment routine
            if 'Ramp slow offset' in line and 'started' in line:
                offset_adj_start = timestamp
            elif 'Ramp slow offset' in line and ('completed' in line or 'stopped' in line):
                if offset_adj_start:
                    data['offset_adjustment'].append((offset_adj_start, timestamp))
                    offset_adj_start = None
            
            # Parse mode finding routine
            if 'Mode finding started' in line or 'Find Mode clicked' in line:
                mode_finding_start = timestamp
            elif 'Mode finding' in line and ('completed' in line or 'stopped' in line or 'finished' in line):
                if mode_finding_start:
                    data['mode_finding'].append((mode_finding_start, timestamp))
                    mode_finding_start = None
            
            # Parse PID enable state changes
            if 'PID enable ->' in line:
                match = re.search(r'PID enable -> (True|False)', line)
                if match:
                    state = match.group(1) == 'True'
                    data['pid_enabled'].append((timestamp, state))
            
            # Parse offset adjustment checkbox state
            if 'Offset adjustment ->' in line or 'Auto offset management' in line:
                match = re.search(r'(True|False)', line)
                if match:
                    state = match.group(1) == 'True'
                    data['offset_adjustment_state'].append((timestamp, state))
            
            # Parse reflection measurements
            if 'Reflection waveform: mean=' in line:
                match = re.search(r'mean=([\d.\-]+) V, std=([\d.\-]+) V', line)
                if match:
                    mean = float(match.group(1))
                    std = float(match.group(2))
                    data['reflection'].append((timestamp, mean, std))
            
            # Parse FG frequency changes
            if 'FG frequency ->' in line:
                match = re.search(r'FG frequency -> ([\d.]+) Hz', line)
                if match:
                    freq = float(match.group(1))
                    data['fg_frequency'].append((timestamp, freq))
            
            # Parse FG amplitude changes
            if 'FG amplitude ->' in line:
                match = re.search(r'FG amplitude -> ([\d.]+) mV', line)
                if match:
                    amp = float(match.group(1))
                    data['fg_amplitude'].append((timestamp, amp))
        
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
    
    def inspect_offset_adjustment_routines(self, 
                                          start_date: Optional[str] = None,
                                          end_date: Optional[str] = None) -> Dict:
        """
        Inspect when offset adjustment routines were started and ended.
        
        Args:
            start_date: Optional start date in format 'yymmdd:hhmm'
            end_date: Optional end date in format 'yymmdd:hhmm'
        
        Returns:
            Dictionary with 'routines' (list of start/end times) and 
            'file_boundaries' (list of file start/end times)
        """
        all_routines = []
        file_boundaries = []
        
        for log_file, data in self.parsed_data.items():
            file_boundaries.append((log_file, data['file_start'], data['file_end']))
            routines = self._filter_by_date(data['offset_adjustment'], start_date, end_date)
            all_routines.extend(routines)
        
        return {
            'routines': all_routines,
            'file_boundaries': file_boundaries
        }
    
    def inspect_mode_finding_routines(self,
                                     start_date: Optional[str] = None,
                                     end_date: Optional[str] = None) -> Dict:
        """
        Inspect when mode finding routines were started and ended.
        
        Args:
            start_date: Optional start date in format 'yymmdd:hhmm'
            end_date: Optional end date in format 'yymmdd:hhmm'
        
        Returns:
            Dictionary with 'routines' (list of start/end times) and 
            'file_boundaries' (list of file start/end times)
        """
        all_routines = []
        file_boundaries = []
        
        for log_file, data in self.parsed_data.items():
            file_boundaries.append((log_file, data['file_start'], data['file_end']))
            routines = self._filter_by_date(data['mode_finding'], start_date, end_date)
            all_routines.extend(routines)
        
        return {
            'routines': all_routines,
            'file_boundaries': file_boundaries
        }
    
    def inspect_cavity_locked(self,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> Dict:
        """
        Inspect when cavity was locked (combining mode finding and offset adjustment).
        Cavity is considered locked when either mode finding or offset adjustment is active.
        
        Args:
            start_date: Optional start date in format 'yymmdd:hhmm'
            end_date: Optional end date in format 'yymmdd:hhmm'
        
        Returns:
            Dictionary with 'locked_periods' (merged periods when cavity was locked)
        """
        offset_data = self.inspect_offset_adjustment_routines(start_date, end_date)
        mode_data = self.inspect_mode_finding_routines(start_date, end_date)
        
        # Combine all events
        all_periods = offset_data['routines'] + mode_data['routines']
        
        if not all_periods:
            return {'locked_periods': []}
        
        # Sort by start time
        all_periods.sort(key=lambda x: x[0])
        
        # Merge overlapping periods
        merged = [all_periods[0]]
        for current in all_periods[1:]:
            last = merged[-1]
            if current[0] <= last[1]:  # Overlapping
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        
        return {'locked_periods': merged}
    
    def inspect_offset_adjustment_off(self,
                                     start_date: Optional[str] = None,
                                     end_date: Optional[str] = None) -> List[Tuple]:
        """
        Return periods when offset adjustment checkbox was off.
        
        Args:
            start_date: Optional start date in format 'yymmdd:hhmm'
            end_date: Optional end date in format 'yymmdd:hhmm'
        
        Returns:
            List of (start_time, end_time) tuples when offset adjustment was off
        """
        periods = []
        
        for log_file, data in self.parsed_data.items():
            states = self._filter_by_date(data['offset_adjustment_state'], start_date, end_date)
            
            if not states:
                continue
            
            # Build off periods
            current_state = False  # Assume starts off
            period_start = data['file_start']
            
            for timestamp, state in states:
                if state and not current_state:  # Turning on
                    if period_start:
                        periods.append((period_start, timestamp))
                    current_state = True
                elif not state and current_state:  # Turning off
                    period_start = timestamp
                    current_state = False
            
            # If ended in off state
            if not current_state and period_start:
                periods.append((period_start, data['file_end']))
        
        return periods
    
    def inspect_mode_finding_off(self,
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None) -> List[Tuple]:
        """
        Return periods when mode finding checkbox was off.
        Note: This is different from mode finding routines - it's the GUI checkbox state.
        
        Args:
            start_date: Optional start date in format 'yymmdd:hhmm'
            end_date: Optional end date in format 'yymmdd:hhmm'
        
        Returns:
            List of (start_time, end_time) tuples when mode finding was off
        """
        periods = []
        
        for log_file, data in self.parsed_data.items():
            states = self._filter_by_date(data['mode_finding_state'], start_date, end_date)
            
            if not states:
                # If no state changes recorded, assume it was off the whole time
                periods.append((data['file_start'], data['file_end']))
                continue
            
            # Build off periods
            current_state = False  # Assume starts off
            period_start = data['file_start']
            
            for timestamp, state in states:
                if state and not current_state:  # Turning on
                    if period_start:
                        periods.append((period_start, timestamp))
                    current_state = True
                elif not state and current_state:  # Turning off
                    period_start = timestamp
                    current_state = False
            
            # If ended in off state
            if not current_state and period_start:
                periods.append((period_start, data['file_end']))
        
        return periods
    
    def inspect_pid_enabled(self,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> List[Tuple]:
        """
        Return periods when PID was enabled.
        
        Args:
            start_date: Optional start date in format 'yymmdd:hhmm'
            end_date: Optional end date in format 'yymmdd:hhmm'
        
        Returns:
            List of (start_time, end_time) tuples when PID was enabled
        """
        periods = []
        
        for log_file, data in self.parsed_data.items():
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
        
        return periods
    
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
    
    def get_fg_changes(self,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> Dict:
        """
        Get all changes in function generator frequency and amplitude over time.
        
        Args:
            start_date: Optional start date in format 'yymmdd:hhmm'
            end_date: Optional end date in format 'yymmdd:hhmm'
        
        Returns:
            Dictionary with 'frequency' and 'amplitude' sub-dictionaries,
            each containing 'timestamps' and 'values' arrays
        """
        all_freq = []
        all_amp = []
        
        for log_file, data in self.parsed_data.items():
            freq_changes = self._filter_by_date(data['fg_frequency'], start_date, end_date)
            amp_changes = self._filter_by_date(data['fg_amplitude'], start_date, end_date)
            all_freq.extend(freq_changes)
            all_amp.extend(amp_changes)
        
        # Sort by timestamp
        all_freq.sort(key=lambda x: x[0])
        all_amp.sort(key=lambda x: x[0])
        
        freq_result = {
            'timestamps': np.array([item[0] for item in all_freq]) if all_freq else np.array([]),
            'values': np.array([item[1] for item in all_freq]) if all_freq else np.array([])
        }
        
        amp_result = {
            'timestamps': np.array([item[0] for item in all_amp]) if all_amp else np.array([]),
            'values': np.array([item[1] for item in all_amp]) if all_amp else np.array([])
        }
        
        return {
            'frequency': freq_result,
            'amplitude': amp_result
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
    analyzer = CavityLogAnalyzer("./cavity_logs")
    
    # Get offset adjustment routines
    offset_routines = analyzer.inspect_offset_adjustment_routines()
    print(f"Found {len(offset_routines['routines'])} offset adjustment routines")
    
    # Get mode finding routines
    mode_routines = analyzer.inspect_mode_finding_routines()
    print(f"Found {len(mode_routines['routines'])} mode finding routines")
    
    # Get locked periods
    locked = analyzer.inspect_cavity_locked()
    print(f"Found {len(locked['locked_periods'])} locked periods")
    
    # Get initial values
    initial = analyzer.initial_values()
    print(f"Initial values from {len(initial)} files")
