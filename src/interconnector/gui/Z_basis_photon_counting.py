"""
Z-Basis Photon Counting with Marker Detection (Optimized)

Author: Andrei Militaru with Github Copilot
Date: 3rd February 2026

Description: 
    Z-basis photon counting with three marker types (Noise, G state, E state)
    and temporal binning. Fully vectorized processing with modular architecture.
"""

from interconnector.libraries.snAPI.Main import *
import matplotlib
import numpy as np
matplotlib.use('TkAgg', force=True)
from matplotlib import pyplot as plt
print("Switched to:", matplotlib.get_backend())
import time
import h5py
import os
from matplotlib.widgets import Button, TextBox
from datetime import datetime
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
SAVE_DATA = True
DETECT_MARKERS = True
SAVE_SECONDS_PER_DATASET = 100
NAME_OF_EXP = "ZZ_100Hz_65mV"
RUNNING_MEAN_SIZE = 300

# Time window configuration (in picoseconds)
START_OFFSET = 1000 * 4000
DETECTION_LENGTH = 1000 * 4500
BIN1_OFFSET = 1000 * 1500  # from start offset
BIN1_LENGTH = 1000 * 500
BIN2_OFFSET = 1000 * 2500  # from start offset
BIN2_LENGTH = 1000 * 500
BIN_SIZE = 250  # picoseconds for time trace

# Marker channel IDs
SYNC_CHANNEL = 0
PHOTON_CHANNEL_1 = 1
PHOTON_CHANNEL_2 = 2
MARKER_1_ID = 128  # Noise marker
MARKER_2_ID = 131  # State marker


# ============================================================================
# DATA CLASSES
# ============================================================================
class CountRateBuffer:
    """Circular buffer for running mean calculation of signal and noise."""
    
    def __init__(self, size):
        self.size = size
        self.signal = np.zeros(size)  # G + E states
        self.noise = np.zeros(size)   # Noise state
        self.index_signal = 0
        self.index_noise = 0
    
    def update(self, bin_counts):
        """
        Add new counts to the buffer.
        
        Args:
            bin_counts: Array [N1, N2, G1, G2, E1, E2]
        """
        self.noise[self.index_noise] = np.sum(bin_counts[:2])
        self.signal[self.index_signal] = np.sum(bin_counts[2:])
        
        self.index_signal = (self.index_signal + 1) % self.size
        self.index_noise = (self.index_noise + 1) % self.size
    
    def get_means(self):
        """Return mean values for signal and noise."""
        return {
            'signal': np.mean(self.signal),
            'noise': np.mean(self.noise)
        }


class TimeTraceData:
    """Manages time trace histograms for Z-basis measurement."""
    
    def __init__(self, num_bins):
        self.x = np.arange(0, num_bins) * BIN_SIZE / 1E3  # in ns
        self.y1_signal = np.zeros(num_bins)  # Ch1 signal (G + E)
        self.y2_signal = np.zeros(num_bins)  # Ch2 signal (G + E)
        self.y1_noise = np.zeros(num_bins)   # Ch1 noise
        self.y2_noise = np.zeros(num_bins)   # Ch2 noise
    
    def clear(self):
        """Reset all histograms to zero."""
        self.y1_signal[:] = 0
        self.y2_signal[:] = 0
        self.y1_noise[:] = 0
        self.y2_noise[:] = 0
    
    def increment(self, channel, bin_idx, marker_type):
        """Increment histogram bin based on channel and marker type."""
        if channel == PHOTON_CHANNEL_1:
            if marker_type == 1:  # Noise
                self.y1_noise[bin_idx] += 1
            elif marker_type > 1:  # G or E state
                self.y1_signal[bin_idx] += 1
        elif channel == PHOTON_CHANNEL_2:
            if marker_type == 1:  # Noise
                self.y2_noise[bin_idx] += 1
            elif marker_type > 1:  # G or E state
                self.y2_signal[bin_idx] += 1
    
    def get_decimated(self, n_decim=40):
        """Return decimated traces for plotting."""
        length = (len(self.x) // n_decim) * n_decim
        x_avg = np.min(np.reshape(self.x[:length], (-1, n_decim)), axis=1)
        
        y1_signal_avg = np.sum(np.reshape(self.y1_signal[:length], (-1, n_decim)), axis=1)
        y2_signal_avg = np.sum(np.reshape(self.y2_signal[:length], (-1, n_decim)), axis=1)
        y1_noise_avg = np.sum(np.reshape(self.y1_noise[:length], (-1, n_decim)), axis=1)
        y2_noise_avg = np.sum(np.reshape(self.y2_noise[:length], (-1, n_decim)), axis=1)
        
        return x_avg, y1_signal_avg, y2_signal_avg, y1_noise_avg, y2_noise_avg


class RecordingState:
    """Manages recording state and file operations."""
    
    def __init__(self):
        self.is_recording = False
        self.file = None
        self.dataset_count = 0
    
    def start(self, filename):
        """Start recording to a new file."""
        if self.is_recording:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Already recording!")
            return False
        
        self.file = h5py.File(f"{filename}.h5", "w")
        self.is_recording = True
        self.dataset_count = 0
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Started recording to {filename}.h5")
        return True
    
    def stop(self):
        """Stop recording and close file."""
        if self.is_recording and self.file is not None:
            self.file.close()
            self.is_recording = False
            self.file = None
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Stopped recording.")
    
    def save_dataset(self, data, lock_log):
        """Save dataset and lock log if recording is active."""
        if self.is_recording and self.file is not None:
            self.file.create_dataset(str(self.dataset_count), data=data)
            if "lock_log" in self.file:
                del self.file["lock_log"]
            self.file.create_dataset("lock_log", data=lock_log)
            self.dataset_count += 1


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================
def classify_marker_type(marker1_detected, marker2_detected):
    """
    Classify marker type based on combination.
    
    Args:
        marker1_detected: Boolean array
        marker2_detected: Boolean array
    
    Returns:
        marker_type: Integer array (1=Noise, 2=G state, 3=E state, 0=none)
    """
    marker_type = np.zeros(len(marker1_detected), dtype=int)
    
    # E state: both markers (128 and 131)
    both = marker1_detected & marker2_detected
    marker_type[both] = 3
    
    # G state: only marker 2 (131)
    only_2 = marker2_detected & ~marker1_detected
    marker_type[only_2] = 2
    
    # Noise: only marker 1 (128)
    only_1 = marker1_detected & ~marker2_detected
    marker_type[only_1] = 1
    
    return marker_type


def process_data_stream_vectorized(channels, times, timetrace, data_to_save, 
                                   seconds_count, unix_timestamp):
    """
    Process the entire data stream using vectorized NumPy operations.
    
    Returns:
        tuple: (bin_counts[6], bin_count1, bin_count2,
                syncs_processed, n_markers, g_markers, e_markers, markers_missed)
    """
    # Convert to numpy arrays
    channels = np.asarray(channels)
    times = np.asarray(times)
    
    # Find all sync pulses
    sync_mask = (channels == SYNC_CHANNEL)
    sync_indices = np.where(sync_mask)[0]
    syncs_processed = len(sync_indices)
    
    if syncs_processed == 0:
        return (np.zeros(6), 0, 0, 0, 0, 0, 0, 0)
    
    # Find all markers
    marker1_mask = (channels == MARKER_1_ID)
    marker2_mask = (channels == MARKER_2_ID)
    marker1_indices = np.where(marker1_mask)[0]
    marker2_indices = np.where(marker2_mask)[0]
    
    # Assign marker types to each sync
    markers_for_syncs = np.zeros(syncs_processed, dtype=int)
    
    for idx, sync_idx in enumerate(sync_indices):
        # Find next sync
        if idx < syncs_processed - 1:
            next_sync = sync_indices[idx + 1]
        else:
            next_sync = len(channels)
        
        # Find markers between this sync and next
        m1_between = marker1_indices[(marker1_indices > sync_idx) & (marker1_indices < next_sync)]
        m2_between = marker2_indices[(marker2_indices > sync_idx) & (marker2_indices < next_sync)]
        
        has_m1 = len(m1_between) > 0
        has_m2 = len(m2_between) > 0
        
        if has_m1 and has_m2:
            markers_for_syncs[idx] = 3  # E state
        elif has_m2:
            markers_for_syncs[idx] = 2  # G state
        elif has_m1:
            markers_for_syncs[idx] = 1  # Noise
    
    # Count marker types
    n_markers = np.sum(markers_for_syncs == 1)
    g_markers = np.sum(markers_for_syncs == 2)
    e_markers = np.sum(markers_for_syncs == 3)
    markers_missed = np.sum(markers_for_syncs == 0)
    
    # Find all photons (both channels)
    photon1_mask = (channels == PHOTON_CHANNEL_1)
    photon2_mask = (channels == PHOTON_CHANNEL_2)
    photon1_indices = np.where(photon1_mask)[0]
    photon2_indices = np.where(photon2_mask)[0]
    
    # Initialize bin counts [N1, N2, G1, G2, E1, E2]
    bin_counts = np.zeros(6)
    bin_count1 = 0
    bin_count2 = 0
    
    # Process Channel 1 photons
    if len(photon1_indices) > 0:
        bin_count1, bin_counts_ch1 = process_channel_photons(
            photon1_indices, times, sync_indices, markers_for_syncs,
            PHOTON_CHANNEL_1, timetrace, data_to_save, 
            seconds_count, unix_timestamp
        )
        bin_counts += bin_counts_ch1
    
    # Process Channel 2 photons
    if len(photon2_indices) > 0:
        bin_count2, bin_counts_ch2 = process_channel_photons(
            photon2_indices, times, sync_indices, markers_for_syncs,
            PHOTON_CHANNEL_2, timetrace, data_to_save, 
            seconds_count, unix_timestamp
        )
        bin_counts += bin_counts_ch2
    
    return (bin_counts, bin_count1, bin_count2,
            syncs_processed, n_markers, g_markers, e_markers, markers_missed)


def process_channel_photons(photon_indices, times, sync_indices, 
                            markers_for_syncs, channel, timetrace, 
                            data_to_save, seconds_count, unix_timestamp):
    """Process photons for a single channel."""
    photon_times = times[photon_indices]
    
    # Assign each photon to a sync
    sync_assignment = np.searchsorted(sync_indices, photon_indices, side='right') - 1
    valid_photons = sync_assignment >= 0
    
    if not np.any(valid_photons):
        return 0, np.zeros(6)
    
    # Filter valid photons
    photon_indices = photon_indices[valid_photons]
    photon_times = photon_times[valid_photons]
    sync_assignment = sync_assignment[valid_photons]
    
    # Get sync times and marker types
    photon_sync_indices = sync_indices[sync_assignment]
    photon_sync_times = times[photon_sync_indices]
    photon_marker_types = markers_for_syncs[sync_assignment]
    
    # Calculate relative times
    relative_times = photon_times - photon_sync_times
    
    # Filter by detection window
    in_window = ((relative_times >= START_OFFSET) & 
                 (relative_times <= (START_OFFSET + DETECTION_LENGTH)))
    
    if not np.any(in_window):
        return 0, np.zeros(6)
    
    # Apply window filter
    photon_times_in = photon_times[in_window]
    relative_times_in = relative_times[in_window]
    sync_assignment_in = sync_assignment[in_window]
    photon_sync_times_in = photon_sync_times[in_window]
    photon_marker_types_in = photon_marker_types[in_window]
    
    # Calculate time trace bin indices
    timetrace_bins = ((relative_times_in - START_OFFSET) // BIN_SIZE).astype(int)
    timetrace_bins = np.clip(timetrace_bins, 0, len(timetrace.x) - 1)
    
    # Update time trace histograms
    for marker_type in [1, 2, 3]:
        mask = photon_marker_types_in == marker_type
        if np.any(mask):
            bins_for_marker = timetrace_bins[mask]
            if channel == PHOTON_CHANNEL_1:
                if marker_type == 1:
                    np.add.at(timetrace.y1_noise, bins_for_marker, 1)
                else:
                    np.add.at(timetrace.y1_signal, bins_for_marker, 1)
            elif channel == PHOTON_CHANNEL_2:
                if marker_type == 1:
                    np.add.at(timetrace.y2_noise, bins_for_marker, 1)
                else:
                    np.add.at(timetrace.y2_signal, bins_for_marker, 1)
    
    # Check temporal bins (bin1 and bin2)
    in_bin1 = ((relative_times_in >= (START_OFFSET + BIN1_OFFSET)) &
               (relative_times_in <= (START_OFFSET + BIN1_OFFSET + BIN1_LENGTH)))
    in_bin2 = ((relative_times_in >= (START_OFFSET + BIN2_OFFSET)) &
               (relative_times_in <= (START_OFFSET + BIN2_OFFSET + BIN2_LENGTH)))
    
    # Count photons by marker type and temporal bin
    bin_counts = np.zeros(6)  # [N1, N2, G1, G2, E1, E2]
    
    for marker_type in [1, 2, 3]:
        marker_mask = photon_marker_types_in == marker_type
        
        # Early bin (index 0 for each marker type)
        bin_counts[2*(marker_type-1)] = np.sum(marker_mask & in_bin1)
        
        # Late bin (index 1 for each marker type)
        bin_counts[2*(marker_type-1) + 1] = np.sum(marker_mask & in_bin2)
    
    # Save data if enabled
    if SAVE_DATA:
        sync_numbers = sync_assignment_in + 1  # 1-indexed
        for i in range(len(timetrace_bins)):
            data_to_save.append([seconds_count, sync_numbers[i], 
                               channel, photon_marker_types_in[i], 
                               timetrace_bins[i], unix_timestamp])
    
    return len(photon_times_in), bin_counts


# ============================================================================
# PLOT MANAGEMENT
# ============================================================================
class PlotManager:
    """Manages plot updates efficiently for Z-basis measurement."""
    
    def __init__(self, ax1, ax2, ax4, ax5):
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax4 = ax4
        self.ax5 = ax5
        
        # Bar plot for bins (will use ax1.bar repeatedly)
        ax1.set_xlabel('Bin number')
        ax1.set_ylabel('# of Coincidences')
        ax1.set_ylim(bottom=0)  # Set lower bound to 0 for coincidence counts
        
        # Time traces
        self.line_signal, = ax2.plot([], [], label='Signal', color='tab:orange')
        self.line_noise, = ax2.plot([], [], label='Noise', color='tab:blue')
        ax2.set_ylabel('Counts')
        ax2.set_xlabel('Time (ns)')
        ax2.set_ylim(bottom=0)  # Set lower bound to 0 for photon counts
        ax2.legend(loc=0)
        
        # Vertical lines for bin boundaries
        self.vline1 = ax2.axvline(x=BIN1_OFFSET/1000, ls='--', color='grey', alpha=0.8)
        self.vline2 = ax2.axvline(x=(BIN1_OFFSET+BIN1_LENGTH)/1000, ls='--', color='grey', alpha=0.8)
        self.vline3 = ax2.axvline(x=BIN2_OFFSET/1000, ls='--', color='grey', alpha=0.8)
        self.vline4 = ax2.axvline(x=(BIN2_OFFSET+BIN2_LENGTH)/1000, ls='--', color='grey', alpha=0.8)
        
        # Running counts
        self.line_noise_rate, = ax4.plot([], [], alpha=0.7, label='Noise')
        self.line_signal_rate, = ax4.plot([], [], label='Signal')
        ax4.grid(ls='--', alpha=0.5)
        ax4.legend(loc=0)
        ax4.set_xlabel('seconds')
        ax4.set_ylabel('Avg count rate')
        ax4.set_ylim(bottom=0)  # Set lower bound to 0 for count rates
        
        self.line_total, = ax5.plot([], [], color='grey', alpha=0.5)
        ax5.set_ylim(bottom=0)  # Set lower bound to 0 for total counts
        
        # Add tracking for ax5 y-limits
        self.ax5_ylim = [0, 1]
        self.first_update = True
    
    def update_bin_plot(self, bin_count_total, sync_count, ch1_count, ch2_count):
        """Update bar plot with bin counts."""
        self.ax1.clear()
        self.ax1.bar(['N1', 'N2', 'G1', 'G2', 'E1', 'E2'], 
                     bin_count_total, width=0.5)
        
        # Calculate E_z values
        if (bin_count_total[2] + bin_count_total[3]) == 0:
            E_z_g = 0
        else:
            E_z_g = bin_count_total[3] / (bin_count_total[3] + bin_count_total[2])
        
        if (bin_count_total[4] + bin_count_total[5]) == 0:
            E_z_e = 0
        else:
            E_z_e = bin_count_total[4] / (bin_count_total[4] + bin_count_total[5])
        
        self.ax1.set_title(f'Sync: {sync_count}, Ch1: {ch1_count}, Ch2: {ch2_count}, '
                          f'E_z_g={E_z_g:.2f}, E_z_e={E_z_e:.2f}')
        self.ax1.set_xlabel('Bin number')
        self.ax1.set_ylabel('# of Coincidences')
        self.ax1.set_ylim(bottom=0)  # Ensure lower bound stays at 0 after clear
    
    def update_timetrace(self, timetrace, bin_count1, bin_count2, means):
        """Update time trace plot."""
        x_avg, y1_sig, y2_sig, y1_noise, y2_noise = timetrace.get_decimated()
        
        self.line_signal.set_data(x_avg, y1_sig + y2_sig)
        self.line_noise.set_data(x_avg, y1_noise + y2_noise)
        
        self.ax2.set_title(f"Ch1: {bin_count1}, Ch2: {bin_count2}, "
                          f"Signal: {means['signal']:.2f}/s, "
                          f"Noise: {means['noise']:.2f}/s")
        
        # Auto-scale if needed, starting from 0
        if len(x_avg) > 0:
            # Get max y value across both traces
            y_max = max(np.max(y1_sig + y2_sig), np.max(y1_noise + y2_noise), 1)
            
            # Set x limits
            self.ax2.set_xlim(0, np.max(x_avg) * 1.05)
            
            # Set y limits: 0 to slightly above max
            self.ax2.set_ylim(0, y_max * 1.1)

    def update_running_plot(self, running_noise, running_signal, lock_log_plot):
        """Update running count rate plot."""
        x_data = np.arange(len(running_noise))
        
        self.line_noise_rate.set_data(x_data, running_noise)
        self.line_signal_rate.set_data(x_data, running_signal)
        
        x_counts = np.arange(len(lock_log_plot))
        self.line_total.set_data(x_counts, lock_log_plot)
        
        # Auto-scale if needed, but keep y minimum at 0
        if len(x_data) > 0:
            self.ax4.relim()
            self.ax4.autoscale_view()
            self.ax4.set_ylim(bottom=0)  # Ensure lower bound stays at 0
            
        # Properly auto-scale ax5 with explicit limits
        if len(lock_log_plot) > 0:
            y_max_secondary = max(lock_log_plot)
            
            # Force initial scaling on first update
            if self.first_update:
                self.ax5_ylim = [0, y_max_secondary * 1.1]
                self.ax5.set_ylim(self.ax5_ylim)
                self.first_update = False
            else:
                # Auto-scale if data exceeds current limits
                if y_max_secondary > self.ax5_ylim[1] * 0.9 or y_max_secondary < self.ax5_ylim[1] * 0.3:
                    self.ax5_ylim = [0, y_max_secondary * 1.1]
                    self.ax5.set_ylim(self.ax5_ylim)


# ============================================================================
# FILE I/O
# ============================================================================
def setup_data_directory():
    """Create and enter data directory inside photon_counting_data.
    
    Returns:
        tuple: (absolute path to parent qubit_photon_counting_data dir, 
                absolute path to timestamped data dir)
    """
    # Get absolute path to qubit_photon_counting_data at original cwd
    original_cwd = Path.cwd()
    parent_dir = original_cwd / "qubit_photon_counting_data"
    parent_dir.mkdir(exist_ok=True)
    
    # Create the timestamped data directory inside qubit_photon_counting_data
    new_dir_name = time.strftime("%Y_%m_%d_%H%M_", time.localtime()) + NAME_OF_EXP
    new_dir = parent_dir / new_dir_name
    new_dir.mkdir(exist_ok=True)
    
    # Copy this script to data directory for reference
    script_file = Path(__file__)
    if script_file.exists():
        import shutil
        try:
            shutil.copy2(script_file, new_dir / script_file.name)
        except Exception:
            pass  # Silently ignore copy errors
    
    # Change to the timestamped data directory for file operations
    os.chdir(new_dir)
    print(f"Data directory: {new_dir}")
    
    return str(parent_dir), str(new_dir)


def save_main_data(data_to_save, lock_log, dataset_count):
    """Save data to main HDF5 file."""
    with h5py.File('data', 'a') as file:
        file.create_dataset(str(dataset_count), data=data_to_save)
        
        if "lock_log" in file:
            del file["lock_log"]
        file.create_dataset("lock_log", data=lock_log)


# ============================================================================
# DEVICE SETUP
# ============================================================================
def setup_device(ptu_path=None):
    """Initialize and configure the time tagger device.
    
    Args:
        ptu_path: Optional absolute path for PTU files. If None, uses a temp directory.
    """
    sn = snAPI(libType=LibType.TH260)
    sn.getDevice()
    
    sn.setLogLevel(LogLevel.DataFile, True)
    sn.initDevice(MeasMode.T2)
    
    sn.setLogLevel(logLevel=LogLevel.Config, onOff=False)
    sn.setLogLevel(logLevel=LogLevel.Api, onOff=False)
    sn.setLogLevel(logLevel=LogLevel.Device, onOff=False)
    sn.setLogLevel(logLevel=LogLevel.Manipulators, onOff=False)
    sn.setLogLevel(logLevel=LogLevel.DataFile, onOff=False)
    
    # Get config file path relative to this module
    config_path = Path(__file__).parent.parent / "libraries" / "config" / "TH260N.ini"
    sn.loadIniConfig(str(config_path))
    
    # Set PTU file path (required even if not saving PTU files)
    # Suppress stderr to hide the harmless "file not found" warning
    try:
        if ptu_path is None:
            ptu_path = Path.cwd() / "temp_ptu"
        ptu_path = Path(ptu_path)
        ptu_path.mkdir(parents=True, exist_ok=True)
        
        import sys
        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        try:
            sn.setPTUFilePath(str(ptu_path / "temp"))
        finally:
            sys.stderr.close()
            sys.stderr = old_stderr
    except Exception:
        pass  # Silently ignore PTU path errors
    
    num_chans = sn.deviceConfig["NumChans"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'[{timestamp}] Num channels: {num_chans}')
    
    return sn


# ============================================================================
# GUI SETUP
# ============================================================================
def setup_gui(timetrace, reset_flag, recording_state):
    """Setup matplotlib GUI with buttons and plots."""
    fig, (ax1, ax2, ax4) = plt.subplots(3, 1, layout='constrained', 
                                        figsize=(6, 10))
    
    # Configure axes
    ax5 = ax4.twinx()
    
    # Create plot manager
    plot_manager = PlotManager(ax1, ax2, ax4, ax5)
    
    # Clear button
    def clear_callback(event):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'[{timestamp}] Clearing...')
        timetrace.clear()
        reset_flag[0] = True
    
    ax_clear = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    btn_clear = Button(ax_clear, 'Clear')
    btn_clear.on_clicked(clear_callback)
    
    # Recording controls
    ax_text = fig.add_axes([0.18, 0.05, 0.2, 0.03])
    ax_start = fig.add_axes([0.41, 0.05, 0.08, 0.03])
    ax_stop = fig.add_axes([0.51, 0.05, 0.08, 0.03])
    
    text_box = TextBox(ax_text, 'Name:')
    btn_start = Button(ax_start, 'Start')
    btn_stop = Button(ax_stop, 'Stop')
    
    def start_callback(event):
        filename = text_box.text.strip().replace(' ', '_')
        if not filename:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Please enter a filename before starting recording.")
            return
        recording_state.start(filename)
    
    def stop_callback(event):
        recording_state.stop()
    
    btn_start.on_clicked(start_callback)
    btn_stop.on_clicked(stop_callback)
    
    return fig, plot_manager


# ============================================================================
# MAIN LOOP
# ============================================================================
def main():
    """Main acquisition and processing loop."""
    sn = None
    
    try:
        # Setup data directory first if saving
        ptu_path = None
        if SAVE_DATA:
            ptu_parent, data_dir = setup_data_directory()
            ptu_path = ptu_parent  # Use the photon_counting_data directory for PTU files
        
        # Setup device with PTU path
        sn = setup_device(ptu_path)
        
        # Initialize data structures
        num_bins = 2 + (DETECTION_LENGTH // BIN_SIZE)
        timetrace = TimeTraceData(num_bins)
        count_buffer = CountRateBuffer(RUNNING_MEAN_SIZE)
        recording_state = RecordingState()
    
        # Running statistics
        running_count_rate_signal = []
        running_count_rate_noise = []
        lock_log = []
        lock_log_plot = []
        
        # Cumulative bin counts
        bin_count_total = np.zeros(6)
        
        # Data saving
        data_to_save = []
        dataset_count = 0
        seconds_count = 0
        
        # GUI
        reset_flag = [False]
        fig, plot_manager = setup_gui(timetrace, reset_flag, recording_state)
        
        # Start first measurement
        sn.unfold.measure(1000, waitFinished=False, savePTU=False)
        
        print("\nAcquisition started. Press Ctrl+C to stop.\n")
        
        # Main acquisition loop
        while True:
            unix_timestamp = datetime.now().timestamp()
            
            # Wait for measurement
            while not sn.unfold.isFinished():
                continue
            
            # Get data
            sync = sn.unfold.getTimesByChannel(0)
            ch1 = sn.unfold.getTimesByChannel(1)
            ch2 = sn.unfold.getTimesByChannel(2)
            times, channels = sn.unfold.getData()
            
            # Start next measurement
            sn.unfold.measure(1000, waitFinished=False, savePTU=False)
            
            # Process data (VECTORIZED)
            start_time = time.time()
            results = process_data_stream_vectorized(channels, times, timetrace, 
                                                    data_to_save, seconds_count,
                                                    unix_timestamp)
            bin_counts, bin_count1, bin_count2, syncs_processed, \
            n_markers, g_markers, e_markers, markers_missed = results
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update cumulative bin counts
            bin_count_total += bin_counts
            
            # Update statistics
            count_buffer.update(bin_counts)
            means = count_buffer.get_means()
            
            # Update running plots
            if reset_flag[0]:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f'[{timestamp}] Resetting running counts...')
                running_count_rate_signal = []
                running_count_rate_noise = []
                lock_log_plot = []
                bin_count_total[:] = 0
                reset_flag[0] = False
            
            running_count_rate_signal.append(means['signal'])
            running_count_rate_noise.append(means['noise'])
            
            total_counts = len(ch1) + len(ch2)
            lock_log.append(total_counts)
            lock_log_plot.append(total_counts)
            
            # Update plots
            plot_manager.update_bin_plot(bin_count_total, len(sync), len(ch1), len(ch2))
            plot_manager.update_timetrace(timetrace, bin_count1, bin_count2, means)
            plot_manager.update_running_plot(running_count_rate_noise, 
                                            running_count_rate_signal,
                                            lock_log_plot)
            
            # Print statistics
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Markers - Noise: {n_markers}, G: {g_markers}, "
                f"E: {e_markers}, Missed: {markers_missed}")
            print(f"[{timestamp}] Processing time: {processing_time:.1f}ms")
            print()
            
            plt.pause(0.0005)
            
            # Save data periodically
            if SAVE_DATA and seconds_count >= SAVE_SECONDS_PER_DATASET - 1:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f'[{timestamp}] Saving data...')
                
                recording_state.save_dataset(data_to_save, lock_log)
                save_main_data(data_to_save, lock_log, dataset_count)
                
                data_to_save = []
                dataset_count += 1
                seconds_count = -1
            
            seconds_count += 1
    
    except KeyboardInterrupt:
        print("\n\nStopping acquisition (Ctrl+C pressed)...)")
    
    except Exception as e:
        print(f"\n\nError in main loop: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("Cleaning up...")
        
        # Close recording file if open
        try:
            if 'recording_state' in locals():
                recording_state.stop()
        except Exception as e:
            print(f"Error stopping recording: {e}")
        
        # Close device
        if sn is not None:
            try:
                print("Closing device...")
                sn.closeDevice()
            except Exception as e:
                print(f"Error closing device: {e}")
        
        # Close plots
        try:
            plt.close('all')
        except Exception:
            pass
        
        print("Cleanup complete. Exiting.")


if __name__ == "__main__":
    main()