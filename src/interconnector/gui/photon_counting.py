"""
Author: Andrei Militaru with Github Copilot, adapted and optimized prior version 
by Rishabh Sahu. 
Date: 3rd February 2026

Description: Photon Counting with Marker Detection and Optimized Plotting
This script performs photon counting using a time tagger device, detects markers,
and visualizes the results in real-time with optimized plotting techniques.
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
from pathlib import Path
from matplotlib.widgets import Button, TextBox
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
SAVE_DATA = True
DETECT_MARKERS = True
SAVE_SECONDS_PER_DATASET = 100
NAME_OF_EXP = "SNR_1kHz_200ns_opt_6p6mV"
RUNNING_MEAN_SIZE = 100

# Time window configuration (in picoseconds)
WINDOW_START = 1000 * 4000
WINDOW_LENGTH = 1000 * 4500
BIN_SIZE = 250  # picoseconds

# Marker channel IDs
SYNC_CHANNEL = 0
PHOTON_CHANNEL_1 = 1
MARKER_1_ID = 128
MARKER_2_ID = 131


# ============================================================================
# DATA CLASSES
# ============================================================================
class CountRateBuffer:
    """Circular buffer for running mean calculation."""
    
    def __init__(self, size):
        self.size = size
        self.ch1 = np.zeros(size)
        self.ch1_m1 = np.zeros(size)
        self.ch1_m2 = np.zeros(size)
        self.index = 0
    
    def update(self, bin_count1, bin_count1_m1, bin_count1_m2):
        """Add new counts to the buffer."""
        self.ch1[self.index] = bin_count1
        self.ch1_m1[self.index] = bin_count1_m1
        self.ch1_m2[self.index] = bin_count1_m2
        self.index = (self.index + 1) % self.size
    
    def get_means(self):
        """Return mean values for all channels."""
        return {
            'ch1': np.mean(self.ch1),
            'ch1_m1': np.mean(self.ch1_m1),
            'ch1_m2': np.mean(self.ch1_m2)
        }


class TimeTraceData:
    """Manages time trace histograms."""
    
    def __init__(self, num_bins):
        self.x = np.arange(0, num_bins) * BIN_SIZE / 1E3  # in ns
        self.y1 = np.zeros(num_bins)
        self.y1_m1 = np.zeros(num_bins)
        self.y1_m2 = np.zeros(num_bins)
    
    def clear(self):
        """Reset all histograms to zero."""
        self.y1[:] = 0
        self.y1_m1[:] = 0
        self.y1_m2[:] = 0
    
    def increment(self, channel, bin_idx, marker):
        """Increment histogram bin."""
        if channel == PHOTON_CHANNEL_1:
            self.y1[bin_idx] += 1
            if marker == 1:
                self.y1_m1[bin_idx] += 1
            elif marker == 2:
                self.y1_m2[bin_idx] += 1
    
    def get_decimated(self, n_decim=40):
        """Return decimated traces for plotting."""
        length = (len(self.x) // n_decim) * n_decim # Trim to multiple of n_decim
        x_avg = np.min(np.reshape(self.x[:length], (-1, n_decim)), axis=1)
        y1_avg = np.sum(np.reshape(self.y1[:length], (-1, n_decim)), axis=1)
        y1_m1_avg = np.sum(np.reshape(self.y1_m1[:length], (-1, n_decim)), axis=1)
        y1_m2_avg = np.sum(np.reshape(self.y1_m2[:length], (-1, n_decim)), axis=1)
        
        return x_avg, y1_avg, y1_m1_avg, y1_m2_avg


class RecordingState:
    """Manages recording state and file operations."""
    
    def __init__(self):
        self.is_recording = False
        self.file = None
        self.dataset_count = 0
    
    def start(self, filename):
        """Start recording to a new file."""
        if self.is_recording:
            print("Already recording!")
            return False
        
        self.file = h5py.File(f"{filename}.h5", "w")
        self.is_recording = True
        self.dataset_count = 0
        print(f"Started recording to {filename}.h5")
        return True
    
    def stop(self):
        """Stop recording and close file."""
        if self.is_recording and self.file is not None:
            self.file.close()
            self.is_recording = False
            self.file = None
            print("Stopped recording.")
    
    def save_dataset(self, data):
        """Save dataset if recording is active."""
        if self.is_recording and self.file is not None:
            self.file.create_dataset(str(self.dataset_count), data=data)
            self.dataset_count += 1


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================
def detect_marker(channels, times, i):
    """Detect marker following a sync pulse.
    
    Returns:
        tuple: (marker_id, next_index) or (None, next_index) if no marker found
    """
    j = i + 1
    while j < len(channels):
        if channels[j] == MARKER_1_ID:
            return 1, j
        elif channels[j] == MARKER_2_ID:
            return 2, j
        elif channels[j] == SYNC_CHANNEL:
            return None, j  # Next sync reached, no marker found
        j += 1
    
    return None, j  # End of data reached


def process_data_stream_vectorized(channels, times, timetrace, data_to_save, 
                                   seconds_count, unix_timestamp):
    """Process the entire data stream using vectorized NumPy operations.
    
    Returns:
        tuple: (bin_count1, bin_count1_m1, bin_count1_m2,
                syncs_processed, markers_missed)
    """
    # Convert to numpy arrays if not already
    channels = np.asarray(channels)
    times = np.asarray(times)
    
    # Find all sync pulses
    sync_mask = (channels == SYNC_CHANNEL)
    sync_indices = np.where(sync_mask)[0]
    syncs_processed = len(sync_indices)
    
    if syncs_processed == 0:
        return (0, 0, 0, 0, 0)
    
    # Find all markers
    marker1_mask = (channels == MARKER_1_ID)
    marker2_mask = (channels == MARKER_2_ID)
    marker1_indices = np.where(marker1_mask)[0]
    marker2_indices = np.where(marker2_mask)[0]
    
    # Find all photons
    photon_mask = (channels == PHOTON_CHANNEL_1)
    photon_indices = np.where(photon_mask)[0]
    photon_times = times[photon_mask]
    
    if len(photon_indices) == 0:
        return (0, 0, 0, syncs_processed, 0)
    
    # For each photon, find which sync it belongs to
    sync_assignment = np.searchsorted(sync_indices, photon_indices, side='right') - 1
    
    # Remove photons that come before the first sync
    valid_photons = sync_assignment >= 0
    photon_indices = photon_indices[valid_photons]
    photon_times = photon_times[valid_photons]
    sync_assignment = sync_assignment[valid_photons]
    
    if len(photon_indices) == 0:
        return (0, 0, 0, syncs_processed, 0)
    
    # Get the sync time for each photon
    photon_sync_indices = sync_indices[sync_assignment]
    photon_sync_times = times[photon_sync_indices]
    
    # Calculate relative times
    relative_times = photon_times - photon_sync_times
    
    # Filter photons within time window
    in_window = (relative_times >= WINDOW_START) & (relative_times <= (WINDOW_START + WINDOW_LENGTH))
    
    if not np.any(in_window):
        return (0, 0, 0, syncs_processed, 0)
    
    # Filter arrays
    photon_indices_in_window = photon_indices[in_window]
    relative_times_in_window = relative_times[in_window]
    sync_assignment_in_window = sync_assignment[in_window]
    photon_sync_indices_in_window = photon_sync_indices[in_window]
    
    # Calculate bin indices
    bin_indices = ((relative_times_in_window - WINDOW_START) // BIN_SIZE).astype(int)
    
    # Clip bin indices to valid range
    bin_indices = np.clip(bin_indices, 0, len(timetrace.y1) - 1)
    
    # Assign markers to photons if enabled
    if DETECT_MARKERS:
        # For each sync, find the next marker
        markers_for_syncs = np.zeros(syncs_processed, dtype=int)
        
        for idx, sync_idx in enumerate(sync_indices):
            # Find markers that come after this sync
            m1_after = marker1_indices[marker1_indices > sync_idx]
            m2_after = marker2_indices[marker2_indices > sync_idx]
            
            # Find next sync
            if idx < syncs_processed - 1:
                next_sync = sync_indices[idx + 1]
            else:
                next_sync = len(channels)
            
            # Check which marker comes first and before next sync
            m1_first = m1_after[0] if len(m1_after) > 0 and m1_after[0] < next_sync else None
            m2_first = m2_after[0] if len(m2_after) > 0 and m2_after[0] < next_sync else None
            
            if m1_first is not None and (m2_first is None or m1_first < m2_first):
                markers_for_syncs[idx] = 1
            elif m2_first is not None:
                markers_for_syncs[idx] = 2
        
        # Assign markers to photons based on their sync
        photon_markers = markers_for_syncs[sync_assignment_in_window]
        markers_missed = np.sum(markers_for_syncs == 0)
    else:
        photon_markers = np.full(len(bin_indices), -1)
        markers_missed = 0
    
    # Update histograms using np.add.at (fast in-place accumulation)
    np.add.at(timetrace.y1, bin_indices, 1)
    
    marker1_photons = photon_markers == 1
    marker2_photons = photon_markers == 2
    
    np.add.at(timetrace.y1_m1, bin_indices[marker1_photons], 1)
    np.add.at(timetrace.y1_m2, bin_indices[marker2_photons], 1)
    
    # Count totals
    bin_count1 = len(bin_indices)
    bin_count1_m1 = np.sum(marker1_photons)
    bin_count1_m2 = np.sum(marker2_photons)
    
    # Save data if needed (NOW WITH UNIX TIMESTAMP)
    if SAVE_DATA:
        sync_numbers = sync_assignment_in_window + 1  # 1-indexed
        for i in range(len(bin_indices)):
            data_to_save.append([seconds_count, sync_numbers[i], 
                               PHOTON_CHANNEL_1, photon_markers[i], 
                               bin_indices[i], unix_timestamp])
    
    return (bin_count1, bin_count1_m1, bin_count1_m2,
           syncs_processed, markers_missed)


# ============================================================================
# PLOTTING FUNCTIONS (OPTIMIZED)
# ============================================================================
class PlotManager:
    """Manages plot updates efficiently using line objects instead of clearing."""
    
    def __init__(self, ax1, ax2, ax4, ax5):
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax4 = ax4
        self.ax5 = ax5
        
        # Create line objects once
        self.line_m1, = ax1.plot([], [], color='tab:blue', label='Marker 1')
        self.line_m2, = ax1.plot([], [], color='tab:orange', label='Marker 2')  # Changed to orange
        ax1.legend(loc=0)
        ax1.set_xlabel('Bin number')
        ax1.set_ylabel('# of Coincidences')
        
        self.line_total, = ax2.plot([], [], label='Ch1')
        ax2.legend(loc=0)
        ax2.set_ylabel('Counts')
        ax2.set_xlabel('Time (ns)')
        
        self.line_rate, = ax4.plot([], [], alpha=0.7, label='Total')
        self.line_rate_m1, = ax4.plot([], [], label='Marker 1')
        self.line_rate_m2, = ax4.plot([], [], label='Marker 2')
        ax4.grid(ls='--', alpha=0.5)
        ax4.legend(loc=0)
        ax4.set_xlabel('seconds')
        ax4.set_ylabel('Avg count rate')
        
        self.line_counts, = ax5.plot([], [], color='grey', alpha=0.5)
        
        # Initialize axis limits
        self.ax1_xlim = [0, 1]
        self.ax1_ylim = [0, 1]
        self.ax2_xlim = [0, 1]
        self.ax2_ylim = [0, 1]
        self.ax4_xlim = [0, 100]
        self.ax4_ylim = [0, 1]
        self.ax5_ylim = [0, 1]
        self.first_update = True  # Flag to force initial scaling
    
    def update_marker_plot(self, timetrace, sync_count, ch1_count):
        """Update marker plot using set_data (fast)."""
        x_avg, y1_avg, y1_m1_avg, y1_m2_avg = timetrace.get_decimated()
        
        self.line_m1.set_data(x_avg, y1_m1_avg)
        self.line_m2.set_data(x_avg, y1_m2_avg)
        
        # Update title
        self.ax1.set_title(f'Sync: {sync_count}, Ch1: {ch1_count}')
        
        # Auto-scale if needed
        if len(x_avg) > 0:
            x_max = np.max(x_avg)
            y_max = max(np.max(y1_m1_avg), np.max(y1_m2_avg), 1)
            
            if x_max > self.ax1_xlim[1] or y_max > self.ax1_ylim[1] * 0.9:
                self.ax1_xlim = [0, x_max * 1.1]
                self.ax1_ylim = [0, y_max * 1.1]
                self.ax1.set_xlim(self.ax1_xlim)
                self.ax1.set_ylim(self.ax1_ylim)
    
    def update_total_plot(self, timetrace, means, bin_count1, 
                         running_count_rate_m1, running_count_rate_m2):
        """Update total counts plot with SNR calculation."""
        x_avg, y1_avg, _, _ = timetrace.get_decimated()
        
        self.line_total.set_data(x_avg, y1_avg)
        
        # Calculate SNR (marker2 / marker1)
        if len(running_count_rate_m1) > 0 and len(running_count_rate_m2) > 0:
            if running_count_rate_m1[-1] > 0:
                snr = running_count_rate_m2[-1] / running_count_rate_m1[-1]
                title = (f"Ch1: {bin_count1} ({means['ch1']:.2f}/s), "
                        f"SNR [{RUNNING_MEAN_SIZE}]: {snr:.2f}")
            else:
                title = f"Ch1: {bin_count1} ({means['ch1']:.2f}/s), SNR: N/A"
        else:
            title = f"Ch1: {bin_count1} ({means['ch1']:.2f}/s)"
        
        self.ax2.set_title(title)
        
        # Auto-scale if needed
        if len(x_avg) > 0:
            x_max = np.max(x_avg)
            y_max = max(np.max(y1_avg), 1)
            
            if x_max > self.ax2_xlim[1] or y_max > self.ax2_ylim[1] * 0.9:
                self.ax2_xlim = [0, x_max * 1.1]
                self.ax2_ylim = [0, y_max * 1.1]
                self.ax2.set_xlim(self.ax2_xlim)
                self.ax2.set_ylim(self.ax2_ylim)
    
    def update_running_plot(self, running_count_rate, running_count_rate_m1, 
                          running_count_rate_m2, total_counts_plot):
        """Update running count rate plot using set_data (fast)."""
        x_data = np.arange(len(running_count_rate))
        
        self.line_rate.set_data(x_data, running_count_rate)
        self.line_rate_m1.set_data(x_data, running_count_rate_m1)
        self.line_rate_m2.set_data(x_data, running_count_rate_m2)
        
        x_counts = np.arange(len(total_counts_plot))
        self.line_counts.set_data(x_counts, total_counts_plot)
        
        # Auto-scale if needed
        if len(x_data) > 0:
            x_max = max(len(x_data), len(x_counts), 1)
            y_max_main = max(
                np.max(running_count_rate) if len(running_count_rate) > 0 else 1,
                np.max(running_count_rate_m1) if len(running_count_rate_m1) > 0 else 1,
                np.max(running_count_rate_m2) if len(running_count_rate_m2) > 0 else 1,
                1
            )
            y_max_secondary = max(np.max(total_counts_plot) if len(total_counts_plot) > 0 else 1, 1)
            
            # Force initial scaling on first update
            if self.first_update:
                self.ax4_xlim = [0, max(x_max * 1.1, 100)]
                self.ax4_ylim = [0, y_max_main * 1.1]
                self.ax5_ylim = [0, y_max_secondary * 1.1]
                self.ax4.set_xlim(self.ax4_xlim)
                self.ax4.set_ylim(self.ax4_ylim)
                self.ax5.set_xlim(self.ax4_xlim)
                self.ax5.set_ylim(self.ax5_ylim)
                self.first_update = False
            else:
                # Normal auto-scaling logic
                if x_max > self.ax4_xlim[1] * 0.9:
                    self.ax4_xlim = [0, x_max * 1.1]
                    self.ax4.set_xlim(self.ax4_xlim)
                    self.ax5.set_xlim(self.ax4_xlim)
                
                if y_max_main > self.ax4_ylim[1] * 0.9 or y_max_main < self.ax4_ylim[1] * 0.3:
                    self.ax4_ylim = [0, y_max_main * 1.1]
                    self.ax4.set_ylim(self.ax4_ylim)
                
                if y_max_secondary > self.ax5_ylim[1] * 0.9 or y_max_secondary < self.ax5_ylim[1] * 0.3:
                    self.ax5_ylim = [0, y_max_secondary * 1.1]
                    self.ax5.set_ylim(self.ax5_ylim)


# ============================================================================
# FILE I/O FUNCTIONS
# ============================================================================
def setup_data_directory():
    """Create and enter data directory inside photon_counting_data.
    
    Returns:
        tuple: (absolute path to parent photon_counting_data dir, 
                absolute path to timestamped data dir)
    """
    # Get absolute path to photon_counting_data at original cwd
    original_cwd = Path.cwd()
    parent_dir = original_cwd / "photon_counting_data"
    parent_dir.mkdir(exist_ok=True)
    
    # Create the timestamped data directory inside photon_counting_data
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
        print('Clearing...')
        timetrace.clear()
        reset_flag[0] = True
    
    ax_clear = fig.add_axes([0.8, 0.05, 0.08, 0.03])
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
            print("Please enter a filename before starting recording.")
            return
        recording_state.start(filename)
    
    def stop_callback(event):
        recording_state.stop()
    
    btn_start.on_clicked(start_callback)
    btn_stop.on_clicked(stop_callback)
    
    return fig, plot_manager


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
    
    # Disable verbose logging
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
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Num channels: {num_chans}')
    
    return sn


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
        num_bins = 2 + (WINDOW_LENGTH // BIN_SIZE)
        timetrace = TimeTraceData(num_bins)
        count_buffer = CountRateBuffer(RUNNING_MEAN_SIZE)
        recording_state = RecordingState()
        
        # Running statistics
        running_count_rate = []
        running_count_rate_m1 = []
        running_count_rate_m2 = []
        lock_log = []  # Renamed from total_counts_log for compatibility
        total_counts_plot = []
        
        # Data saving
        data_to_save = []
        dataset_count = 0
        seconds_count = 0
        
        # GUI (now returns plot_manager instead of individual axes)
        reset_flag = [False]
        fig, plot_manager = setup_gui(timetrace, reset_flag, recording_state)
        
        # Start first measurement
        sn.unfold.measure(1000, waitFinished=False, savePTU=False)
        
        print("\nAcquisition started. Press Ctrl+C to stop.\n")
        
        # Main acquisition loop
        while True:
            # Get unix timestamp for this measurement cycle
            unix_timestamp = datetime.now().timestamp()
            
            # Wait for measurement to complete
            while not sn.unfold.isFinished():
                continue
            
            # Get data
            sync = sn.unfold.getTimesByChannel(0)
            ch1 = sn.unfold.getTimesByChannel(1)
            times, channels = sn.unfold.getData()
            
            # Start next measurement immediately
            sn.unfold.measure(1000, waitFinished=False, savePTU=False)
            
            # Process data (VECTORIZED VERSION WITH TIMESTAMP)
            start_time = time.time()
            counts = process_data_stream_vectorized(channels, times, timetrace, 
                                                   data_to_save, seconds_count,
                                                   unix_timestamp)
            bin_count1, bin_count1_m1, bin_count1_m2, \
            syncs_processed, markers_missed = counts
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update statistics
            count_buffer.update(bin_count1, bin_count1_m1, bin_count1_m2)
            means = count_buffer.get_means()
            
            # Update running plots
            if reset_flag[0]:
                print('Resetting running counts...')
                running_count_rate = []
                running_count_rate_m1 = []
                running_count_rate_m2 = []
                total_counts_plot = []
                reset_flag[0] = False
            
            # Append to running counts (Total/2 to match old behavior)
            running_count_rate.append(means['ch1'] / 2)
            running_count_rate_m1.append(means['ch1_m1'])
            running_count_rate_m2.append(means['ch1_m2'])
            
            total_counts = len(ch1)
            lock_log.append(total_counts)  # Save as lock_log for compatibility
            total_counts_plot.append(total_counts)
            
            # Update plots (OPTIMIZED - uses line updates, now with SNR)
            plot_manager.update_marker_plot(timetrace, len(sync), len(ch1))
            plot_manager.update_total_plot(timetrace, means, bin_count1,
                                          running_count_rate_m1, running_count_rate_m2)
            plot_manager.update_running_plot(running_count_rate, 
                                            running_count_rate_m1, 
                                            running_count_rate_m2,
                                            total_counts_plot)
            
            # Print statistics
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Markers missed: {markers_missed}")
            print(f"[{timestamp}] Processing time: {processing_time:.1f}ms")
            print()
            
            plt.pause(0.0005)
            
            # Save data periodically
            if SAVE_DATA and seconds_count >= SAVE_SECONDS_PER_DATASET - 1:
                print('Saving data...')
                
                # Save to recording file if active
                recording_state.save_dataset(data_to_save)
                
                # Save to main file (using lock_log for compatibility)
                save_main_data(data_to_save, lock_log, dataset_count)
                
                # Reset for next dataset
                data_to_save = []
                dataset_count += 1
                seconds_count = -1
            
            seconds_count += 1

    except KeyboardInterrupt:
        print("\n\nStopping acquisition (Ctrl+C pressed)...")
    
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