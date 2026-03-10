'''
Laser and Cavity Lock System with Adaptive Control

Author: Andrei Militaru with contributions from Github Copilot
Date: 31st January 2026
Updated: Added adaptive behavior with drift detection and oscillation handling

Description:
    Queries an RS oscilloscope to measure the reflected pulsed pump shape.
    Uses this information to lock:
    1. Laser frequency to the optical mode of the electro-optic converter
    2. Cavity used to clear Raman noise
    
    Both locks use adaptive step sizing with drift and oscillation detection.
'''

import pyvisa as visa
import socket
import time
import os
import sys
import subprocess
from datetime import datetime
from collections import deque

import numpy as np
from scipy import interpolate
from interconnector.control.device.piezo_control_client import ControllerClient


# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================

# Lock enable flags
ENABLE_LASER_LOCK = False
ENABLE_CAVITY_LOCK = False
READ_CAVITY_POSITION_DIRECTLY = False  # Now ignored - always use open-loop

# Hardware connection settings
LASER_IP = '10.21.217.189'
LASER_PORT = 1998
OSCILLOSCOPE_ADDRESS = 'TCPIP0::10.21.217.42::inst0::INSTR'

# Timing parameters (seconds)
UPDATE_INTERVAL = 0.25
OSC_RETRY_DELAY = 0.1

# Lock loss detection
CONTRAST_LOSS_THRESHOLD = 0.975

# Cavity recovery parameters
CAVITY_RECOVERY_THRESHOLD = 1.0      # Start recovery ramp above 1V
CAVITY_LOCK_RESUME_THRESHOLD = 0.8   # Resume lock below 300mV
CAVITY_PAUSE_LASER_THRESHOLD = 0.6   # Pause laser lock above 200mV
CAVITY_RAMP_RANGE = 5.0            # Ramp +/- 10V
CAVITY_RAMP_STEP = 0.02              # Step size during ramp
CAVITY_RAMP_DELAY = 0.1            # Delay between ramp steps (seconds)
CAVITY_MIN_VOLTAGE = 0.0            # Minimum allowed voltage
CAVITY_MAX_VOLTAGE = 110.0          # Maximum allowed voltage

# Laser step size parameters (Volts)
LASER_MIN_STEP = 0.0001
LASER_MED_STEP = 0.002
LASER_MAX_STEP = 0.005
LASER_ABSOLUTE_MAX_STEP = 0.01  # 10mV absolute maximum per update

# Laser contrast ranges for step interpolation
LASER_SHORT_RANGE = 0.45  # Close to minimum
LASER_MID_RANGE = 0.65    # Getting closer
LASER_FAR_RANGE = 1.0     # Almost off resonance

# Cavity step size parameters (Volts)
CAVITY_MIN_STEP = 0.0002  # Reduced from 0.0005 for finer control near minimum
CAVITY_MAX_STEP = 0.08
CAVITY_MIN_RANGE = 40e-3   # 50 mV - very close to minimum
CAVITY_MAX_RANGE = 1300e-3  # 1300 mV - far from minimum

# NEW: Gradual step reduction parameters
CAVITY_REDUCTION_THRESHOLD = 50e-3  # 50 mV - below this, reduce steps gradually
CAVITY_STEP_REDUCTION_RATE = 0.9    # Multiply by 0.9 (10% reduction) per iteration
CAVITY_STEP_RESTORE_VALUE = 0.5     # Restore to 50% when going above threshold

# Adaptive behavior parameters
HISTORY_LENGTH = 10
OSCILLATION_THRESHOLD = 4      # Direction changes to detect oscillation
NOISE_THRESHOLD = 0.01         # Relative change considered noise
DRIFT_THRESHOLD = 0.05         # Threshold for drift detection (currently unused)
STEP_REDUCTION_FACTOR = 0.7    # Multiplier when oscillating
STEP_INCREASE_FACTOR = 1.3     # Multiplier when drifting
MAXIMUM_CONFIDENCE = 1         # Max confidence in direction

# Logging
LOG_FOLDER = 'adaptive_laser_lock_logs'


# ==============================================================================
# ADAPTIVE LOCK STATE TRACKER
# ==============================================================================

class LockState:
    """
    Tracks lock behavior history to make intelligent adaptive decisions.
    
    Monitors:
        - Recent measurement history
        - Direction change patterns
        - Performance improvement trends
    
    Provides:
        - Oscillation detection (near optimum)
        - Drift detection (systematic changes)
        - Direction confidence tracking
        - Adaptive step size multipliers
    """
    
    def __init__(self, history_length=HISTORY_LENGTH):
        self.measurement_history = deque(maxlen=history_length)
        self.direction_history = deque(maxlen=history_length)
        self.improvement_history = deque(maxlen=history_length)
        
        self.step_multiplier = 1.0
        self.direction_confidence = 0  # Range: [-MAXIMUM_CONFIDENCE, +MAXIMUM_CONFIDENCE]
        
    def update(self, measurement, step_direction, improved):
        """Record new measurement and update confidence."""
        self.measurement_history.append(measurement)
        self.direction_history.append(step_direction)
        self.improvement_history.append(improved)
        
        # Adjust confidence based on whether we improved
        if improved:
            self.direction_confidence = min(self.direction_confidence + 1, MAXIMUM_CONFIDENCE)
        else:
            self.direction_confidence = max(self.direction_confidence - 1, -MAXIMUM_CONFIDENCE)
    
    def is_oscillating(self):
        """Detect if we're oscillating around the optimum."""
        if len(self.direction_history) < 4:
            return False
        
        direction_changes = sum(
            1 for i in range(1, len(self.direction_history))
            if self.direction_history[i] != self.direction_history[i-1]
        )
        
        return direction_changes >= OSCILLATION_THRESHOLD
    
    def detect_drift(self):
        """
        Detect systematic drift in one direction.
        
        Returns:
            (is_drifting, drift_direction): bool and +1/-1
        """
        if len(self.measurement_history) < 5:
            return False, 0
        
        recent_measurements = list(self.measurement_history)[-5:]
        
        # Count trends
        increasing = sum(
            1 for i in range(1, len(recent_measurements))
            if recent_measurements[i] - recent_measurements[i-1] > NOISE_THRESHOLD
        )
        decreasing = sum(
            1 for i in range(1, len(recent_measurements))
            if recent_measurements[i-1] - recent_measurements[i] > NOISE_THRESHOLD
        )
        
        if increasing >= 3:
            return True, -1  # Measurement increasing, step opposite direction
        elif decreasing >= 3:
            return True, +1  # Measurement decreasing, continue correction
        
        return False, 0
    
    def get_adaptive_step_multiplier(self):
        """
        Calculate step size multiplier based on recent behavior.
        
        Returns:
            (multiplier, status_string): float and str
        """
        is_oscillating = self.is_oscillating()
        is_drifting, _ = self.detect_drift()
        
        if is_oscillating:
            # Near optimum, reduce step size
            self.step_multiplier *= STEP_REDUCTION_FACTOR
            self.step_multiplier = max(self.step_multiplier, 0.3)
            status = "OSC"
        elif is_drifting:
            # Tracking drift, increase step size
            self.step_multiplier *= STEP_INCREASE_FACTOR
            self.step_multiplier = min(self.step_multiplier, 2.0)
            status = "DRIFT"
        else:
            # Normal operation, slowly return to baseline
            if self.step_multiplier > 1.0:
                self.step_multiplier *= 0.95
            else:
                self.step_multiplier *= 1.05
            self.step_multiplier = np.clip(self.step_multiplier, 0.5, 1.5)
            status = "TRACK"
        
        return self.step_multiplier, status
    
    def should_reverse_direction(self, new_measurement, old_measurement):
        """
        Decide whether to reverse step direction.
        
        Considers:
            - Magnitude of change (filter noise)
            - Direction confidence
            - Whether performance worsened
        """
        change = new_measurement - old_measurement
        
        # Ignore noise
        if abs(change) < NOISE_THRESHOLD:
            return False
        
        # If we made it worse
        if change > 0:
            # Reverse if not confident, or reset confidence and reverse
            if self.direction_confidence <= 0:
                return True
            else:
                self.direction_confidence = -1
                return True
        
        return False
    
    def get_status_string(self):
        """Return human-readable status for logging."""
        if len(self.measurement_history) < 2:
            return "INIT"
        
        is_osc = self.is_oscillating()
        is_drift, drift_dir = self.detect_drift()
        
        parts = []
        if is_osc:
            parts.append("OSC")
        if is_drift:
            parts.append(f"DRIFT{'^' if drift_dir > 0 else 'v'}")
        
        parts.append(f"C:{self.direction_confidence:+d}")
        parts.append(f"M:{self.step_multiplier:.2f}")
        
        return " ".join(parts) if parts else "OK"


# ==============================================================================
# HARDWARE COMMUNICATION
# ==============================================================================

def initialize_hardware():
    """
    Connect to all hardware components.
    
    Returns:
        (laser_socket, oscilloscope, cavity_controller)
    """
    # Connect to laser via socket
    laser = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    laser.connect((LASER_IP, LASER_PORT))
    time.sleep(0.3)
    laser.recv(1000)  # Clear welcome message
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Connected to laser')
    
    # Connect to oscilloscope via VISA
    rm = visa.ResourceManager()
    oscilloscope = rm.open_resource(OSCILLOSCOPE_ADDRESS)
    oscilloscope.write("FORMat:DATA ASCii")
    oscilloscope.write("EXPort:WAVeform:INCXvalues OFF")
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Connected to oscilloscope')
    
    # Connect to cavity controller
    cavity = ControllerClient()
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Connected to cavity controller')
    
    return laser, oscilloscope, cavity


def read_laser_offset(laser_socket):
    """
    Read current laser scan offset voltage with full safety validation.
    
    CRITICAL SAFETY: Never returns 0.0 as default - raises exception instead!
    """
    try:
        laser_socket.send(b"(param-ref 'laser1:scan:offset)\n")
        time.sleep(0.1)
        response = laser_socket.recv(100).strip()[:-1].strip()
        
        # Parse value
        try:
            value = float(response)
        except (ValueError, TypeError) as e:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}] ERROR: Cannot parse laser response '{response}': {e}")
            raise RuntimeError(f"Failed to parse laser offset from response: '{response}'") from e
        
        # SAFETY: Validate range before returning
        if not (30.0 <= value <= 90.0):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}] ERROR: Laser offset {value}V is outside safe range [-10, 150]V!")
            raise ValueError(f"Laser offset {value}V out of safe range")
        
        return value
        
    except socket.error as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] ERROR: Socket error reading laser: {e}")
        raise RuntimeError("Socket communication failed") from e

def set_laser_offset(laser_socket, voltage):
    """
    Set laser scan offset voltage with multiple safety checks.
    
    CRITICAL SAFETY: Validates range and logs all changes.
    """
    # SAFETY CHECK 1: Range validation
    if not (30.0 <= voltage <= 90.0):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] ERROR: REFUSING to set dangerous laser voltage {voltage}V!")
        raise ValueError(f"Laser voltage {voltage}V out of safe range [-10, 150]V")
    
    try:
        command = b"(param-set! 'laser1:scan:offset " + str(voltage).encode('utf-8') + b")\n"
        laser_socket.send(command)
        time.sleep(0.1)
        laser_socket.recv(100)
        
        # Voltage set successfully (no logging)
        
    except socket.error as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] ERROR: Failed to set laser voltage: {e}")
        raise RuntimeError("Failed to set laser voltage") from e

def read_oscilloscope_signals(oscilloscope):
    """
    Read all required signals from oscilloscope.
    
    Returns:
        (in_pulse, peak_value, cavity_rms): All in volts
    """
    while True:
        try:
            in_pulse = float(oscilloscope.query("CURS1:Y1p?")) - float(oscilloscope.query("MEAS2:RES:ACT?"))
            peak_value = float(oscilloscope.query("MEAS4:RES:ACT?"))
            cavity_rms = float(oscilloscope.query("MEAS1:RES:ACT?"))
            return in_pulse, peak_value, cavity_rms
        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] OSC read failed, retrying...", end='\r')
            time.sleep(OSC_RETRY_DELAY)


# ==============================================================================
# LOCK CONTROL FUNCTIONS
# ==============================================================================

def update_actuator_position(actuator, state, current_position, measurement, 
                             step_direction, step_interpolator, is_enabled,
                             read_position_func, set_position_func, 
                             read_directly=False, actuator_name="Actuator"):
    """
    Generic function to update any actuator position based on adaptive control.
    
    CRITICAL SAFETY FOR LASER:
    - Validates all steps before applying
    - Clamps to LASER_ABSOLUTE_MAX_STEP
    - Logs all large steps
    """
    if not is_enabled:
        try:
            return True, read_position_func(actuator)
        except Exception:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {actuator_name} read failed (disabled)", end='\r')
            return True, current_position
    
    try:
        if read_directly:
            current_position = read_position_func(actuator)
        
        multiplier, _ = state.get_adaptive_step_multiplier()
        base_step = step_interpolator(measurement)
        adaptive_step = base_step * multiplier
        
        # SAFETY: Clamp step size for laser
        if actuator_name == "Laser":
            if abs(adaptive_step) > LASER_ABSOLUTE_MAX_STEP:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{timestamp}] WARNING: Clamping laser step from {adaptive_step*1000:.2f}mV to {LASER_ABSOLUTE_MAX_STEP*1000:.1f}mV")
                adaptive_step = np.sign(adaptive_step) * LASER_ABSOLUTE_MAX_STEP
        
        new_position = current_position + adaptive_step * step_direction
        
        # SAFETY: For laser, validate final position before setting
        if actuator_name == "Laser":
            if not (30.0 <= new_position <= 90.0):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{timestamp}] ERROR: Calculated laser position {new_position}V is unsafe! Skipping update.")
                return False, current_position
        
        set_position_func(actuator, new_position)
        return True, new_position
        
    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] ERROR: {actuator_name} update failed: {e}")
        return False, current_position


def evaluate_step_direction(state, old_measurement, new_measurement, current_direction):
    """
    Generic function to evaluate and update step direction.
    Works for both laser and cavity.
    
    Args:
        state: LockState object
        old_measurement: Previous measurement value
        new_measurement: New measurement value
        current_direction: Current step direction (+1 or -1)
    
    Returns:
        new_direction: +1 or -1
    """
    improved = (new_measurement < old_measurement)
    state.update(old_measurement, current_direction, improved)
    
    if state.should_reverse_direction(new_measurement, old_measurement):
        return -current_direction
    
    return current_direction

# ==============================================================================
# LOGGING AND DISPLAY
# ==============================================================================

def initialize_log_file():
    """
    Create log file with headers.
    Returns:
        filepath: str
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H-%M-%S')
    
    os.makedirs(LOG_FOLDER, exist_ok=True)
    filepath = os.path.join(LOG_FOLDER, f'{timestamp}_log.txt')
    
    with open(filepath, 'w') as f:
        f.write('Laser and Cavity Lock - Adaptive Control\n')
        f.write('timestamp,in_pulse(V),peak(V),laser_offset(V),cavity_rms(V),cavity_position(V),status\n')
    
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Logging to: {filepath}')
    return filepath

def log_measurement(filepath, in_pulse, peak, laser_offset, cavity_rms, cavity_position, status):
    """Append measurement to log file."""
    timestamp = datetime.now().timestamp()
    with open(filepath, 'a') as f:
        f.write(f'{timestamp},{in_pulse},{peak},{laser_offset},{cavity_rms},{cavity_position},{status}\n')

def display_status(contrast, laser_offset, cavity_rms, new_cavity_rms, cavity_position, laser_status, cavity_status):
    """Print current status to console."""
    if ENABLE_CAVITY_LOCK:
        status_line = (
            f'Contrast:{contrast:.3f} | ' 
            f'Laser:{laser_offset:.4f}V [{laser_status}] | '
            f'Cavity:{new_cavity_rms*1e3:.1f}mV@{cavity_position:.3f}V [{cavity_status}]'
        )
    else:
        status_line = f'Contrast:{contrast:.3f} | Laser:{laser_offset:.4f}V [{laser_status}]'
    print(status_line, end='\r')

# ==============================================================================
# MAIN LOCK LOOP
# ==============================================================================

class LockController:
    """
    Encapsulates lock control logic for easy integration with CLI or GUI.
    
    Provides:
        - Start/stop control
        - Status callbacks
        - Thread-safe operation
        - Parameter updates
        - Automatic cavity recovery
        - Open-loop cavity position tracking
    """
    
    def __init__(self, laser_socket, oscilloscope, cavity_controller, log_filepath,
                 laser_step_interpolator, cavity_step_interpolator,
                 status_callback=None, log_callback=None):
        """
        Initialize lock controller.
        
        Args:
            laser_socket: Connected laser socket
            oscilloscope: Connected oscilloscope
            cavity_controller: Connected cavity controller
            log_filepath: Path to log file
            laser_step_interpolator: Function for laser step size
            cavity_step_interpolator: Function for cavity step size
            status_callback: Optional function(status_dict) called on each iteration
            log_callback: Optional function(message) called for logging
        """        
        self.laser_socket = laser_socket
        self.oscilloscope = oscilloscope
        self.cavity_controller = cavity_controller
        self.log_filepath = log_filepath
        self.laser_step_interpolator = laser_step_interpolator
        self.cavity_step_interpolator = cavity_step_interpolator
        self.status_callback = status_callback or (lambda x: None)
        self.log_callback = log_callback or print
        
        self.is_running = False
        self.should_stop = False
        
        # Lock enable flags (GUI-controllable)
        self.laser_enabled = ENABLE_LASER_LOCK
        self.cavity_enabled = ENABLE_CAVITY_LOCK
        
        # Initialize state - READ INITIAL CAVITY POSITION ONCE
        if self.cavity_enabled:
            try:
                self.cavity_position = cavity_controller.get_z_voltage()
            except Exception as e:
                self.cavity_position = 0.0
        else:
            self.cavity_position = 0.0
        self.laser_state = LockState()
        self.cavity_state = LockState()
        self.laser_direction = -1
        self.cavity_direction = -1
        self.current_contrast = 0.0
        self.current_laser_offset = 0.0
        self.current_cavity_rms = 0.0
        
        # Cavity recovery state
        self.in_cavity_recovery = False
        self.recovery_start_position = None
        
        # NEW: Gradual step reduction state
        self.cavity_step_reduction_multiplier = 1.0  # Starts at 100%
        
    def start(self):
        """Start the lock loop."""
        if self.is_running:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_callback(f"[{timestamp}] Lock already running")
            return
        
        self.should_stop = False
        self.is_running = True
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_callback(f"[{timestamp}] Lock started")
    
    def stop(self):
        """Request lock to stop gracefully."""
        if not self.is_running:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_callback(f"[{timestamp}] Lock not running")
            return
        
        self.should_stop = True
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")    
        self.log_callback(f"[{timestamp}] Stop requested, finishing current iteration...")
    def reset_state(self):
        """Reset adaptive state (useful after lock loss)."""
        self.laser_state = LockState()
        self.cavity_state = LockState()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_callback(f"[{timestamp}] Adaptive state reset")
    
    def set_laser_enabled(self, enabled):
        """Enable or disable laser lock."""
        self.laser_enabled = enabled
        status = "enabled" if enabled else "disabled"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_callback(f"[{timestamp}] Laser lock {status}")
    
    def set_cavity_enabled(self, enabled):
        """Enable or disable cavity lock."""
        self.cavity_enabled = enabled
        status = "enabled" if enabled else "disabled"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_callback(f"[{timestamp}] Cavity lock {status}")
    
    def perform_cavity_recovery(self):
        """
        Perform cavity recovery by ramping voltage to find low RMS region.
        
        Returns:
            bool: True if recovery successful, False otherwise
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_callback(f"[{timestamp}] === CAVITY RECOVERY MODE ACTIVATED ===")
        self.in_cavity_recovery = True
        # Store starting position (current tracked position)
        self.recovery_start_position = self.cavity_position
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Try ramping in positive direction first
        self.log_callback(f"[{timestamp}] Ramping +{CAVITY_RAMP_RANGE}V...")
        if self._ramp_and_check(+1, CAVITY_RAMP_RANGE):
            self.in_cavity_recovery = False
            return True
        self.in_cavity_recovery = False
        # Return to start + 1V before trying negative direction
        try:
            return_position = self.recovery_start_position + 1.0
            # Ensure we don't exceed bounds
            return_position = max(CAVITY_MIN_VOLTAGE, min(return_position, CAVITY_MAX_VOLTAGE))
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_callback(f"[{timestamp}] Returning to {return_position:.3f}V before negative ramp")
            self.cavity_controller.set_z_voltage(return_position)
            self.cavity_position = return_position
            time.sleep(0.2)
        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_callback(f"[{timestamp}] Failed to return to start: {e}")
        # Try ramping in negative direction
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_callback(f"[{timestamp}] Ramping -{CAVITY_RAMP_RANGE}V...")
        if self._ramp_and_check(-1, CAVITY_RAMP_RANGE):
            self.in_cavity_recovery = False
            return True
        # Recovery failed, return to original start position
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_callback(f"[{timestamp}] Recovery failed, returning to original position")
        try:
            self.cavity_controller.set_z_voltage(self.recovery_start_position)
            self.cavity_position = self.recovery_start_position
        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_callback(f"[{timestamp}] Failed to return to start: {e}")
        self.in_cavity_recovery = False
        return False
    def _ramp_and_check(self, direction, total_range):
        """
        Ramp cavity voltage in one direction while checking RMS.
        
        Args:
            direction: +1 or -1
            total_range: Total voltage range to scan
        
        Returns:
            bool: True if low RMS found, False otherwise
        """
        num_steps = int(total_range / CAVITY_RAMP_STEP)
        
        for step in range(num_steps):
            try:
                # Calculate new position (open-loop)
                new_pos = self.cavity_position + direction * CAVITY_RAMP_STEP
                
                # Check voltage bounds
                if new_pos < CAVITY_MIN_VOLTAGE:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.log_callback(f"[{timestamp}] Reached minimum voltage limit ({CAVITY_MIN_VOLTAGE}V)")
                    return False
                if new_pos > CAVITY_MAX_VOLTAGE:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.log_callback(f"[{timestamp}] Reached maximum voltage limit ({CAVITY_MAX_VOLTAGE}V)")
                    return False
                
                # Set new position
                self.cavity_controller.set_z_voltage(new_pos)
                self.cavity_position = new_pos  # Update tracked position
                time.sleep(CAVITY_RAMP_DELAY)
                
                # Check RMS
                try:
                    _, _, cavity_rms = read_oscilloscope_signals(self.oscilloscope)
                    
                    # Send progress update via callback (every 10 steps to avoid spam)
                    if step % 10 == 0:
                        progress_pct = (step / num_steps) * 100
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.log_callback(f"[{timestamp}] Ramping: {progress_pct:.1f}% | Pos: {new_pos:.3f}V | RMS: {cavity_rms*1e3:.1f}mV")
                    
                    # Check if we found good region
                    if cavity_rms < CAVITY_LOCK_RESUME_THRESHOLD:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.log_callback(f"[{timestamp}] Found good region at {new_pos:.3f}V (RMS: {cavity_rms*1e3:.1f}mV)")
                        return True
                
                except Exception as e:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.log_callback(f"[{timestamp}] OSC read error during ramp: {e}")
                    continue
            
            except Exception as e:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.log_callback(f"[{timestamp}] Ramp step failed: {e}")
                return False
        
        return False
    def run_single_iteration(self):
        """
        Execute one iteration of the lock loop.
        Returns True if should continue, False if should stop.
        
        SAFETY: All laser operations wrapped in exception handling.
        """
        if not self.is_running or self.should_stop:
            self.is_running = False
            return False
        
        try:
            # === READ CURRENT STATE ===
            in_pulse, peak, cavity_rms = read_oscilloscope_signals(self.oscilloscope)
            contrast = in_pulse / peak
            
            # SAFETY: Wrap laser read in try-except, skip iteration on failure
            try:
                laser_offset = read_laser_offset(self.laser_socket)
            except (RuntimeError, ValueError, socket.error) as e:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.log_callback(f"[{timestamp}] CRITICAL: Laser read failed: {e}")
                self.log_callback(f"[{timestamp}] Skipping lock iteration for safety")
                time.sleep(UPDATE_INTERVAL)
                return True
            
            self.current_contrast = contrast
            self.current_laser_offset = laser_offset
            self.current_cavity_rms = cavity_rms
            
            # === CHECK FOR CAVITY RECOVERY NEEDED ===
            if self.cavity_enabled and cavity_rms > CAVITY_RECOVERY_THRESHOLD:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.log_callback(f"[{timestamp}] Cavity RMS too high ({cavity_rms*1e3:.1f}mV), starting recovery...")
                
                self.recovery_start_position = self.cavity_position
                recovery_success = self.perform_cavity_recovery()
                
                if recovery_success:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.log_callback(f"[{timestamp}] Cavity recovered, resuming adaptive lock")
                    self.reset_state()
                else:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.log_callback(f"[{timestamp}] WARNING: Cavity recovery failed!")
                    try:
                        self.cavity_controller.set_z_voltage(self.recovery_start_position)
                        self.cavity_position = self.recovery_start_position
                    except Exception as e:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.log_callback(f"[{timestamp}] Failed to return to start: {e}")
                
                time.sleep(UPDATE_INTERVAL)
                return True
            
            # === CHECK IF LASER LOCK SHOULD BE PAUSED ===
            laser_lock_active = self.laser_enabled
            if self.cavity_enabled and cavity_rms > CAVITY_PAUSE_LASER_THRESHOLD:
                laser_lock_active = False
            
            # Get status strings for logging
            _, laser_behavior = self.laser_state.get_adaptive_step_multiplier()
            _, cavity_behavior = self.cavity_state.get_adaptive_step_multiplier()
            
            laser_status_prefix = "" if laser_lock_active else "PAUSED:"
            status = f"L:{laser_status_prefix}{laser_behavior}|C:{cavity_behavior}|{self.laser_state.get_status_string()}"
            
            log_measurement(self.log_filepath, in_pulse, peak, laser_offset, cavity_rms, self.cavity_position, status)
            
            # === CHECK FOR LOCK LOSS ===
            if contrast > CONTRAST_LOSS_THRESHOLD:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.log_callback(f"[{timestamp}] LOCK LOST (contrast={contrast:.3f})")
                self.reset_state()
                time.sleep(UPDATE_INTERVAL)
                
                self.status_callback({
                    'lock_lost': True,
                    'contrast': contrast,
                    'laser_offset': laser_offset,
                    'cavity_rms': cavity_rms,
                    'cavity_position': self.cavity_position
                })
                return True
            
            # === UPDATE POSITIONS ===
            # SAFETY: update_actuator_position has built-in safety checks
            success, new_laser_offset = update_actuator_position(
                self.laser_socket, self.laser_state, laser_offset, contrast, 
                self.laser_direction, self.laser_step_interpolator, laser_lock_active,
                read_laser_offset, set_laser_offset, 
                read_directly=False, actuator_name="Laser"
            )
            if not success:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.log_callback(f"[{timestamp}] Laser update failed, skipping iteration")
                time.sleep(UPDATE_INTERVAL)
                return True
            
            laser_offset = new_laser_offset  # Update tracked value
            
            # CAVITY: Open-loop position tracking with gradual step reduction
            if self.cavity_enabled:
                try:
                    multiplier, _ = self.cavity_state.get_adaptive_step_multiplier()
                    base_step = self.cavity_step_interpolator(cavity_rms)
                    adaptive_step = base_step * multiplier
                    
                    # NEW: Apply gradual step reduction when below threshold
                    if cavity_rms < CAVITY_REDUCTION_THRESHOLD:
                        # Reduce step size by 10% each iteration (can go to zero)
                        self.cavity_step_reduction_multiplier *= CAVITY_STEP_REDUCTION_RATE
                    else:
                        # Above threshold: restore to fixed value immediately
                        self.cavity_step_reduction_multiplier = CAVITY_STEP_RESTORE_VALUE
                    
                    # Apply the reduction multiplier
                    adaptive_step *= self.cavity_step_reduction_multiplier
                    
                    new_cavity_position = self.cavity_position + adaptive_step * self.cavity_direction
                    new_cavity_position = max(CAVITY_MIN_VOLTAGE, min(new_cavity_position, CAVITY_MAX_VOLTAGE))
                    
                    self.cavity_controller.set_z_voltage(new_cavity_position)
                    self.cavity_position = new_cavity_position
                    
                except Exception as e:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.log_callback(f"[{timestamp}] Cavity update failed: {e}")
                    time.sleep(UPDATE_INTERVAL)
                    return True
            
            time.sleep(UPDATE_INTERVAL)
            
            # === READ NEW STATE ===
            in_pulse, new_peak, new_cavity_rms = read_oscilloscope_signals(self.oscilloscope)
            new_contrast = in_pulse / new_peak
            
            # SAFETY: Try to read laser again, but don't fail if it errors
            try:
                current_laser_offset = read_laser_offset(self.laser_socket)
            except Exception:
                current_laser_offset = laser_offset  # Use last known good value
            
            _, laser_behavior = self.laser_state.get_adaptive_step_multiplier()
            _, cavity_behavior = self.cavity_state.get_adaptive_step_multiplier()
            laser_status_prefix = "" if (self.laser_enabled and new_cavity_rms <= CAVITY_PAUSE_LASER_THRESHOLD) else "PAUSED:"
            status = f"L:{laser_status_prefix}{laser_behavior}|C:{cavity_behavior}|{self.laser_state.get_status_string()}"
            
            log_measurement(self.log_filepath, in_pulse, new_peak, 
                           current_laser_offset, new_cavity_rms, self.cavity_position, status)
            
            # === EVALUATE AND UPDATE DIRECTIONS ===
            if laser_lock_active:
                self.laser_direction = evaluate_step_direction(
                    self.laser_state, contrast, new_contrast, self.laser_direction
                )
            
            self.cavity_direction = evaluate_step_direction(
                self.cavity_state, cavity_rms, new_cavity_rms, self.cavity_direction
            )
            
            self.current_contrast = new_contrast
            self.current_cavity_rms = new_cavity_rms
            
            # === NOTIFY GUI ===
            self.status_callback({
                'lock_lost': False,
                'contrast': new_contrast,
                'laser_offset': current_laser_offset,
                'laser_behavior': laser_behavior,
                'laser_confidence': self.laser_state.direction_confidence,
                'laser_multiplier': self.laser_state.step_multiplier,
                'laser_paused': not laser_lock_active,
                'cavity_rms': new_cavity_rms,
                'cavity_position': self.cavity_position,
                'cavity_behavior': cavity_behavior,
                'cavity_confidence': self.cavity_state.direction_confidence,
                'cavity_multiplier': self.cavity_state.step_multiplier,
                'in_recovery': self.in_cavity_recovery,
                'status': status
            })
            
            return True
            
        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_callback(f"[{timestamp}] CRITICAL ERROR in lock iteration: {e}")
            import traceback
            self.log_callback(traceback.format_exc())
            time.sleep(UPDATE_INTERVAL)
            return True

    def run_continuous(self):
        """Run lock continuously (blocking, for CLI usage)."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_callback('='*70)
        self.log_callback(f'[{timestamp}] STARTING ADAPTIVE LOCK')
        self.log_callback('='*70)
        
        self.start()
        
        while self.run_single_iteration():
            pass
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_callback(f"[{timestamp}] Lock stopped")


def run_lock_cli(laser_socket, oscilloscope, cavity_controller, log_filepath,
                 laser_step_interpolator, cavity_step_interpolator):
    """CLI version of lock - prints to console."""
    last_print_time = [0]
    
    def print_status(status_dict):
        """Format and print status to console."""
        if status_dict['lock_lost']:
            print()
            return
        
        if status_dict.get('in_recovery', False):
            return
        
        current_time = time.time()
        if current_time - last_print_time[0] < 1.0:
            return
        last_print_time[0] = current_time
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        parts = [f"[{timestamp}]", f"Contrast:{status_dict['contrast']:.3f}"]
        
        if ENABLE_LASER_LOCK:
            paused = " [PAUSED]" if status_dict.get('laser_paused', False) else ""
            parts.append(f"Laser:{status_dict['laser_offset']:.4f}V [{status_dict['laser_behavior']}]{paused}")
        
        if ENABLE_CAVITY_LOCK:
            parts.append(f"Cavity:{status_dict['cavity_rms']*1e3:.1f}mV@{status_dict['cavity_position']:.3f}V [{status_dict['cavity_behavior']}]")
        
        print(" | ".join(parts))
    
    def print_log(message):
        """Print log messages to console."""
        print(message)
    
    controller = LockController(
        laser_socket, oscilloscope, cavity_controller, log_filepath,
        laser_step_interpolator, cavity_step_interpolator,
        status_callback=print_status,
        log_callback=print_log
    )
    
    controller.run_continuous()


# ==============================================================================
# SERVER MANAGEMENT
# ==============================================================================

def ensure_server_running():
    """Check if server is running, start it if not."""
    HOST = '10.21.217.17'
    PORT = 65433
    
    # Try to connect to see if server is already running
    try:
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_sock.settimeout(0.5)
        test_sock.connect((HOST, PORT))
        test_sock.close()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Piezo control server already running")
        return None  # Server already running, we didn't start it
    except (socket.error, socket.timeout):
        pass  # Server not running, we'll start it
    
    # Start the server as a subprocess
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Starting piezo control server...")
    
    try:
        # Use python -m to run the module
        server_process = subprocess.Popen(
            [sys.executable, '-m', 'interconnector.control.device.piezo_control_server'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
        )
        
        # Wait for server to start (try connecting a few times)
        for attempt in range(10):
            time.sleep(0.5)
            try:
                test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_sock.settimeout(0.5)
                test_sock.connect((HOST, PORT))
                test_sock.close()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Server started successfully")
                return server_process  # Return process so it can be cleaned up
            except (socket.error, socket.timeout):
                if server_process.poll() is not None:
                    # Process terminated unexpectedly
                    raise Exception(f"Server process terminated with code {server_process.returncode}")
                continue
        
        raise Exception("Server failed to start after 5 seconds")
        
    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Failed to start server: {e}")
        if 'server_process' in locals():
            server_process.terminate()
        raise


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == '__main__':
    server_process = None
    
    try:
        # Ensure piezo control server is running
        print('\n' + '='*70)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'[{timestamp}] CHECKING PIEZO CONTROL SERVER')
        print('='*70)
        server_process = ensure_server_running()
        
        # Create step size interpolators
        laser_step_interpolator = interpolate.interp1d(
            [-20, LASER_SHORT_RANGE, LASER_MID_RANGE, LASER_FAR_RANGE, 20],
            [LASER_MIN_STEP, LASER_MIN_STEP, LASER_MED_STEP, LASER_MAX_STEP, LASER_MAX_STEP]
        )
        
        cavity_step_interpolator = interpolate.interp1d(
            [0, CAVITY_MIN_RANGE, CAVITY_MAX_RANGE, 5.0],
            [CAVITY_MIN_STEP, CAVITY_MIN_STEP, CAVITY_MAX_STEP, CAVITY_MAX_STEP],
            bounds_error=False,
            fill_value=(CAVITY_MIN_STEP, CAVITY_MAX_STEP)
        )
        
        # Initialize hardware
        print('\n' + '='*70)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'[{timestamp}] INITIALIZING HARDWARE')
        print('='*70)
        laser_socket, oscilloscope, cavity_controller = initialize_hardware()
        
        # Set up logging
        print('\n' + '='*70)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'[{timestamp}] SETTING UP LOGGING')
        print('='*70)
        log_filepath = initialize_log_file()
        
        # Run the lock (CLI version)
        time.sleep(1)
        run_lock_cli(laser_socket, oscilloscope, cavity_controller, log_filepath,
                     laser_step_interpolator, cavity_step_interpolator)
    
    finally:
        # Clean up server if we started it
        if server_process:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}] Stopping piezo control server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                server_process.kill()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Server stopped")