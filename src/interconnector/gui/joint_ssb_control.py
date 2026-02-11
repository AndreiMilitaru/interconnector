"""
Joint SSB Control GUI
Author: Andrei Militaru with GitHub Copilot
Organization: Institute of Science and Technology Austria (ISTA)
Date: February 2026

Description: Combined GUI that embeds both Cavity Control and Interferometer
Control in a single window with two tabs.  Both share the same MFLI lock-in
amplifier connection (mdrec) for thread-safe operation.
"""

import sys
import os
import threading
import logging
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from PyQt5.QtGui import QCloseEvent

from interconnector.gui.cavity_control import CavityControlGUI
from interconnector.gui.interferometer_control import InterferometerControlGUI


class JointSSBControlGUI(QMainWindow):
    """Combined GUI with Cavity Control and Interferometer Control as tabs."""

    def __init__(self, cavity_gui, interferometer_gui, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Joint SSB Control")
        self.setGeometry(200, 50, 450, 700)

        self.cavity_gui = cavity_gui
        self.interferometer_gui = interferometer_gui

        # Create tab widget and reparent central widgets from sub-GUIs
        self.tabs = QTabWidget()
        self.tabs.addTab(self.cavity_gui.centralWidget(),
                         "Cavity Control")
        self.tabs.addTab(self.interferometer_gui.centralWidget(),
                         "Interferometer Control")
        self.setCentralWidget(self.tabs)

    def closeEvent(self, event):
        """Forward close to sub-GUIs so they can stop background threads."""
        # Each sub-GUI gets its own event object
        cavity_event = QCloseEvent()
        self.cavity_gui.closeEvent(cavity_event)

        interferometer_event = QCloseEvent()
        self.interferometer_gui.closeEvent(interferometer_event)

        event.accept()


# ======================================================================
# Entry points
# ======================================================================
def main_entry():
    """Standalone entry point -- creates shared mdrec and both GUIs."""
    from interconnector.zhinst_utils.demodulation_recorder import zhinst_demod_recorder
    from interconnector.control.device.function_generator import SingleOutput

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # ----------------------------------------------------------------
    # User-configurable parameters
    # ----------------------------------------------------------------
    verbose = False
    keep_offset_zero = True

    ip_PC = '10.21.217.191'
    ip_fg = '10.21.217.150'
    device_id = 'dev30794'

    # Cavity control
    pid_dither = 1
    dither_drive_demod = 2
    dither_in_demod = 3
    slow_offset = 2  # auxout2 -> Aux3 on device

    locked_reflection_threshold = 0.8
    mid_baseline_threshold = 0.5

    mode_finding_settings = {
        'fg_amplitude_mv': 400.0,
        'fg_amplitude_frequency_hz': 40.0
    }

    # Interferometer control
    piezo_pid = 0
    laser_pid = 3

    # ----------------------------------------------------------------
    # Instruments (shared)
    # ----------------------------------------------------------------
    visa_resource_fg = f"TCPIP0::{ip_fg}::inst0::INSTR"

    mdrec_lock = threading.Lock()
    fg_lock = threading.Lock()

    mdrec = zhinst_demod_recorder(ip_PC, devtype='MFLI')
    fg = SingleOutput(visa_resource_fg)

    # ----------------------------------------------------------------
    # Log files
    # ----------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    cavity_logpath = Path(".", "cavity_logs")
    cavity_logfile = cavity_logpath / f"cavity_control_{timestamp}.log"
    cavity_logfile.parent.mkdir(parents=True, exist_ok=True)

    interferometer_logpath = Path(".", "interferometer_logs")
    interferometer_logfile = (
        interferometer_logpath / f"interferometer_control_{timestamp}.log")
    interferometer_logfile.parent.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Build sub-GUIs (not shown individually)
    # ----------------------------------------------------------------
    cavity_gui = CavityControlGUI(
        mdrec=mdrec, fg=fg, device_id=device_id,
        dither_pid=pid_dither,
        dither_drive_demod=dither_drive_demod,
        dither_in_demod=dither_in_demod,
        verbose=verbose,
        mdrec_lock=mdrec_lock, fg_lock=fg_lock,
        slow_offset=slow_offset,
        keep_offset_zero=keep_offset_zero,
        mid_baseline_threshold=mid_baseline_threshold,
        mode_finding_settings=mode_finding_settings,
        locked_reflection_threshold=locked_reflection_threshold,
        logfile=cavity_logfile)

    interferometer_gui = InterferometerControlGUI(
        mdrec=mdrec, device_id=device_id,
        piezo_pid=piezo_pid, laser_pid=laser_pid,
        mdrec_lock=mdrec_lock, verbose=verbose,
        logfile=interferometer_logfile)

    # ----------------------------------------------------------------
    # Joint window
    # ----------------------------------------------------------------
    window = JointSSBControlGUI(cavity_gui, interferometer_gui)
    window.show()

    try:
        sys.exit(app.exec_())
    except Exception as e:
        logging.error(f"Error running Joint SSB Control GUI: {e}",
                      exc_info=True)


if __name__ == "__main__":
    main_entry()
