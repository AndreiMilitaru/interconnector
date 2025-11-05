"""
Mach-Zehnder Control GUI
Author: GitHub Copilot (based on requirements by Andrei Militaru)
Date: October 2025
Description: GUI interface for controlling Mach-Zehnder interferometer with
PID control, range calibration, visibility measurement, and lock monitoring.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import yaml  # Add this import
from mach_zehnder_utils.dummy_manager import DummyMZManager
from gui.config_dialog_alt import ConfigDialog
from visualization.mach_zehnder_visualizer import MachZehnderVisualizer 
from control.mach_zehnder_stabilization import MachZehnderManager
import matplotlib.pyplot as plt
import os

# Try to import hardware-dependent modules
HARDWARE_AVAILABLE = False
try:
    from zhinst_utils.demodulation_recorder import zhinst_demod_recorder
    from control.mach_zehnder_stabilization import MachZehnderManager
    HARDWARE_AVAILABLE = True
except ImportError:
    print("Hardware modules not available. Running in dummy mode.")

class ToolTip:
    """Simple tooltip class for tkinter widgets"""
    def __init__(self, widget, text, delay=1000):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tooltip_window = None
        self.timer_id = None
        
        self.widget.bind("<Enter>", self.on_enter)
        self.widget.bind("<Leave>", self.on_leave)
        self.widget.bind("<Motion>", self.on_motion)
    
    def on_enter(self, event=None):
        self.schedule_tooltip()
    
    def on_leave(self, event=None):
        self.cancel_tooltip()
        self.hide_tooltip()
    
    def on_motion(self, event=None):
        self.cancel_tooltip()
        self.hide_tooltip()
        self.schedule_tooltip()
    
    def schedule_tooltip(self):
        self.cancel_tooltip()
        self.timer_id = self.widget.after(self.delay, self.show_tooltip)
    
    def cancel_tooltip(self):
        if self.timer_id:
            self.widget.after_cancel(self.timer_id)
            self.timer_id = None
    
    def show_tooltip(self):
        if self.tooltip_window:
            return
        
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 25
        
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(self.tooltip_window, text=self.text,
                        background="#ffffe0", foreground="black",
                        relief="solid", borderwidth=1,
                        font=("Arial", 11))
        label.pack()
    
    def hide_tooltip(self):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

class MZControlGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mach-Zehnder Phase Control")
        
        # Hide main window initially
        self.withdraw()
        
        # Initialize manager to None first
        self.manager = None
        self.visualizer = None  # Will be initialized in _check_config_and_continue
        
        # Get configuration
        config_dialog = ConfigDialog()
        
        # Wait for the dialog to complete properly
        self._wait_for_config_dialog(config_dialog)
        
    def _wait_for_config_dialog(self, config_dialog):
        # Check if the dialog window still exists
        try:
            if config_dialog.winfo_exists():
                # Dialog is still open, check again in 100ms
                self.after(100, lambda: self._wait_for_config_dialog(config_dialog))
                return
        except tk.TclError:
            # Dialog has been destroyed, proceed
            pass
        
        # Dialog is closed, now check the result
        self._check_config_and_continue(config_dialog)
        
    def _check_config_and_continue(self, config_dialog):
        # Debug: Check what we got from the dialog
        print(f"Dialog result: {getattr(config_dialog, 'result', 'No result attribute')}")
        
        # Check if configuration was successful, if not use dummy mode as fallback
        config = None
        if hasattr(config_dialog, 'result') and config_dialog.result:
            config = config_dialog.result
            print("Configuration successful, proceeding with initialization")
        else:
            print("No valid configuration from dialog, using dummy mode fallback")
            config = {
                'dummy_mode': True,
                'interval': 1.0,  # Default interval
                'ip': '',
                'device_type': '',
                'config_path': ''
            }

        # Get the project root directory (parent of gui directory)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Resolve config path
        if not config.get('config_path'):
            # Use default config path relative to project root
            config['config_path'] = os.path.join(project_root, 'config', 'mach_zehnder', 'default_config.yaml')
        elif not os.path.isabs(config['config_path']):
            # If path is relative, make it relative to project root
            config['config_path'] = os.path.join(project_root, config['config_path'])
        
        # Normalize the path for the current OS
        config['config_path'] = os.path.normpath(config['config_path'])
        
        print(f"Resolved config path: {config['config_path']}")

        # Initialize manager based on mode
        try:
            if not config['dummy_mode'] and HARDWARE_AVAILABLE: 
                # Initialize hardware
                self.mdrec = zhinst_demod_recorder(
                    config['ip'],
                    devtype=config['device_type']
                )
                
                # Initialize real manager
                self.manager = MachZehnderManager(
                    self.mdrec,
                    config_path=config['config_path'],
                    lock_check_interval=config['interval']
                )
            else:
                self.manager = DummyMZManager(
                    lock_check_interval=config['interval']
                )
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize manager: {str(e)}")
            self.destroy()
            return
        
        if self.manager is None:
            messagebox.showerror("Error", "Failed to create manager")
            self.destroy()
            return
        
        # Initialize visualizer with config path
        config_path = config.get('config_path')
        self.visualizer = MachZehnderVisualizer(config_path)

        self._create_widgets()
        self._center_window()
        
        # Show the main window
        self.deiconify()
        self.lift()
        self.focus_force()

    def _create_widgets(self):
        # Range Calibration Frame (now includes visibility)
        range_frame = ttk.LabelFrame(self, text="Range Calibration")
        range_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        
        range_calib_btn = ttk.Button(range_frame, text="Range Calibration", 
                                   command=self._range_calibration)
        range_calib_btn.pack(pady=5)
        ToolTip(range_calib_btn, "Calibrate the voltage range of the Mach-Zehnder interferometer\nby scanning and finding minimum and maximum transmission points")
        
        vis_btn = ttk.Button(range_frame, text="Measure Visibility", 
                           command=self._measure_visibility)
        vis_btn.pack(pady=5)
        ToolTip(vis_btn, "Measure the visibility (fringe contrast) of the interferometer\nHigher visibility indicates better interference quality")
        
        # Range calibration results
        self.range_label = ttk.Label(range_frame, text="No range calibration")
        self.range_label.pack(pady=5)
        ToolTip(self.range_label, "Shows the calibrated voltage range (Vmin - Vmax)\nThese values define the operating range of the interferometer")
        
        self.range_time = ttk.Label(range_frame, text="")
        self.range_time.pack(pady=5)
        
        # Visibility results
        self.vis_label = ttk.Label(range_frame, text="No visibility measurement")
        self.vis_label.pack(pady=5)
        ToolTip(self.vis_label, "Visibility value between 0 and 1\nHigher values indicate better fringe contrast and interferometer quality")
        
        self.vis_time = ttk.Label(range_frame, text="")
        self.vis_time.pack(pady=5)
        
        # PID Configuration Frame
        pid_frame = ttk.LabelFrame(self, text="PID Configuration")
        pid_frame.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")
        
        save_pid_btn = ttk.Button(pid_frame, text="Save PID Config", 
                                command=self.manager.save_current_pid_config)
        save_pid_btn.pack(pady=5)
        ToolTip(save_pid_btn, "Save the current PID controller parameters to file\nThis preserves your tuned settings for future use")
        
        load_pid_btn = ttk.Button(pid_frame, text="Load PID Config", 
                                command=self._load_pid_config)
        load_pid_btn.pack(pady=5)
        ToolTip(load_pid_btn, "Load previously saved PID parameters\nThis will overwrite current controller settings")
        
        # Lock Quality Frame
        lock_frame = ttk.LabelFrame(self, text="Lock Quality")
        lock_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        eval_lock_btn = ttk.Button(lock_frame, text="Evaluate Lock", 
                                 command=self._evaluate_lock)
        eval_lock_btn.pack(pady=5)
        ToolTip(eval_lock_btn, "Evaluate the current lock stability and quality\nLower values indicate more stable phase locking")
        
        self.lock_label = ttk.Label(lock_frame, text="No measurement")
        self.lock_label.pack(pady=5)
        ToolTip(self.lock_label, "Lock quality metric: phase standard deviation")
        
        self.lock_time = ttk.Label(lock_frame, text="") 
        self.lock_time.pack(pady=5)  
        
        # Control Frame
        ctrl_frame = ttk.LabelFrame(self, text="Control")
        ctrl_frame.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")
        
        # Setpoint control
        sp_frame = ttk.Frame(ctrl_frame)
        sp_frame.pack(pady=5)
        
        sp_label = ttk.Label(sp_frame, text="Setpoint:")
        sp_label.pack(side=tk.LEFT)
        ToolTip(sp_label, "Target phase setpoint for the PID controller\nThis is the desired phase value to maintain")
        
        # Safe setpoint initialization
        initial_setpoint = getattr(self.manager, 'setpoint', 0.0)
        self.sp_var = tk.StringVar(value=str(initial_setpoint))
        self.sp_entry = ttk.Entry(sp_frame, textvariable=self.sp_var, width=10)
        self.sp_entry.pack(side=tk.LEFT, padx=5)
        self.sp_entry.bind('<Return>', self._update_setpoint)
        ToolTip(self.sp_entry, "Enter the desired phase setpoint value\nPress Enter to apply the new setpoint")
        
        # Auto setpoint button
        auto_sp_btn = ttk.Button(sp_frame, text="Auto", command=self._auto_setpoint)
        auto_sp_btn.pack(side=tk.LEFT, padx=2)
        ToolTip(auto_sp_btn, "Automatically set setpoint to the middle value\nbetween Vmin and Vmax from range calibration")
        
        # Create a frame for checkboxes to place them side by side
        check_frame = ttk.Frame(ctrl_frame)
        check_frame.pack(pady=5)
        
        # Lock enable control
        self.lock_var = tk.BooleanVar()
        self.lock_check = ttk.Checkbutton(
            check_frame,  # Changed parent to check_frame
            text="Enable Lock",
            variable=self.lock_var,
            command=self._toggle_lock
        )
        self.lock_check.pack(side=tk.LEFT, padx=5)  # Added side and padx
        ToolTip(self.lock_check, "Enable/disable the PID lock\nWhen disabled, the phase drifts freely.")
        
        # Monitoring control
        self.monitor_var = tk.BooleanVar()
        self.monitor_check = ttk.Checkbutton(
            check_frame,  # Changed parent to check_frame
            text="Monitor Locks",
            variable=self.monitor_var,
            command=self._toggle_monitoring
        )
        self.monitor_check.pack(side=tk.LEFT, padx=5)  # Added side and padx
        ToolTip(self.monitor_check, "Enable/disable continuous monitoring of phase locks\nWhen enabled, the system will automatically check and maintain lock stability")
        
        # Add Visualization Frame
        vis_frame = ttk.LabelFrame(self, text="Visualization")
        vis_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        
        # Create button frame for visualization
        vis_btn_frame = ttk.Frame(vis_frame)
        vis_btn_frame.pack(pady=5)
        
        plot_range_btn = ttk.Button(vis_btn_frame, text="Plot Range Calibration",
                                  command=self._plot_range_calibration)
        plot_range_btn.pack(side=tk.LEFT, padx=5)
        ToolTip(plot_range_btn, "Display the latest range calibration data and fit")
        
        plot_lock_btn = ttk.Button(vis_btn_frame, text="Plot Lock Performance",
                                 command=self._plot_lock_performance)
        plot_lock_btn.pack(side=tk.LEFT, padx=5)
        ToolTip(plot_lock_btn, "Display the latest lock performance data and fit")
        
        plot_combined_btn = ttk.Button(vis_btn_frame, text="Plot Combined Analysis",
                                    command=self._plot_combined_analysis)
        plot_combined_btn.pack(side=tk.LEFT, padx=5)
        ToolTip(plot_combined_btn, "Display both calibration and lock performance plots")

        # Auto-load latest results
        self._load_latest_results()
    
    def _load_latest_results(self):
        """Automatically load and display the latest available results"""
        try:
            # Try to get latest range calibration
            if hasattr(self.manager, 'get_latest_range_calibration'):
                range_result = self.manager.get_latest_range_calibration()
                if range_result:
                    # Extract vmin and vmax from par array (indices 1 and 2)
                    if 'par' in range_result and len(range_result['par']) >= 3:
                        vmin = range_result['par'][1]
                        vmax = range_result['par'][2]
                        self.range_label.config(
                            text=f"Range: {vmin:.3f} - {vmax:.3f}V")
                        self.range_time.config(
                            text=f"Calibrated: {self._format_timestamp(range_result['timestamp'])}")
                    else:
                        # Fallback to direct keys if par array not available
                        vmin = range_result.get('vmin', 'N/A')
                        vmax = range_result.get('vmax', 'N/A')
                        self.range_label.config(
                            text=f"Range: {vmin:.3f} - {vmax:.3f}V" if isinstance(vmin, (int, float)) else f"Range: {vmin} - {vmax}V")
                        if 'timestamp' in range_result:
                            self.range_time.config(
                                text=f"Calibrated: {self._format_timestamp(range_result['timestamp'])}")
        except Exception as e:
            print(f"Could not load latest range calibration: {e}")
        
        try:
            # Try to get latest visibility
            if hasattr(self.manager, 'get_latest_visibility'):
                vis_result = self.manager.get_latest_visibility()
                if vis_result:
                    self.vis_label.config(
                        text=f"Visibility: {vis_result['visibility']:.3f}")
                    self.vis_time.config(
                        text=f"Measured: {self._format_timestamp(vis_result['timestamp'])}")
        except Exception as e:
            print(f"Could not load latest visibility: {e}")
        
        try:
            # Try to get latest lock quality
            if hasattr(self.manager, 'latest_lock_quality') and self.manager.latest_lock_quality is not None:
                if hasattr(self.manager, 'get_latest_lock_evaluation'):
                    lock_result = self.manager.get_latest_lock_evaluation()
                    if lock_result:
                        self.lock_label.config(
                            text=f"Lock Quality: {self.manager.latest_lock_quality:.3f}")
                        self.lock_time.config(
                            text=f"Measured: {self._format_timestamp(lock_result['timestamp'])}")
        except Exception as e:
            print(f"Could not load latest lock quality: {e}")
    
    def _load_pid_config(self):
        """Load PID config with confirmation dialog"""
        if messagebox.askyesno("Confirm", "Load the latest PID configuration? This will overwrite current settings."):
            self.manager.load_latest_pid_config()
    
    def _range_calibration(self):
        if messagebox.askyesno("Confirm", "Run range calibration? This will temporarily disable the locks\nand drive the piezo."):
            result = self.manager.perform_range_calibration()
            if result:
                # Extract vmin and vmax from par array (indices 1 and 2)
                if 'par' in result and len(result['par']) >= 3:
                    vmin = result['par'][1]
                    vmax = result['par'][2]
                    self.range_label.config(
                        text=f"Range: {vmin:.3f} - {vmax:.3f}V")
                    self.range_time.config(
                        text=f"Calibrated: {self._format_timestamp(result['timestamp'])}")
                else:
                    # Fallback to direct keys if par array not available
                    vmin = result.get('vmin', 'N/A')
                    vmax = result.get('vmax', 'N/A')
                    self.range_label.config(
                        text=f"Range: {vmin:.3f} - {vmax:.3f}V" if isinstance(vmin, (int, float)) else f"Range: {vmin} - {vmax}V")
                    if 'timestamp' in result:
                        self.range_time.config(
                            text=f"Calibrated: {self._format_timestamp(result['timestamp'])}")

    def _format_timestamp(self, timestamp_str: str) -> str:
        """Convert ISO timestamp to readable format"""
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%d.%m.%Y at %H:%M:%S")

    def _measure_visibility(self):
        result = self.manager.perform_visibility_calibration()
        self.vis_label.config(
            text=f"Visibility: {result['visibility']:.3f}")
        self.vis_time.config(
            text=f"Measured: {self._format_timestamp(result['timestamp'])}")
    
    def _evaluate_lock(self):
        result = self.manager.evaluate_current_lock()
        self.lock_label.config(
            text=f"Lock Quality: {self.manager.latest_lock_quality:.3f}")
        self.lock_time.config(
            text=f"Measured: {self._format_timestamp(result['timestamp'])}")  
    
    def _update_setpoint(self, event=None):
        try:
            value = float(self.sp_var.get())
            self.manager.setpoint = value
        except ValueError:
            self.sp_var.set(str(self.manager.setpoint))
    
    def _toggle_lock(self):
        """Toggle the PID lock on/off"""
        try:
            if self.lock_var.get():
                self.manager.toggle_locks(True)
            else:
                self.manager.toggle_locks(False)
        except AttributeError:
            messagebox.showwarning("Feature Unavailable", "Lock control not available with current manager.")
            self.lock_var.set(False)
    
    def _toggle_monitoring(self):
        if self.monitor_var.get():
            self.manager.start_monitoring()
        else:
            self.manager.stop_monitoring()
    
    def _center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'+{x}+{y}')
    
    def _auto_setpoint(self):
        """Automatically set setpoint to middle of calibrated range"""
        try:
            if hasattr(self.manager, 'get_latest_range_calibration'):
                range_result = self.manager.get_latest_range_calibration()
                if range_result:
                    # Extract vmin and vmax from par array (indices 1 and 2)
                    if 'par' in range_result and len(range_result['par']) >= 3:
                        vmin = range_result['par'][1]
                        vmax = range_result['par'][2]
                        middle_value = (vmin + vmax) / 2.0
                        self.sp_var.set(f"{middle_value:.3f}")
                        self.manager.setpoint = middle_value
                        print(f"Auto-set setpoint to middle value: {middle_value:.3f}V")
                    elif 'vmin' in range_result and 'vmax' in range_result:
                        # Fallback to direct keys
                        vmin = range_result['vmin']
                        vmax = range_result['vmax']
                        middle_value = (vmin + vmax) / 2.0
                        self.sp_var.set(f"{middle_value:.3f}")
                        self.manager.setpoint = middle_value
                        print(f"Auto-set setpoint to middle value: {middle_value:.3f}V")
                    else:
                        messagebox.showwarning("Invalid Data", "Range calibration data format not recognized.")
                else:
                    messagebox.showwarning("No Range Data", "No range calibration data available.\nPlease run range calibration first.")
            else:
                messagebox.showwarning("Feature Unavailable", "Auto setpoint feature not available with current manager.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to auto-set setpoint: {str(e)}")
    
    def _plot_range_calibration(self):
        """Display range calibration plot"""
        try:
            fig, _ = self.visualizer.plot_range_calibration()
            plt.show()
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to plot range calibration: {str(e)}")
    
    def _plot_lock_performance(self):
        """Display lock performance plot"""
        try:
            fig, _ = self.visualizer.plot_lock_performance()
            plt.show()
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to plot lock performance: {str(e)}")
    
    def _plot_combined_analysis(self):
        """Display combined analysis plots"""
        try:
            fig, _ = self.visualizer.plot_combined_analysis()
            plt.show()
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to plot combined analysis: {str(e)}")
            print(f"Debug info - Error details: {str(e)}")  # Added debug info

if __name__ == "__main__":
    try:
        app = MZControlGUI()
        app.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        traceback.print_exc()