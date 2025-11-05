import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

class ConfigDialog(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MZ Configuration")
        self.result = None
        
        # Default values
        self.default_ip = "10.21.217.191"
        self.default_device = "MFLI"
        self.default_config = "./config/mach_zehnder/"
        self.default_interval = "0.1"
        
        self._create_widgets()
        self._center_window()
    
    def _create_widgets(self):
        # Add dummy mode checkbox before IP address
        self.dummy_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text="Dummy Mode", 
                       variable=self.dummy_var).grid(row=0, column=0, columnspan=2, pady=5)
        
        # IP Address
        ttk.Label(self, text="IP Address:").grid(row=1, column=0, padx=5, pady=5)
        self.ip_entry = ttk.Entry(self)
        self.ip_entry.insert(0, self.default_ip)
        self.ip_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Device Type
        ttk.Label(self, text="Device Type:").grid(row=2, column=0, padx=5, pady=5)
        self.device_combo = ttk.Combobox(self, values=["MFLI", "HF2LI"])
        self.device_combo.set(self.default_device)
        self.device_combo.grid(row=2, column=1, padx=5, pady=5)
        
        # Config File
        ttk.Label(self, text="Config File:").grid(row=3, column=0, padx=5, pady=5)
        self.config_entry = ttk.Entry(self)
        self.config_entry.insert(0, self.default_config)
        self.config_entry.grid(row=3, column=1, padx=5, pady=5, sticky='ew')
        ttk.Button(self, text="Browse", command=self._browse_config).grid(
            row=3, column=2, padx=5, pady=5)
        
        # Lock Check Interval
        ttk.Label(self, text="Check Interval (s):").grid(row=4, column=0, padx=5, pady=5)
        self.interval_entry = ttk.Entry(self)
        self.interval_entry.insert(0, self.default_interval)
        self.interval_entry.grid(row=4, column=1, padx=5, pady=5)
        
        # OK/Cancel buttons
        ttk.Button(self, text="OK", command=self._on_ok).grid(
            row=5, column=0, padx=5, pady=20)
        ttk.Button(self, text="Cancel", command=self._on_cancel).grid(
            row=5, column=1, padx=5, pady=20)
    
    def _browse_config(self):
        filename = filedialog.askopenfilename(
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")])
        if filename:
            self.config_entry.delete(0, tk.END)
            self.config_entry.insert(0, filename)
    
    def _center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'+{x}+{y}')
    
    def _on_ok(self):
        try:
            # Only validate IP and device if not in dummy mode
            if not self.dummy_var.get():
                if not self.ip_entry.get().strip():
                    raise ValueError("IP address cannot be empty")
                if not self.device_combo.get() in ["MFLI", "HF2LI"]:
                    raise ValueError("Invalid device type")
            
            interval = float(self.interval_entry.get())
            if interval <= 0:
                raise ValueError("Interval must be positive")
            
            # Store validated results
            self.result = {
                'dummy_mode': self.dummy_var.get(),
                'ip': self.ip_entry.get().strip(),
                'device_type': self.device_combo.get(),
                'config_path': self.config_entry.get().strip(),
                'interval': interval
            }
            self.destroy()
            
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

    def _on_cancel(self):
        self.destroy()
