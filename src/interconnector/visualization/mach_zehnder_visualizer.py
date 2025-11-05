"""
Mach-Zehnder Visualization
Author: Andrei Militaru
Contributions: GitHub Copilot (code organization and Python patterns)
Organization: Institute of Science and Technology Austria (ISTA)
Date: October 2025
Description: Visualization tools for Mach-Zehnder interferometer calibration
            and lock performance data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
from ..mach_zehnder_utils.phase_calibration import unlock_model, lock_model, evaluate_visibility
from .set_axes import set_ax

class MachZehnderVisualizer:
    def __init__(self, calibration_path: str):
        """Initialize visualizer with path to calibration data."""
        self.calib_path = Path(calibration_path) / "calibrations"
    
    def plot_range_calibration(self, timestamp: Optional[str] = None, ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot range calibration data and fit."""
        # Get the data file
        range_path = self.calib_path / "range"
        if timestamp:
            data_file = range_path / f"data_{timestamp}.npy"
        else:
            files = list(range_path.glob("data_*.npy"))
            if not files:
                raise FileNotFoundError("No range calibration data found")
            data_file = max(files, key=lambda x: x.stat().st_mtime)
        
        # Load data
        data = np.load(str(data_file), allow_pickle=True).item()
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(3.14, 2.5), dpi=150)
        else:
            fig = ax.figure
        
        # Generate and plot fit curve using unlock_model
        x = np.linspace(min(data['edges']), max(data['edges']), 1000)
        # Parameters are [A, x0, x1] for unlock_model
        A, x0, x1 = data['parameters'][:3]
        y = unlock_model(x, A, x0, x1)
        
        ax.plot(x, y, '-', color='C2', linewidth=2, 
                label='Fit', zorder=1)
        
        # Plot data points as circles
        ax.plot(data['edges'][:-1], data['histogram'], 
                'o', color='C0', markeredgecolor='black', 
                markeredgewidth=0.5, alpha=0.7, markersize=4,
                label='Measurement', zorder=2)
        
        # Calculate visibility from parameters
        visibility = evaluate_visibility(data['parameters'])
        
        # Format timestamp for display
        ts = data["timestamp"].split("T")[0]  # Get just the date part
        
        set_ax(ax, xlabel='Voltage (V)', ylabel='Probability Density', legend=True,
               title=f'Range Calibration (Data from {ts})\nVisibility: {visibility:.3f}') 
        
        return fig, ax
    
    def plot_lock_performance(self, timestamp: Optional[str] = None, ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot lock performance data and fit."""
        lock_path = self.calib_path / "lock_precision"
        if timestamp:
            data_file = lock_path / f"data_{timestamp}.npy"
        else:
            files = list(lock_path.glob("data_*.npy"))
            if not files:
                raise FileNotFoundError("No lock performance data found")
            data_file = max(files, key=lambda x: x.stat().st_mtime)
        
        data = np.load(str(data_file), allow_pickle=True).item()
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(3.14, 2.5), dpi=150)
        else:
            fig = ax.figure
        
        # Generate and plot fit curve using lock_model (Gaussian)
        x = np.linspace(min(data['edges']), max(data['edges']), 1000)
        mu, sig2 = data['lock_parameters']  # Parameters are [mu, sigma^2]
        y = lock_model(x, mu, sig2)
        
        ax.plot(x/np.pi, y, '-', color='C2', linewidth=2,
                label='Fit', zorder=1)
        
        # Plot data points as circles
        ax.plot(data['edges'][:-1], data['histogram'], 
                'o', color='C0', markeredgecolor='black',
                markeredgewidth=0.5, alpha=0.7, markersize=4,
                label='Measurement', zorder=2)

        # Format timestamp for display
        ts = data["timestamp"].split("T")[0]  # Get just the date part
        
        set_ax(ax, xlabel=r'Phase (rad / $\pi$)', ylabel='Probability Density',
               title=f'Lock Performance (Data from {ts})', legend=True)
        
        return fig, ax
    
    def plot_combined_analysis(self, 
                             range_timestamp: Optional[str] = None,
                             lock_timestamp: Optional[str] = None) -> Tuple[plt.Figure, list[plt.Axes]]:
        """Create a combined figure with both range calibration and lock performance."""
        fig = plt.figure(figsize=(3.14, 6), dpi=150)
        
        # Create both axes with more space for titles
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        # Plot on the provided axes
        self.plot_range_calibration(timestamp=range_timestamp, ax=ax1)
        self.plot_lock_performance(timestamp=lock_timestamp, ax=ax2)
        
        # Adjust spacing between subplots to accommodate longer titles
        plt.tight_layout(h_pad=2.0)
        
        return fig, [ax1, ax2]