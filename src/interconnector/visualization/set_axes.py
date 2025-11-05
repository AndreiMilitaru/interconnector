"""
Module to record demodulator time traces from Zurich Instruments lock-in amplifiers.
The original version was previously written in the Photonics Laboratory of ETH Zurich
by Andrei Militaru, and it has now been adapted and expanded upon with help from GitHub Copilot.
"""

from typing import Optional, Sequence
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from .plot_styles import PlotStyle
import matplotlib

def set_colorbar(
    mappable,
    ax: plt.Axes,
    style: Optional[PlotStyle] = None,
    ticks: Optional[Sequence[float]] = None,
    ticklabels: Optional[Sequence[str]] = None,
    label: Optional[str] = None,
    label_position: Optional[str] = None,
    colorbar_position: Optional[str] = None,
    **kwargs
) -> Colorbar:
    """Set colorbar style and properties with error handling and version checks."""
    if style is None:
        style = PlotStyle(**kwargs)
    else:
        for key, value in kwargs.items():
            setattr(style, key, value)

    # Check matplotlib version for 'location' argument
    mpl_version = tuple(map(int, matplotlib.__version__.split('.')[:2]))
    location_supported = mpl_version >= (3, 4)

    # Set colorbar position and create it
    position = colorbar_position or style.cbar_position
    try:
        if location_supported:
            clb = plt.colorbar(
                mappable,
                ax=ax,
                orientation=style.cbar_orientation,
                fraction=style.cbar_fraction,
                pad=style.cbar_pad,
                location=position 
            )
        else:
            clb = plt.colorbar(
                mappable,
                ax=ax,
                orientation=style.cbar_orientation,
                fraction=style.cbar_fraction,
                pad=style.cbar_pad
            )
    except Exception as e:
        print(f"Error creating colorbar: {e}")
        return None
    
    # Set label with position adjustment
    if label or style.cbar_label:
        position = label_position or style.cbar_label_position
        try:
            if style.cbar_orientation == 'vertical':
                clb.ax.yaxis.set_label_position(position)
            else:
                clb.ax.xaxis.set_label_position('top')
        except Exception as e:
            print(f"Error setting colorbar label position: {e}")
        clb.set_label(
            label or style.cbar_label,
            fontsize=style.fs,
            labelpad=style.cbar_labelpad
        )

    if ticks is not None:
        clb.set_ticks(ticks)
    elif style.cbar_ticks is not None:
        clb.set_ticks(style.cbar_ticks)
        
    if ticklabels is not None:
        if ticklabels == '':
            # Use set_ticklabels([]) to clear labels safely
            clb.set_ticklabels([])
        else:
            clb.set_ticklabels(ticklabels, fontsize=style.fs_ticks)
    elif style.cbar_ticklabels is not None:
        clb.set_ticklabels(style.cbar_ticklabels, fontsize=style.fs_ticks)
    else:
        clb.ax.tick_params(labelsize=style.fs_ticks)
    
    # Set tick parameters including direction
    clb.ax.tick_params(
        which='both',
        direction=style.tick_direction,
        labelsize=style.fs_ticks
    )
    
    return clb

def set_ax(
    ax: plt.Axes,
    style: Optional[PlotStyle] = None,
    xticks: Optional[Sequence[float]] = None,
    yticks: Optional[Sequence[float]] = None,
    xticklabels: Optional[Sequence[str]] = None,
    yticklabels: Optional[Sequence[str]] = None,
    **kwargs
) -> plt.Axes:
    """Set plot style and axis properties with safer tick label handling."""
    if style is None:
        style = PlotStyle(**kwargs)
    else:
        for key, value in kwargs.items():
            setattr(style, key, value)

    if style.fs_ticks is None:
        style.fs_ticks = style.fs
    if style.fs_title is None:
        style.fs_title = style.fs
    if style.fs_legend is None:
        style.fs_legend = style.fs_ticks - 1
    if style.xlabel is not None:
        ax.set_xlabel(style.xlabel,fontsize=style.fs, color=style.xlabel_color)
    if style.ylabel is not None:
        ax.set_ylabel(style.ylabel,fontsize=style.fs, color=style.ylabel_color)
    if style.title is not None:
        ax.set_title(style.title,fontsize=style.fs_title, color=style.title_color)

    # Handle ticks and labels (now as function parameters)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    
    if xticklabels is not None:
        if xticklabels == '':
            ax.set_xticklabels([])  # Safer way to clear labels
        else:
            ax.set_xticklabels(xticklabels, fontsize=style.fs_ticks)
    else:
        ax.tick_params(axis='x', labelsize=style.fs_ticks, colors=style.xtick_color)
    
    if yticklabels is not None:
        if yticklabels == '':
            ax.set_yticklabels([])  # Safer way to clear labels
        else:
            ax.set_yticklabels(yticklabels, fontsize=style.fs_ticks)
    else:
        ax.tick_params(axis='y', labelsize=style.fs_ticks, colors=style.ytick_color)
    
    if style.legend:
        try:
            ax.legend(fontsize=style.fs_legend)
        except Exception as e:
            print(f"Error setting legend: {e}")
    if style.axis is not None:
        ax.axis(style.axis)
    if style.colorbar:
        try:
            clb = set_colorbar(ax.get_images()[0], ax, style)
        except Exception as e:
            print(f"Error adding colorbar: {e}")
    if style.grid:
        ax.grid()
    ax.tick_params(which='both', direction=style.tick_direction, bottom=style.tick_bottom,
                   top=style.tick_top, left=style.tick_left, right=style.tick_right)
    if style.grid:
        try:
            ax.tick_params(grid_alpha=getattr(style, 'grid_alpha', 0.3))
        except Exception as e:
            print(f"Error setting grid alpha: {e}")
    return ax