from dataclasses import dataclass, asdict
from typing import Optional, Sequence
import yaml

@dataclass
class PlotStyle:
    fs: int = 7
    fs_ticks: Optional[int] = None
    fs_title: Optional[int] = None
    fs_legend: Optional[int] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    title: Optional[str] = None
    legend: bool = False
    axis: Optional[Sequence[float]] = None
    colorbar: bool = False
    grid: bool = True
    grid_alpha: float = 0.3
    tick_direction: str = 'in'
    tick_bottom: bool = True
    tick_top: bool = True
    tick_left: bool = True
    tick_right: bool = True
    ylabel_color: Optional[str] = 'black'
    xlabel_color: Optional[str] = 'black'
    title_color: Optional[str] = 'black'
    xtick_color: Optional[str] = 'black'
    ytick_color: Optional[str] = 'black'

    # Colorbar specific parameters
    cbar_label: Optional[str] = None
    cbar_labelpad: Optional[float] = None
    cbar_ticks: Optional[Sequence[float]] = None
    cbar_ticklabels: Optional[Sequence[str]] = None
    cbar_orientation: str = 'vertical'
    cbar_position: str = 'right'  # 'right'/'left' for vertical, 'top'/'bottom' for horizontal
    cbar_fraction: float = 0.15
    cbar_pad: float = 0.05
    cbar_label_position: str = 'left'  # Add this line

    def save(self, filepath: str):
        """Save style to YAML file"""
        with open(filepath, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def load(cls, filepath: str) -> 'PlotStyle':
        """Load style from YAML file"""
        with open(filepath, 'r') as f:
            return cls(**yaml.safe_load(f))

# Predefined styles
PUBLICATION_STYLE = PlotStyle(
    fs=7,
    grid=True,
    tick_direction='in',
    grid_alpha=0.3
)

PRESENTATION_STYLE = PlotStyle(
    fs=12,
    grid=True,
    tick_direction='in',
    grid_alpha=0.3
)
