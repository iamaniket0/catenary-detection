"""
Configuration classes for the catenary detector package.

This module centralizes all configurable parameters, default values,
and constants used throughout the package.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
from enum import Enum
import numpy as np


class WireType(Enum):
    """Classification of wire arrangements."""
    SINGLE = "single"           # Single wire
    STACKED = "stacked"         # Multiple wires at different heights
    PARALLEL = "parallel"       # Multiple wires at same height, different X positions


@dataclass
class ClusteringConfig:
    """
    Configuration for wire clustering algorithms.
    
    Attributes:
        dbscan_eps: DBSCAN epsilon (neighborhood radius)
        dbscan_min_samples: Minimum points to form a cluster
        dbscan_eps_range: Range of eps values to try for auto-tuning
        kmeans_max_clusters: Maximum clusters for KMeans
        parallel_wire_x_gap: Minimum X gap to consider parallel wires
        stacked_wire_z_range: Minimum Z range to consider stacked wires
        max_noise_percent: Maximum acceptable noise percentage
    """
    dbscan_eps: float = 1.0
    dbscan_min_samples: int = 10
    dbscan_eps_range: List[float] = field(default_factory=lambda: [0.5, 0.8, 1.0, 1.5, 2.0])
    kmeans_max_clusters: int = 4
    kmeans_n_init: int = 10
    kmeans_random_state: int = 42
    parallel_wire_x_gap: float = 5.0
    stacked_wire_z_range: float = 3.0
    max_noise_percent: float = 10.0


@dataclass
class FittingConfig:
    """
    Configuration for catenary curve fitting.
    
    Attributes:
        max_iterations: Maximum iterations for optimizer
        tolerance: Convergence tolerance
        initial_c: Initial guess for curvature parameter
        c_bounds: Min/max bounds for curvature parameter
        use_robust_fitting: Whether to use robust loss function
        robust_loss: Type of robust loss ('linear', 'soft_l1', 'huber', 'cauchy')
        min_points: Minimum points required for fitting
        min_r_squared: Minimum R² to accept a fit
    """
    max_iterations: int = 5000
    tolerance: float = 1e-8
    initial_c: float = 10.0
    c_bounds: Tuple[float, float] = (0.1, 500.0)
    use_robust_fitting: bool = True
    robust_loss: str = "soft_l1"
    min_points: int = 10
    min_r_squared: float = 0.7


@dataclass
class VisualizationConfig:
    """
    Configuration for visualization.
    
    Attributes:
        figure_size: Default figure size (width, height)
        dpi: Resolution for saved figures
        point_size: Size of scatter plot points
        line_width: Width of fitted curve lines
        alpha: Transparency for points
        wire_colors: Color palette for different wires
        cmap: Colormap for height coloring
    """
    figure_size: Tuple[int, int] = (16, 12)
    dpi: int = 150
    point_size: int = 3
    line_width: float = 2.0
    alpha: float = 0.6
    wire_colors: List[str] = field(default_factory=lambda: [
        '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', 
        '#f39c12', '#1abc9c', '#e67e22', '#16a085'
    ])
    cmap: str = 'viridis'


@dataclass
class Config:
    """
    Main configuration container.
    
    Attributes:
        clustering: Clustering configuration
        fitting: Fitting configuration
        visualization: Visualization configuration
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress messages
    """
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    fitting: FittingConfig = field(default_factory=FittingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    random_seed: int = 42
    verbose: bool = True
    
    def __post_init__(self):
        """Set random seed for reproducibility."""
        np.random.seed(self.random_seed)


# Default configuration instance
DEFAULT_CONFIG = Config()