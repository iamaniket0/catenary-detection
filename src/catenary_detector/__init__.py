"""
Catenary Detector
=================

A Python package for detecting wires in LiDAR point clouds and fitting
3D catenary curves to model power transmission lines.

Main Features:
- Automatic wire detection using clustering algorithms
- 3D catenary curve fitting using non-linear optimization
- Support for multiple wire configurations (stacked, parallel)
- Comprehensive visualization tools

Example:
    >>> from catenary_detector import CatenaryDetector
    >>> 
    >>> # Initialize detector
    >>> detector = CatenaryDetector()
    >>> 
    >>> # Run detection and fitting
    >>> results = detector.fit("data/points.parquet")
    >>> 
    >>> # Print summary
    >>> detector.print_summary(results)
    >>> 
    >>> # Visualize results
    >>> detector.plot(results)
    >>> 
    >>> # Save results
    >>> detector.save_results(results, "output.json")

For more control, use individual components:
    >>> from catenary_detector import (
    ...     PointCloudLoader,
    ...     WireClusterer,
    ...     CatenaryFitter,
    ...     CatenaryVisualizer
    ... )
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Main detector class
from catenary_detector.detector import CatenaryDetector

# Configuration
from catenary_detector.config import (
    Config,
    ClusteringConfig,
    FittingConfig,
    VisualizationConfig,
    WireType,
    DEFAULT_CONFIG
)

# I/O
from catenary_detector.io import PointCloud, PointCloudLoader

# Models
from catenary_detector.models import (
    Plane,
    CatenaryParams,
    Catenary2D,
    Catenary3D,
    Wire,
    WireCollection,
    FitMetrics
)

# Clustering
from catenary_detector.clustering import WireClusterer, ClusteringResult

# Fitting
from catenary_detector.fitting import CatenaryFitter, fit_catenary_simple

# Visualization
from catenary_detector.visualization import CatenaryVisualizer, plot_results

__all__ = [
    # Main class
    'CatenaryDetector',
    
    # Configuration
    'Config',
    'ClusteringConfig',
    'FittingConfig',
    'VisualizationConfig',
    'WireType',
    'DEFAULT_CONFIG',
    
    # I/O
    'PointCloud',
    'PointCloudLoader',
    
    # Models
    'Plane',
    'CatenaryParams',
    'Catenary2D',
    'Catenary3D',
    'Wire',
    'WireCollection',
    'FitMetrics',
    
    # Clustering
    'WireClusterer',
    'ClusteringResult',
    
    # Fitting
    'CatenaryFitter',
    'fit_catenary_simple',
    
    # Visualization
    'CatenaryVisualizer',
    'plot_results',
]
