"""
Main CatenaryDetector class that orchestrates the full pipeline.

This module provides the high-level API for:
- Loading point cloud data
- Detecting individual wires
- Fitting catenary curves
- Visualizing and exporting results
"""

import numpy as np
from typing import Optional, Union, Dict, Any
from pathlib import Path
import json
import logging

from catenary_detector.config import Config, DEFAULT_CONFIG
from catenary_detector.io import PointCloudLoader, PointCloud
from catenary_detector.clustering import WireClusterer
from catenary_detector.fitting import CatenaryFitter
from catenary_detector.models import Wire, WireCollection
from catenary_detector.visualization import CatenaryVisualizer

logger = logging.getLogger(__name__)


class CatenaryDetector:
    """
    High-level interface for catenary detection and fitting.
    
    This class orchestrates the complete pipeline:
    1. Load point cloud data
    2. Cluster points into individual wires
    3. Fit catenary curves to each wire
    4. Visualize and export results
    
    Example:
        >>> detector = CatenaryDetector()
        >>> results = detector.fit("data/points.parquet")
        >>> print(f"Found {results.n_wires} wires")
        >>> detector.plot(results)
        >>> detector.save_results(results, "output.json")
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize detector.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or DEFAULT_CONFIG
        
        # Initialize components
        self.loader = PointCloudLoader()
        self.clusterer = WireClusterer(self.config.clustering)
        self.fitter = CatenaryFitter(self.config.fitting)
        self.visualizer = CatenaryVisualizer(self.config.visualization)
        
        if self.config.verbose:
            logger.info("CatenaryDetector initialized")
    
    def fit(
        self,
        source: Union[str, Path, np.ndarray, PointCloud],
        cluster: bool = True
    ) -> WireCollection:
        """
        Detect wires and fit catenary curves.
        
        This is the main entry point for the pipeline.
        
        Args:
            source: Point cloud data (file path, array, or PointCloud)
            cluster: Whether to cluster points into wires
            
        Returns:
            WireCollection with fitted catenaries
        """
        # Load data
        if isinstance(source, PointCloud):
            points = source.points
            source_name = source.source or "pointcloud"
        elif isinstance(source, np.ndarray):
            points = source
            source_name = "array"
        else:
            cloud = self.loader.load(source)
            points = cloud.points
            source_name = str(source)
        
        if self.config.verbose:
            logger.info(f"Processing {len(points)} points from {source_name}")
        
        # Cluster into wires
        if cluster:
            collection = self.clusterer.cluster(points)
        else:
            # Treat all points as single wire
            wire = Wire(wire_id=0, points=points)
            collection = WireCollection(wires=[wire])
        
        collection.source = source_name
        
        if self.config.verbose:
            logger.info(f"Detected {collection.n_wires} wires")
        
        # Fit catenaries
        self.fitter.fit_collection(collection)
        
        if self.config.verbose:
            summary = collection.get_summary()
            if 'fit_metrics' in summary:
                logger.info(
                    f"Fitting complete: "
                    f"mean R²={summary['fit_metrics']['mean_r_squared']:.4f}"
                )
        
        return collection
    
    def fit_multiple(
        self,
        sources: list,
        names: Optional[list] = None
    ) -> Dict[str, WireCollection]:
        """
        Process multiple datasets.
        
        Args:
            sources: List of data sources
            names: Optional list of names (auto-generated if None)
            
        Returns:
            Dictionary mapping names to WireCollections
        """
        if names is None:
            names = [f"dataset_{i}" for i in range(len(sources))]
        
        results = {}
        for name, source in zip(names, sources):
            if self.config.verbose:
                logger.info(f"\nProcessing {name}")
            results[name] = self.fit(source)
        
        return results
    
    def plot(
        self,
        collection: WireCollection,
        output_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Visualize results.
        
        Args:
            collection: WireCollection to visualize
            output_path: Optional path to save figure
            show: Whether to display figure
        """
        import matplotlib.pyplot as plt
        
        fig = self.visualizer.plot_results(collection)
        
        if output_path:
            self.visualizer.save(fig, output_path)
        
        if show:
            plt.show()
        
        return fig
    
    def plot_wire(
        self,
        wire: Wire,
        output_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Visualize a single wire in detail.
        
        Args:
            wire: Wire to visualize
            output_path: Optional path to save
            show: Whether to display
        """
        import matplotlib.pyplot as plt
        
        fig = self.visualizer.plot_wire_detail(wire)
        
        if output_path:
            self.visualizer.save(fig, output_path)
        
        if show:
            plt.show()
        
        return fig
    
    def plot_comparison(
        self,
        collections: Dict[str, WireCollection],
        output_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Compare multiple datasets.
        
        Args:
            collections: Dict of name -> WireCollection
            output_path: Optional save path
            show: Whether to display
        """
        import matplotlib.pyplot as plt
        
        fig = self.visualizer.plot_comparison(collections)
        
        if output_path:
            self.visualizer.save(fig, output_path)
        
        if show:
            plt.show()
        
        return fig
    
    def save_results(
        self,
        collection: WireCollection,
        filepath: str,
        include_points: bool = False
    ):
        """
        Save results to JSON file.
        
        Args:
            collection: WireCollection to save
            filepath: Output file path
            include_points: Whether to include point coordinates
        """
        data = collection.to_dict(include_points=include_points)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        if self.config.verbose:
            logger.info(f"Results saved to {filepath}")
    
    def get_summary(self, collection: WireCollection) -> Dict[str, Any]:
        """
        Get summary of detection results.
        
        Args:
            collection: WireCollection
            
        Returns:
            Summary dictionary
        """
        return collection.get_summary()
    
    def print_summary(self, collection: WireCollection):
        """
        Print formatted summary of results.
        
        Args:
            collection: WireCollection
        """
        summary = collection.get_summary()
        
        print("\n" + "="*60)
        print("CATENARY DETECTION RESULTS")
        print("="*60)
        print(f"Source: {summary.get('source', 'N/A')}")
        print(f"Wires detected: {summary['n_wires']}")
        print(f"Total points: {summary['total_points']}")
        
        if 'points_per_wire' in summary:
            ppw = summary['points_per_wire']
            print(f"Points per wire: {ppw['min']}-{ppw['max']} (mean: {ppw['mean']:.0f})")
        
        if 'fit_metrics' in summary:
            fm = summary['fit_metrics']
            print(f"\nFit Quality:")
            print(f"  Fitted: {fm['n_fitted']}/{summary['n_wires']} wires")
            print(f"  Mean R²: {fm['mean_r_squared']:.4f}")
            print(f"  Min R²: {fm['min_r_squared']:.4f}")
            print(f"  Mean RMSE: {fm['mean_rmse']:.4f}m")
        
        print("\nPer-wire details:")
        for wire in collection:
            if wire.is_fitted:
                print(f"  Wire {wire.wire_id}: {wire.n_points} pts, "
                      f"R²={wire.catenary.r_squared:.4f}, "
                      f"RMSE={wire.catenary.rmse:.4f}m")
            else:
                print(f"  Wire {wire.wire_id}: {wire.n_points} pts, not fitted")
        
        print("="*60 + "\n")
