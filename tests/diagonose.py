#!/usr/bin/env python3
"""
Diagnose wire clustering for medium dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from catenary_detector.io import PointCloudLoader
from catenary_detector.clustering import WireClusterer
from catenary_detector.config import DEFAULT_CONFIG

def main():
    # Load medium dataset
    loader = PointCloudLoader()
    cloud = loader.load('data/lidar_cable_points_medium.parquet')
    points = cloud.points
    
    print(f"Medium dataset: {len(points)} points")
    
    # Create clusterer with debug
    clusterer = WireClusterer(DEFAULT_CONFIG.clustering)
    
    # Manually segment spans
    spans = clusterer.segmenter.detect_spans(points)
    print(f"\nSpans detected: {len(spans)}")
    
    for i, span in enumerate(spans):
        print(f"\nSpan {i}: {len(span)} points")
        print(f"  X range: {span[:, 0].min():.1f} to {span[:, 0].max():.1f}m")
        
        # Analyze elevation
        elevations = span[:, 2]
        hist, bins = np.histogram(elevations, bins=30)
        from scipy import signal
        peaks, _ = signal.find_peaks(hist, height=max(hist)*0.2)
        
        print(f"  Elevation peaks: {len(peaks)}")
        for peak in peaks:
            peak_elev = (bins[peak] + bins[peak+1]) / 2
            print(f"    Peak at {peak_elev:.2f}m")
        
        # Test elevation-based clustering
        print(f"\n  Testing elevation-based clustering:")
        result = clusterer._cluster_by_elevation(span)
        print(f"    Elevation clusters: {result.n_clusters}")
        
        if result.n_clusters > 0:
            # Show the clusters from elevation-based method
            for label in set(result.labels):
                if label == -1:
                    continue
                cluster_mask = result.labels == label
                cluster_points = span[cluster_mask]
                if len(cluster_points) > 0:
                    print(f"    Cluster {label}: {len(cluster_points)} pts, "
                          f"elevation: {cluster_points[:, 2].mean():.2f}m")
        
        # Check wire type
        wire_type = clusterer._detect_wire_type(span)
        print(f"\n  Wire type detected: {wire_type.value}")
        
        if wire_type.value == 'stacked':
            # Now test the actual clustering flow
            print(f"\n  Testing full clustering flow:")
            if wire_type.value == 'stacked':
                # Try elevation-based clustering first (more reliable for wires)
                result = clusterer._cluster_by_elevation(span)
                
                # If elevation clustering fails, fall back to DBSCAN
                if result.n_clusters <= 1:
                    print("    Elevation clustering failed, trying DBSCAN")
                    result = clusterer._cluster_dbscan(span)
            
            print(f"    Method used: {result.method}")
            print(f"    Clusters found: {result.n_clusters}")
            
            # Create wires from result
            if result.method == 'elevation':
                wires = []
                for label in set(result.labels):
                    if label == -1:
                        continue
                    cluster_mask = result.labels == label
                    cluster_points = span[cluster_mask]
                    if len(cluster_points) > 10:
                        wires.append(cluster_points)
                
                print(f"    Wires from elevation: {len(wires)}")
                for j, wire_pts in enumerate(wires):
                    print(f"      Wire {j}: {len(wire_pts)} pts, "
                          f"elevation: {wire_pts[:, 2].mean():.2f}m, "
                          f"length: {wire_pts[:, 0].max()-wire_pts[:, 0].min():.1f}m")
            else:
                # Old method (for DBSCAN)
                wires = clusterer._create_wires_from_result(span, result)
                print(f"    Wires created: {len(wires)}")
                
                # Show wire details
                for j, wire in enumerate(wires):
                    pts = wire.points
                    print(f"      Wire {j}: {len(pts)} pts, "
                          f"elevation: {pts[:, 2].mean():.2f}m, "
                          f"length: {pts[:, 0].max()-pts[:, 0].min():.1f}m")
                
                # Apply outlier removal
                print(f"\n    After outlier removal:")
                filtered_wires = clusterer._remove_outlier_wires(wires)
                for j, wire in enumerate(filtered_wires):
                    pts = wire.points
                    print(f"      Wire {j}: {len(pts)} pts, "
                          f"elevation: {pts[:, 2].mean():.2f}m, "
                          f"length: {pts[:, 0].max()-pts[:, 0].min():.1f}m")

if __name__ == "__main__":
    main()