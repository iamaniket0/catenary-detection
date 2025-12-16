#!/usr/bin/env python3

"""
Test plotting functionality.
"""
import matplotlib
matplotlib.use('Agg')  # Must be FIRST, before pyplot

import pytest
import numpy as np
from pathlib import Path
import sys
import os

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from catenary_detector import CatenaryDetector
from catenary_detector.io import PointCloudLoader


@pytest.mark.parametrize("dataset_name", ['easy', 'medium', 'hard', 'extrahard'])
def test_plotting(dataset_name):
    """Test plotting for a specific dataset."""
    print(f"\n{'='*60}")
    print(f"Testing plotting for: {dataset_name}")
    print(f"{'='*60}")
    
    filepath = f'data/lidar_cable_points_{dataset_name}.parquet'
    
    if not Path(filepath).exists():
        pytest.skip(f"File not found: {filepath}")
    
    # Load data
    loader = PointCloudLoader()
    cloud = loader.load(filepath)
    points = cloud.points
    
    print(f"Loaded {len(points)} points")
    
    # Run detector
    detector = CatenaryDetector()
    results = detector.fit(points)
    
    print(f"Detected {results.n_wires} wires")
    
    # Save plot
    os.makedirs('outputs', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red', 'green', 'blue', 'orange']
    
    for i, wire in enumerate(results.wires):
        color = colors[i % len(colors)]
        ax.scatter(wire.points[:, 0], wire.points[:, 2], 
                  s=2, alpha=0.7, color=color, label=f'Wire {i}')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title(f'{dataset_name.upper()} - {results.n_wires} wires')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_path = f'outputs/{dataset_name}_plot.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Plot saved to: {output_path}")
    
    assert results.n_wires >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])