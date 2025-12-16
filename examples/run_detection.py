#!/usr/bin/env python3
"""
Example script for running catenary detection on LiDAR data.

Usage:
    python run_detection.py --input data/points.parquet --output outputs/
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from catenary_detector import CatenaryDetector


def main():
    parser = argparse.ArgumentParser(
        description="Detect wires and fit catenary curves to LiDAR point clouds"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input point cloud file (parquet, csv, or npy)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable visualization"
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Save plot to file"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    print("Initializing CatenaryDetector...")
    detector = CatenaryDetector()
    
    # Run detection
    print(f"\nProcessing: {args.input}")
    results = detector.fit(args.input)
    
    # Print summary
    detector.print_summary(results)
    
    # Save results
    input_name = Path(args.input).stem
    results_file = output_dir / f"{input_name}_results.json"
    detector.save_results(results, str(results_file))
    print(f"Results saved to: {results_file}")
    
    # Visualization
    if not args.no_plot:
        if args.save_plot:
            plot_file = output_dir / f"{input_name}_plot.png"
            detector.plot(results, output_path=str(plot_file), show=True)
            print(f"Plot saved to: {plot_file}")
        else:
            detector.plot(results, show=True)
    
    return results


def run_all_datasets():
    """Run detection on all standard datasets."""
    
    # Find data files
    data_dir = Path("data")
    datasets = {
        'easy': data_dir / "lidar_cable_points_easy.parquet",
        'medium': data_dir / "lidar_cable_points_medium.parquet",
        'hard': data_dir / "lidar_cable_points_hard.parquet",
        'extrahard': data_dir / "lidar_cable_points_extrahard.parquet"
    }
    
    # Filter to existing files
    datasets = {k: v for k, v in datasets.items() if v.exists()}
    
    if not datasets:
        print("No dataset files found in data/ directory")
        return
    
    print(f"Found {len(datasets)} datasets")
    
    # Initialize detector
    detector = CatenaryDetector()
    
    # Process each dataset
    results = {}
    for name, path in datasets.items():
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print('='*60)
        
        collection = detector.fit(str(path))
        results[name] = collection
        detector.print_summary(collection)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, collection in results.items():
        print(f"  {name}: {collection.n_wires} wire(s) detected")
    
    return results


if __name__ == "__main__":
    # Check if running with arguments
    if len(sys.argv) > 1:
        main()
    else:
        # Run interactive demo
        print("No arguments provided. Running demo on all datasets...")
        print("Use --help for command line options.\n")
        run_all_datasets()