import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def analyze_easy_dataset():
    """Check if easy dataset has more than 2 wires."""
    df = pd.read_parquet("data/lidar_cable_points_easy.parquet")
    points = df[['x', 'y', 'z']].values
    
    print(f"Easy dataset: {len(points)} points")
    
    # Try VERY sensitive clustering
    for eps in [0.2, 0.3, 0.4, 0.5]:
        clustering = DBSCAN(eps=eps, min_samples=5).fit(points)
        labels = clustering.labels_
        
        # Count meaningful clusters (size > 20)
        unique_labels = set(labels)
        cluster_sizes = []
        
        for label in unique_labels:
            if label == -1:
                continue
            size = np.sum(labels == label)
            if size >= 20:
                cluster_sizes.append((label, size))
        
        print(f"  eps={eps}: {len(cluster_sizes)} clusters with sizes: {[s for _, s in cluster_sizes]}")
        
        if len(cluster_sizes) > 2:
            print(f"    ⚠️  FOUND {len(cluster_sizes)} POTENTIAL WIRES!")
            
            # Quick elevation check
            for label, size in cluster_sizes:
                cluster_points = points[labels == label]
                z_mean = cluster_points[:, 2].mean()
                print(f"      Cluster {label}: {size} pts, mean Z={z_mean:.1f}m")
    
    # Check elevation distribution
    print(f"\nElevation analysis:")
    z = df['z'].values
    plt.hist(z, bins=30, alpha=0.7)
    plt.xlabel('Elevation (Z)')
    plt.ylabel('Count')
    plt.title('Easy Dataset - Elevation Distribution (Look for peaks = wires)')
    plt.show()

# RUN THIS NOW!
analyze_easy_dataset()