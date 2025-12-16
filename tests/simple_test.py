"""
Test script for WireClusterer using Medium LiDAR dataset.
This script loads the point cloud, runs clustering, and prints/plots results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catenary_detector.clustering.wire_clusterer import WireClusterer


# -------------------------------
# Load the Medium dataset
# -------------------------------
file_path = "data/lidar_cable_points_medium.parquet"
df = pd.read_parquet(file_path)

# Assume the point cloud has columns 'x', 'y', 'z'
points = df[['x', 'y', 'z']].to_numpy()

print(f"Loaded {points.shape[0]} points from {file_path}")

# -------------------------------
# Initialize the clusterer
# -------------------------------
clusterer = WireClusterer()

# -------------------------------
# Cluster points
# -------------------------------
collection = clusterer.cluster(points)

print(f"\nTotal wires detected: {collection.n_wires}")

for i, wire in enumerate(collection.wires):
    pts = wire.points
    print(f"Wire {i}: {len(pts)} points, "
          f"X range: {pts[:,0].min():.2f}-{pts[:,0].max():.2f}, "
          f"Z range: {pts[:,2].min():.2f}-{pts[:,2].max():.2f}, "
          f"Mean Z: {pts[:,2].mean():.2f}")

# -------------------------------
# Optional: Plot the wires in 3D
# -------------------------------
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'b', 'g', 'c', 'm', 'y']
for i, wire in enumerate(collection.wires):
    pts = wire.points
    color = colors[i % len(colors)]
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=5, color=color, label=f'Wire {i}')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Medium Dataset Wire Clusters')
ax.legend()
plt.show()
