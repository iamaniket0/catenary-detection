"""
3D Plane model for wire projection.

A plane is used to project 3D wire points into a 2D coordinate system
for catenary fitting.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional
from sklearn.decomposition import PCA


@dataclass
class Plane:
    """
    Represents a plane in 3D space.
    
    The plane is defined by:
    - A point on the plane (typically the centroid of points)
    - A normal vector perpendicular to the plane
    - Two orthonormal basis vectors (u, v) within the plane
    
    Points can be projected onto this plane using the basis vectors,
    converting 3D coordinates to 2D plane coordinates.
    
    Attributes:
        normal: Unit normal vector of the plane (3,)
        point: A point on the plane, typically centroid (3,)
        u: First basis vector in plane (along wire direction) (3,)
        v: Second basis vector in plane (typically vertical component) (3,)
    """
    normal: np.ndarray
    point: np.ndarray
    u: np.ndarray = field(default=None)
    v: np.ndarray = field(default=None)
    
    def __post_init__(self):
        """Normalize vectors and compute basis if needed."""
        # Ensure numpy arrays
        self.normal = np.asarray(self.normal, dtype=np.float64)
        self.point = np.asarray(self.point, dtype=np.float64)
        
        # Normalize normal vector
        norm = np.linalg.norm(self.normal)
        if norm > 0:
            self.normal = self.normal / norm
        
        # Compute basis vectors if not provided
        if self.u is None or self.v is None:
            self._compute_basis()
        else:
            self.u = np.asarray(self.u, dtype=np.float64)
            self.v = np.asarray(self.v, dtype=np.float64)
    
    def _compute_basis(self):
        """
        Compute orthonormal basis vectors u and v within the plane.
        
        u is chosen to be as aligned with the horizontal plane as possible,
        v is perpendicular to both normal and u.
        """
        # Find a reference vector not parallel to normal
        if abs(self.normal[2]) < 0.9:
            # Normal is not vertical, use Z-axis as reference
            ref = np.array([0, 0, 1])
        else:
            # Normal is nearly vertical, use Y-axis
            ref = np.array([0, 1, 0])
        
        # Gram-Schmidt: u = ref - (ref·normal)*normal
        self.u = ref - np.dot(ref, self.normal) * self.normal
        u_norm = np.linalg.norm(self.u)
        
        if u_norm > 1e-10:
            self.u = self.u / u_norm
        else:
            # Fallback if ref was parallel to normal
            self.u = np.array([1, 0, 0])
            self.u = self.u - np.dot(self.u, self.normal) * self.normal
            self.u = self.u / np.linalg.norm(self.u)
        
        # v = normal × u (perpendicular to both)
        self.v = np.cross(self.normal, self.u)
        self.v = self.v / np.linalg.norm(self.v)
    
    def project_point(self, point_3d: np.ndarray) -> np.ndarray:
        """
        Project a single 3D point onto the plane.
        
        Args:
            point_3d: 3D point coordinates (3,)
            
        Returns:
            2D coordinates in plane coordinate system (2,)
        """
        centered = point_3d - self.point
        x = np.dot(centered, self.u)
        y = np.dot(centered, self.v)
        return np.array([x, y])
    
    def project_points(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project multiple 3D points onto the plane.
        
        Args:
            points_3d: Array of shape (N, 3)
            
        Returns:
            Array of shape (N, 2) with plane coordinates
        """
        centered = points_3d - self.point
        x_coords = np.dot(centered, self.u)
        y_coords = np.dot(centered, self.v)
        return np.column_stack([x_coords, y_coords])
    
    def unproject_point(self, point_2d: np.ndarray) -> np.ndarray:
        """
        Convert 2D plane coordinates back to 3D.
        
        Args:
            point_2d: 2D coordinates in plane (2,)
            
        Returns:
            3D point coordinates (3,)
        """
        return self.point + point_2d[0] * self.u + point_2d[1] * self.v
    
    def unproject_points(self, points_2d: np.ndarray) -> np.ndarray:
        """
        Convert multiple 2D plane coordinates back to 3D.
        
        Args:
            points_2d: Array of shape (N, 2)
            
        Returns:
            Array of shape (N, 3)
        """
        return (self.point + 
                np.outer(points_2d[:, 0], self.u) + 
                np.outer(points_2d[:, 1], self.v))
    
    def distance_to_point(self, point_3d: np.ndarray) -> float:
        """
        Compute signed distance from a point to the plane.
        
        Args:
            point_3d: 3D point coordinates (3,)
            
        Returns:
            Signed distance (positive = same side as normal)
        """
        return np.dot(point_3d - self.point, self.normal)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'normal': self.normal.tolist(),
            'point': self.point.tolist(),
            'u': self.u.tolist(),
            'v': self.v.tolist()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Plane':
        """Create Plane from dictionary."""
        return cls(
            normal=np.array(data['normal']),
            point=np.array(data['point']),
            u=np.array(data['u']),
            v=np.array(data['v'])
        )
    
    def __repr__(self) -> str:
        return f"Plane(point={self.point}, normal={self.normal})"


def fit_plane_to_wire(points: np.ndarray) -> Plane:
    """
    Fit a plane to wire points using PCA.
    
    The plane is oriented so that:
    - u-axis is along the wire direction (maximum variance)
    - v-axis captures the catenary sag (includes vertical component)
    - normal is perpendicular to the wire's principal plane
    
    Args:
        points: Array of shape (N, 3) with wire point coordinates
        
    Returns:
        Plane object optimized for catenary fitting
    """
    # Compute centroid
    centroid = np.mean(points, axis=0)
    
    # PCA to find principal directions
    pca = PCA(n_components=3)
    pca.fit(points)
    
    # Wire direction is the first principal component (max variance)
    wire_direction = pca.components_[0]
    
    # Ensure consistent direction (positive Y component)
    if wire_direction[1] < 0:
        wire_direction = -wire_direction
    
    # For catenary fitting, we want the plane to contain:
    # 1. The wire direction
    # 2. The vertical (Z) direction as much as possible
    
    # Normal should be perpendicular to wire direction
    # and lie in the horizontal (XY) plane as much as possible
    z_axis = np.array([0, 0, 1])
    
    # Normal = wire_direction × z_axis (perpendicular to both)
    normal = np.cross(wire_direction, z_axis)
    
    if np.linalg.norm(normal) < 0.1:
        # Wire is nearly vertical, use second PC as guide
        normal = pca.components_[2]
    else:
        normal = normal / np.linalg.norm(normal)
    
    # Create plane
    plane = Plane(normal=normal, point=centroid)
    
    # Override u to be exactly the wire direction (projected onto plane)
    u = wire_direction - np.dot(wire_direction, normal) * normal
    u = u / np.linalg.norm(u)
    
    # v perpendicular to both normal and u
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    
    # Ensure v has positive Z component (so Y in 2D corresponds to height)
    if v[2] < 0:
        v = -v
        u = -u  # Keep right-handed system
    
    plane.u = u
    plane.v = v
    
    return plane
