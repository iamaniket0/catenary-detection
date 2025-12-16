"""
Wire and WireCollection models for representing detected cables.

A Wire represents a single detected cable with its:
- Point cloud data
- Fitted catenary curve (if fitted)
- Quality metrics
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Iterator
import json

from catenary_detector.models.catenary import Catenary3D


@dataclass
class FitMetrics:
    """
    Quality metrics for a catenary fit.
    
    Attributes:
        r_squared: Coefficient of determination (0-1, higher is better)
        rmse: Root mean square error in meters
        max_error: Maximum absolute residual error
        residual_std: Standard deviation of residuals
        n_points: Number of points used in fitting
    """
    r_squared: float
    rmse: float
    max_error: float
    residual_std: float
    n_points: int
    
    @property
    def is_good_fit(self) -> bool:
        """Check if fit quality is acceptable."""
        return self.r_squared >= 0.95 and self.rmse <= 0.1
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'r_squared': self.r_squared,
            'rmse': self.rmse,
            'max_error': self.max_error,
            'residual_std': self.residual_std,
            'n_points': self.n_points
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FitMetrics':
        """Create from dictionary."""
        return cls(**data)
    
    def __repr__(self) -> str:
        return f"FitMetrics(R²={self.r_squared:.4f}, RMSE={self.rmse:.4f}m)"


@dataclass
class Wire:
    """
    Represents a single detected wire.
    
    Contains the point cloud data for a single wire and optionally
    a fitted catenary curve.
    
    Attributes:
        wire_id: Unique identifier for this wire
        points: Point cloud array of shape (N, 3)
        catenary: Fitted catenary curve (None if not fitted)
        cluster_label: Original cluster label from detection
    """
    wire_id: int
    points: np.ndarray
    catenary: Optional[Catenary3D] = None
    cluster_label: int = -1
    
    def __post_init__(self):
        """Validate points array."""
        self.points = np.asarray(self.points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"Points must be (N, 3) array, got {self.points.shape}")
    
    # -------------------------------------------------------------------------
    # Properties for accessing point data
    # -------------------------------------------------------------------------
    
    @property
    def n_points(self) -> int:
        """Number of points in this wire."""
        return len(self.points)
    
    @property
    def x(self) -> np.ndarray:
        """X coordinates."""
        return self.points[:, 0]
    
    @property
    def y(self) -> np.ndarray:
        """Y coordinates."""
        return self.points[:, 1]
    
    @property
    def z(self) -> np.ndarray:
        """Z coordinates (heights)."""
        return self.points[:, 2]
    
    @property
    def centroid(self) -> np.ndarray:
        """Center point of the wire."""
        return np.mean(self.points, axis=0)
    
    @property
    def bounds(self) -> Dict[str, tuple]:
        """Bounding box for each axis."""
        return {
            'x': (self.x.min(), self.x.max()),
            'y': (self.y.min(), self.y.max()),
            'z': (self.z.min(), self.z.max())
        }
    
    @property
    def z_range(self) -> float:
        """Height range (indicates sag amount)."""
        return self.z.max() - self.z.min()
    
    @property
    def span_length(self) -> float:
        """Approximate span length (XY distance between endpoints)."""
        return np.sqrt(
            (self.x.max() - self.x.min())**2 +
            (self.y.max() - self.y.min())**2
        )
    
    # -------------------------------------------------------------------------
    # Fitting-related properties and methods
    # -------------------------------------------------------------------------
    
    @property
    def is_fitted(self) -> bool:
        """Check if catenary has been fitted."""
        return self.catenary is not None
    
    @property
    def metrics(self) -> Optional[FitMetrics]:
        """Get fit metrics (if fitted)."""
        if not self.is_fitted:
            return None
        return FitMetrics(
            r_squared=self.catenary.r_squared,
            rmse=self.catenary.rmse,
            max_error=0.0,  # Would need to compute from residuals
            residual_std=self.catenary.rmse,  # Approximation
            n_points=self.catenary.n_points
        )
    
    def get_fitted_curve_2d(self, n_points: int = 100) -> Optional[tuple]:
        """
        Get 2D fitted curve points.
        
        Args:
            n_points: Number of points to generate
            
        Returns:
            Tuple of (x, y) arrays or None if not fitted
        """
        if not self.is_fitted:
            return None
        return self.catenary.generate_curve_2d(n_points)
    
    def get_fitted_curve_3d(self, n_points: int = 100) -> Optional[np.ndarray]:
        """
        Get 3D fitted curve points.
        
        Args:
            n_points: Number of points to generate
            
        Returns:
            Array of shape (n_points, 3) or None if not fitted
        """
        if not self.is_fitted:
            return None
        return self.catenary.generate_curve_3d(n_points)
    
    def get_residuals(self) -> Optional[np.ndarray]:
        """
        Compute fitting residuals for each point.
        
        Returns:
            Array of residuals (N,) or None if not fitted
        """
        if not self.is_fitted:
            return None
        
        # Project points to 2D
        points_2d = self.catenary.plane.project_points(self.points)
        
        # Get predicted y values
        y_pred = self.catenary.evaluate_2d(points_2d[:, 0])
        
        # Residuals
        return points_2d[:, 1] - y_pred
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_dict(self, include_points: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Args:
            include_points: Whether to include point coordinates
            
        Returns:
            Dictionary representation
        """
        result = {
            'wire_id': self.wire_id,
            'n_points': self.n_points,
            'centroid': self.centroid.tolist(),
            'bounds': {k: list(v) for k, v in self.bounds.items()},
            'z_range': self.z_range,
            'span_length': self.span_length,
            'cluster_label': self.cluster_label,
            'is_fitted': self.is_fitted
        }
        
        if include_points:
            result['points'] = self.points.tolist()
        
        if self.is_fitted:
            result['catenary'] = self.catenary.to_dict()
        
        return result
    
    def __repr__(self) -> str:
        status = f"R²={self.catenary.r_squared:.3f}" if self.is_fitted else "not fitted"
        return f"Wire(id={self.wire_id}, points={self.n_points}, {status})"
    
    def __len__(self) -> int:
        return self.n_points


class WireCollection:
    """
    Collection of detected wires.
    
    Provides methods for managing multiple wires and aggregate statistics.
    
    Example:
        >>> collection = WireCollection()
        >>> collection.add_wire(wire1)
        >>> collection.add_wire(wire2)
        >>> print(f"Total: {collection.n_wires} wires, {collection.total_points} points")
    """
    
    def __init__(
        self,
        wires: Optional[List[Wire]] = None,
        source: Optional[str] = None
    ):
        """
        Initialize wire collection.
        
        Args:
            wires: List of Wire objects
            source: Source identifier (e.g., filename)
        """
        self._wires: List[Wire] = wires or []
        self.source = source
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def n_wires(self) -> int:
        """Number of wires in collection."""
        return len(self._wires)
    
    @property
    def total_points(self) -> int:
        """Total points across all wires."""
        return sum(w.n_points for w in self._wires)
    
    @property
    def wires(self) -> List[Wire]:
        """List of wires."""
        return self._wires
    
    @property
    def is_fitted(self) -> bool:
        """Check if all wires have been fitted."""
        return all(w.is_fitted for w in self._wires) if self._wires else False
    
    @property
    def all_points(self) -> np.ndarray:
        """All points from all wires combined."""
        if not self._wires:
            return np.array([]).reshape(0, 3)
        return np.vstack([w.points for w in self._wires])
    
    # -------------------------------------------------------------------------
    # Wire management
    # -------------------------------------------------------------------------
    
    def add_wire(self, wire: Wire) -> None:
        """Add a wire to the collection."""
        self._wires.append(wire)
    
    def get_wire(self, wire_id: int) -> Optional[Wire]:
        """Get wire by ID."""
        for wire in self._wires:
            if wire.wire_id == wire_id:
                return wire
        return None
    
    def remove_wire(self, wire_id: int) -> bool:
        """
        Remove wire by ID.
        
        Returns:
            True if wire was found and removed
        """
        for i, wire in enumerate(self._wires):
            if wire.wire_id == wire_id:
                del self._wires[i]
                return True
        return False
    
    def sort_by_height(self, descending: bool = True) -> None:
        """Sort wires by average height."""
        self._wires.sort(key=lambda w: w.centroid[2], reverse=descending)
    
    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the collection."""
        if not self._wires:
            return {'n_wires': 0, 'total_points': 0}
        
        summary = {
            'source': self.source,
            'n_wires': self.n_wires,
            'total_points': self.total_points,
            'all_fitted': self.is_fitted,
        }
        
        # Points per wire statistics
        points_per_wire = [w.n_points for w in self._wires]
        summary['points_per_wire'] = {
            'min': min(points_per_wire),
            'max': max(points_per_wire),
            'mean': np.mean(points_per_wire)
        }
        
        # Fit metrics (if fitted)
        fitted_wires = [w for w in self._wires if w.is_fitted]
        if fitted_wires:
            r_squared_values = [w.catenary.r_squared for w in fitted_wires]
            rmse_values = [w.catenary.rmse for w in fitted_wires]
            
            summary['fit_metrics'] = {
                'n_fitted': len(fitted_wires),
                'mean_r_squared': np.mean(r_squared_values),
                'min_r_squared': np.min(r_squared_values),
                'max_r_squared': np.max(r_squared_values),
                'mean_rmse': np.mean(rmse_values),
                'max_rmse': np.max(rmse_values)
            }
        
        return summary
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_dict(self, include_points: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'source': self.source,
            'n_wires': self.n_wires,
            'total_points': self.total_points,
            'wires': [w.to_dict(include_points) for w in self._wires],
            'summary': self.get_summary()
        }
    
    def to_json(self, filepath: str, include_points: bool = False) -> None:
        """Save to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(include_points), f, indent=2)
    
    # -------------------------------------------------------------------------
    # Iteration and indexing
    # -------------------------------------------------------------------------
    
    def __iter__(self) -> Iterator[Wire]:
        return iter(self._wires)
    
    def __len__(self) -> int:
        return self.n_wires
    
    def __getitem__(self, idx: int) -> Wire:
        return self._wires[idx]
    
    def __repr__(self) -> str:
        fitted_str = "all fitted" if self.is_fitted else "not all fitted"
        return f"WireCollection(n_wires={self.n_wires}, points={self.total_points}, {fitted_str})"
