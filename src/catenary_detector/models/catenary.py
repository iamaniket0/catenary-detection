"""
Catenary curve models for representing hanging cables.

The catenary is the curve formed by a uniform chain or cable hanging
freely under gravity. Mathematical form:

    y(x) = a * cosh(x/a) = a * (e^(x/a) + e^(-x/a)) / 2

This module provides 2D and 3D catenary representations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

from catenary_detector.models.plane import Plane


@dataclass
class CatenaryParams:
    """
    Parameters defining a 2D catenary curve.
    
    The catenary equation is:
        y(x) = y0 + c * (cosh((x - x0) / c) - 1)
    
    This form ensures that:
        - The minimum point is at (x0, y0)
        - c controls the curvature (larger c = flatter curve)
        - y(x0) = y0 (the vertex)
    
    Attributes:
        x0: X-position of the lowest point (vertex)
        y0: Y-position (height) of the lowest point
        c: Curvature parameter (must be positive)
           Physically: c = H/w where H is horizontal tension, w is weight per unit length
    """
    x0: float
    y0: float
    c: float
    
    def __post_init__(self):
        """Validate parameters."""
        if self.c <= 0:
            raise ValueError(f"Curvature parameter c must be positive, got {self.c}")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {'x0': self.x0, 'y0': self.y0, 'c': self.c}
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'CatenaryParams':
        """Create from dictionary."""
        return cls(x0=data['x0'], y0=data['y0'], c=data['c'])


class Catenary2D:
    """
    2D Catenary curve model.
    
    Equation: y(x) = y0 + c * (cosh((x - x0) / c) - 1)
    
    Example:
        >>> params = CatenaryParams(x0=0, y0=10, c=20)
        >>> catenary = Catenary2D(params)
        >>> x = np.linspace(-25, 25, 100)
        >>> y = catenary.evaluate(x)
    """
    
    def __init__(self, params: CatenaryParams):
        """
        Initialize with catenary parameters.
        
        Args:
            params: CatenaryParams object
        """
        self.params = params
    
    @property
    def x0(self) -> float:
        """X-position of minimum."""
        return self.params.x0
    
    @property
    def y0(self) -> float:
        """Y-position of minimum (lowest point)."""
        return self.params.y0
    
    @property
    def c(self) -> float:
        """Curvature parameter."""
        return self.params.c
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the catenary at given x coordinates.
        
        Args:
            x: Array of x-coordinates
            
        Returns:
            Array of y-coordinates (heights)
        """
        # Protect against overflow in cosh
        arg = (x - self.x0) / self.c
        arg = np.clip(arg, -100, 100)
        return self.y0 + self.c * (np.cosh(arg) - 1)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Compute first derivative dy/dx.
        
        Args:
            x: Array of x-coordinates
            
        Returns:
            Array of derivative values
        """
        arg = (x - self.x0) / self.c
        arg = np.clip(arg, -100, 100)
        return np.sinh(arg)
    
    def curvature(self, x: np.ndarray) -> np.ndarray:
        """
        Compute curvature κ at given x coordinates.
        
        κ = |y''| / (1 + y'²)^(3/2)
        
        Args:
            x: Array of x-coordinates
            
        Returns:
            Array of curvature values
        """
        arg = (x - self.x0) / self.c
        arg = np.clip(arg, -100, 100)
        
        y_prime = np.sinh(arg)
        y_double_prime = np.cosh(arg) / self.c
        
        return np.abs(y_double_prime) / (1 + y_prime**2)**(3/2)
    
    def arc_length(self, x1: float, x2: float) -> float:
        """
        Compute arc length between two x-coordinates.
        
        For catenary: L = c * (sinh(x2/c) - sinh(x1/c))
        
        Args:
            x1: Start x-coordinate
            x2: End x-coordinate
            
        Returns:
            Arc length
        """
        arg1 = (x1 - self.x0) / self.c
        arg2 = (x2 - self.x0) / self.c
        return self.c * (np.sinh(arg2) - np.sinh(arg1))
    
    def generate_points(
        self,
        x_min: float,
        x_max: float,
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate evenly spaced points along the catenary.
        
        Args:
            x_min: Start of x range
            x_max: End of x range
            n_points: Number of points
            
        Returns:
            Tuple of (x_values, y_values)
        """
        x = np.linspace(x_min, x_max, n_points)
        y = self.evaluate(x)
        return x, y


@dataclass
class Catenary3D:
    """
    3D Catenary curve model with plane projection.
    
    Represents a catenary in 3D space by combining:
    - A plane containing the wire
    - 2D catenary parameters within that plane
    
    The catenary can be evaluated and plotted in both 2D (plane)
    and 3D (world) coordinate systems.
    
    Attributes:
        wire_id: Identifier for this wire
        plane: The Plane containing this catenary
        params: 2D catenary parameters
        r_squared: Coefficient of determination (fit quality)
        rmse: Root mean square error of fit (in meters)
        n_points: Number of points used in fitting
    """
    wire_id: int
    plane: Plane
    params: CatenaryParams
    r_squared: float = 0.0
    rmse: float = 0.0
    n_points: int = 0
    x_range: Tuple[float, float] = field(default=(0, 0))
    
    def __post_init__(self):
        """Create internal 2D catenary object."""
        self._catenary_2d = Catenary2D(self.params)
    
    @property
    def x0(self) -> float:
        """X-position of minimum in plane coordinates."""
        return self.params.x0
    
    @property
    def y0(self) -> float:
        """Y-position of minimum (lowest height) in plane coordinates."""
        return self.params.y0
    
    @property
    def c(self) -> float:
        """Curvature parameter."""
        return self.params.c
    
    @property
    def is_good_fit(self) -> bool:
        """Check if this is a high-quality fit."""
        return self.r_squared >= 0.95 and self.rmse <= 0.1
    
    @property
    def equation_str(self) -> str:
        """Return catenary equation as string."""
        return f"z(t) = {self.y0:.3f} + {self.c:.3f} × (cosh((t - {self.x0:.3f}) / {self.c:.3f}) - 1)"
    
    def evaluate_2d(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate catenary in 2D plane coordinates.
        
        Args:
            x: X-coordinates in plane coordinate system
            
        Returns:
            Y-coordinates (heights) in plane coordinate system
        """
        return self._catenary_2d.evaluate(x)
    
    def evaluate_3d(self, x_plane: np.ndarray) -> np.ndarray:
        """
        Evaluate catenary and return 3D world coordinates.
        
        Args:
            x_plane: X-coordinates in plane coordinate system
            
        Returns:
            3D points in world coordinates, shape (N, 3)
        """
        # Get 2D catenary points
        y_plane = self.evaluate_2d(x_plane)
        
        # Convert to 3D using plane transformation
        points_2d = np.column_stack([x_plane, y_plane])
        points_3d = self.plane.unproject_points(points_2d)
        
        return points_3d
    
    def generate_curve_2d(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 2D curve points.
        
        Args:
            n_points: Number of points
            
        Returns:
            Tuple of (x_values, y_values) in plane coordinates
        """
        if self.x_range[0] == self.x_range[1]:
            # Default range based on catenary parameter
            half_span = self.c * 2
            x_min, x_max = self.x0 - half_span, self.x0 + half_span
        else:
            x_min, x_max = self.x_range
        
        return self._catenary_2d.generate_points(x_min, x_max, n_points)
    
    def generate_curve_3d(self, n_points: int = 100) -> np.ndarray:
        """
        Generate 3D curve points.
        
        Args:
            n_points: Number of points
            
        Returns:
            Array of shape (n_points, 3) with 3D coordinates
        """
        x, _ = self.generate_curve_2d(n_points)
        return self.evaluate_3d(x)
    
    def distance_to_point(self, point_3d: np.ndarray) -> float:
        """
        Compute minimum distance from a 3D point to the catenary curve.
        
        This is an approximation using sampled curve points.
        
        Args:
            point_3d: 3D point coordinates (3,)
            
        Returns:
            Minimum distance to curve
        """
        curve_points = self.generate_curve_3d(200)
        distances = np.linalg.norm(curve_points - point_3d, axis=1)
        return np.min(distances)
    
    def compute_sag(self) -> float:
        """
        Compute the sag (vertical drop) of the catenary.
        
        Returns:
            Sag in the same units as coordinates (typically meters)
        """
        x_min, x_max = self.x_range
        if x_min == x_max:
            return 0.0
        
        # Sag is difference between endpoint heights and minimum height
        y_at_ends = self.evaluate_2d(np.array([x_min, x_max]))
        return np.max(y_at_ends) - self.y0
    
    def compute_span(self) -> float:
        """
        Compute the horizontal span of the catenary.
        
        Returns:
            Span length
        """
        return abs(self.x_range[1] - self.x_range[0])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'wire_id': self.wire_id,
            'params': self.params.to_dict(),
            'plane': self.plane.to_dict(),
            'r_squared': self.r_squared,
            'rmse': self.rmse,
            'n_points': self.n_points,
            'x_range': list(self.x_range),
            'equation': self.equation_str
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Catenary3D':
        """Create from dictionary."""
        return cls(
            wire_id=data['wire_id'],
            plane=Plane.from_dict(data['plane']),
            params=CatenaryParams.from_dict(data['params']),
            r_squared=data.get('r_squared', 0.0),
            rmse=data.get('rmse', 0.0),
            n_points=data.get('n_points', 0),
            x_range=tuple(data.get('x_range', (0, 0)))
        )
    
    def __repr__(self) -> str:
        return f"Catenary3D(wire={self.wire_id}, R²={self.r_squared:.4f}, c={self.c:.2f})"


# =============================================================================
# Standalone Catenary Function (for curve_fit)
# =============================================================================

def catenary_func(x: np.ndarray, x0: float, y0: float, c: float) -> np.ndarray:
    """
    Catenary function for use with scipy.optimize.curve_fit.
    
    y(x) = y0 + c * (cosh((x - x0) / c) - 1)
    
    Args:
        x: Independent variable (position along wire)
        x0: Position of minimum
        y0: Value at minimum
        c: Curvature parameter (must be positive)
        
    Returns:
        y values
    """
    if c <= 0:
        return np.full_like(x, np.inf)
    
    arg = (x - x0) / c
    arg = np.clip(arg, -100, 100)  # Prevent overflow
    return y0 + c * (np.cosh(arg) - 1)
