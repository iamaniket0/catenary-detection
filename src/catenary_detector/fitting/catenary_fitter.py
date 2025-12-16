"""
Catenary curve fitting algorithms.

This module provides methods for fitting catenary curves to wire point clouds:
- 2D fitting using non-linear least squares
- 3D fitting with automatic plane projection
- Robust fitting for noisy/sparse data
"""

import numpy as np
from typing import Tuple, Optional, Dict
import warnings
import logging

from scipy.optimize import curve_fit, least_squares

from catenary_detector.config import FittingConfig, DEFAULT_CONFIG
from catenary_detector.models.plane import Plane, fit_plane_to_wire
from catenary_detector.models.catenary import (
    CatenaryParams,
    Catenary3D,
    catenary_func
)
from catenary_detector.models.wire import Wire, WireCollection, FitMetrics

logger = logging.getLogger(__name__)


def compute_initial_guess(
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute initial guess for catenary parameters.
    
    Args:
        x: X-coordinates (along wire)
        y: Y-coordinates (heights)
        
    Returns:
        Tuple of (x0, y0, c) initial estimates
    """
    # x0: position of minimum y
    min_idx = np.argmin(y)
    x0 = x[min_idx]
    
    # y0: minimum y value
    y0 = y[min_idx]
    
    # c: estimate from span and sag
    span = x.max() - x.min()
    sag = y.max() - y.min()
    
    if sag > 0.01:
        # For shallow catenary: c ≈ span² / (8 * sag)
        c = span**2 / (8 * sag)
        c = np.clip(c, 1.0, 200.0)
    else:
        c = 10.0
    
    return x0, y0, c


def compute_fit_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Compute fitting quality metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Tuple of (r_squared, rmse, max_error, residual_std)
    """
    residuals = y_true - y_pred
    
    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # RMSE
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Max error
    max_error = np.max(np.abs(residuals))
    
    # Residual std
    residual_std = np.std(residuals)
    
    return r_squared, rmse, max_error, residual_std


class CatenaryFitter:
    """
    Fits catenary curves to wire point clouds.
    
    Example:
        >>> fitter = CatenaryFitter()
        >>> catenary = fitter.fit_wire(wire)
        >>> print(f"R² = {catenary.r_squared:.4f}")
    """
    
    def __init__(self, config: Optional[FittingConfig] = None):
        """
        Initialize fitter.
        
        Args:
            config: Fitting configuration (uses defaults if None)
        """
        self.config = config or DEFAULT_CONFIG.fitting
    
    def fit_wire(self, wire: Wire) -> Optional[Catenary3D]:
        """
        Fit catenary curve to a wire's point cloud.
        
        This is the main fitting method:
        1. Fits a plane to the 3D points
        2. Projects points to 2D plane coordinates
        3. Fits 2D catenary using non-linear least squares
        4. Creates 3D catenary with plane transformation
        
        Args:
            wire: Wire object with point cloud
            
        Returns:
            Catenary3D object if successful, None if fitting fails
        """
        points = wire.points
        
        if len(points) < self.config.min_points:
            logger.warning(
                f"Wire {wire.wire_id}: Not enough points "
                f"({len(points)} < {self.config.min_points})"
            )
            return None
        
        logger.info(f"Fitting catenary to wire {wire.wire_id} ({len(points)} points)")
        
        # Step 1: Fit plane to points
        plane = fit_plane_to_wire(points)
        
        # Step 2: Project to 2D
        points_2d = plane.project_points(points)
        x_data = points_2d[:, 0]
        y_data = points_2d[:, 1]
        
        # Sort by x for stable fitting
        sort_idx = np.argsort(x_data)
        x_sorted = x_data[sort_idx]
        y_sorted = y_data[sort_idx]
        
        # Step 3: Fit 2D catenary
        params = self._fit_2d(x_sorted, y_sorted, wire.wire_id)
        
        if params is None:
            return None
        
        x0, y0, c, r_squared, rmse, max_error, residual_std = params
        
        # Check fit quality
        if r_squared < self.config.min_r_squared:
            logger.warning(
                f"Wire {wire.wire_id}: Poor fit quality "
                f"(R²={r_squared:.4f} < {self.config.min_r_squared})"
            )
        
        # Step 4: Create Catenary3D object
        catenary = Catenary3D(
            wire_id=wire.wire_id,
            plane=plane,
            params=CatenaryParams(x0=x0, y0=y0, c=c),
            r_squared=r_squared,
            rmse=rmse,
            n_points=len(points),
            x_range=(x_sorted.min(), x_sorted.max())
        )
        
        # Store in wire
        wire.catenary = catenary
        
        logger.info(f"Wire {wire.wire_id}: R²={r_squared:.4f}, RMSE={rmse:.4f}m, c={c:.2f}")
        
        return catenary
    
    def _fit_2d(
        self,
        x: np.ndarray,
        y: np.ndarray,
        wire_id: int = 0
    ) -> Optional[Tuple[float, float, float, float, float, float, float]]:
        """
        Fit 2D catenary to data.
        
        Args:
            x: X-coordinates
            y: Y-coordinates (heights)
            wire_id: Wire identifier for logging
            
        Returns:
            Tuple of (x0, y0, c, r_squared, rmse, max_error, residual_std)
            or None if fitting fails
        """
        # Initial guess
        x0_init, y0_init, c_init = compute_initial_guess(x, y)
        
        # Bounds
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        c_min, c_max = self.config.c_bounds
        
        bounds = (
            [x.min() - x_range, y.min() - y_range, c_min],
            [x.max() + x_range, y.min() + y_range * 0.5, c_max]
        )
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                if self.config.use_robust_fitting:
                    # Robust fitting using least_squares
                    def residuals(params):
                        x0, y0, c = params
                        if c <= 0:
                            return np.full_like(y, 1e10)
                        return y - catenary_func(x, x0, y0, c)
                    
                    result = least_squares(
                        residuals,
                        [x0_init, y0_init, c_init],
                        bounds=([bounds[0][0], bounds[0][1], c_min],
                               [bounds[1][0], bounds[1][1], c_max]),
                        loss=self.config.robust_loss,
                        max_nfev=self.config.max_iterations
                    )
                    
                    if not result.success:
                        logger.warning(f"Wire {wire_id}: Robust fitting did not converge")
                    
                    x0, y0, c = result.x
                    
                else:
                    # Standard curve_fit
                    popt, _ = curve_fit(
                        catenary_func,
                        x, y,
                        p0=[x0_init, y0_init, c_init],
                        bounds=bounds,
                        maxfev=self.config.max_iterations
                    )
                    x0, y0, c = popt
            
            # Compute metrics
            y_pred = catenary_func(x, x0, y0, c)
            r_squared, rmse, max_error, residual_std = compute_fit_metrics(y, y_pred)
            
            return x0, y0, c, r_squared, rmse, max_error, residual_std
            
        except Exception as e:
            logger.warning(f"Wire {wire_id}: Fitting failed - {e}")
            return None
    
    def fit_collection(self, collection: WireCollection) -> WireCollection:
        """
        Fit catenaries to all wires in a collection.
        
        Args:
            collection: WireCollection with detected wires
            
        Returns:
            Same collection with fitted catenaries
        """
        logger.info(f"Fitting catenaries to {collection.n_wires} wires")
        
        successful = 0
        for wire in collection:
            catenary = self.fit_wire(wire)
            if catenary is not None:
                successful += 1
        
        logger.info(f"Successfully fitted {successful}/{collection.n_wires} wires")
        return collection
    
    def fit_points(self, points: np.ndarray, wire_id: int = 0) -> Optional[Catenary3D]:
        """
        Convenience method to fit catenary directly to points.
        
        Args:
            points: 3D point array (N, 3)
            wire_id: Wire identifier
            
        Returns:
            Catenary3D if successful
        """
        wire = Wire(wire_id=wire_id, points=points)
        return self.fit_wire(wire)


def fit_catenary_simple(
    points: np.ndarray,
    wire_id: int = 0
) -> Optional[Catenary3D]:
    """
    Simple function to fit catenary to 3D points.
    
    Args:
        points: 3D points (N, 3)
        wire_id: Wire identifier
        
    Returns:
        Catenary3D if successful
    """
    fitter = CatenaryFitter()
    return fitter.fit_points(points, wire_id)
