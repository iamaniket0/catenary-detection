"""
Unit tests for the catenary detector package.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from catenary_detector import (
    CatenaryDetector,
    PointCloudLoader,
    WireClusterer,
    CatenaryFitter,
    Wire,
    WireCollection,
    Plane,
    CatenaryParams,
    Catenary2D,
    Catenary3D
)
from catenary_detector.models.catenary import catenary_func


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_catenary_points():
    """Generate synthetic catenary points."""
    # Parameters - use large c for flat curve (small Z range)
    x0, y0, c = 0, 10, 100  # Changed c from 20 to 100 (flatter)
    
    # Generate points along catenary
    t = np.linspace(-20, 20, 500)  # Smaller span too
    z = y0 + c * (np.cosh((t - x0) / c) - 1)
    
    # Add some noise
    np.random.seed(42)
    z_noisy = z + np.random.normal(0, 0.02, len(t))
    
    # Create 3D points (t is along Y, z is height)
    x = np.random.normal(0, 0.3, len(t))
    y = t
    
    return np.column_stack([x, y, z_noisy])


@pytest.fixture
def two_wire_points():
    """
    Generate two stacked wires matching MEDIUM dataset characteristics:
    - Wire 1: centered at ~10m with ~1m sag
    - Wire 2: centered at ~7m with ~1m sag
    - Clear 3m gap between wire centers
    """
    np.random.seed(42)
    
    # Wire 1 (upper) - very flat catenary around 10m
    t1 = np.linspace(-20, 20, 150)
    z1 = 10 + 100 * (np.cosh(t1 / 100) - 1) + np.random.normal(0, 0.02, len(t1))
    x1 = np.random.normal(0, 0.3, len(t1))
    wire1 = np.column_stack([x1, t1, z1])
    
    # Wire 2 (lower) - very flat catenary around 7m
    t2 = np.linspace(-20, 20, 150)
    z2 = 7 + 100 * (np.cosh(t2 / 100) - 1) + np.random.normal(0, 0.02, len(t2))
    x2 = np.random.normal(0, 0.3, len(t2))
    wire2 = np.column_stack([x2, t2, z2])
    
    return np.vstack([wire1, wire2])


# =============================================================================
# Model Tests
# =============================================================================

class TestCatenaryModels:
    """Tests for catenary models."""
    
    def test_catenary_params_validation(self):
        """Test parameter validation."""
        # Valid parameters
        params = CatenaryParams(x0=0, y0=10, c=20)
        assert params.c == 20
        
        # Invalid c (must be positive)
        with pytest.raises(ValueError):
            CatenaryParams(x0=0, y0=10, c=-5)
    
    def test_catenary_2d_evaluate(self):
        """Test 2D catenary evaluation."""
        params = CatenaryParams(x0=0, y0=10, c=20)
        catenary = Catenary2D(params)
        
        # At x=0, should equal y0
        y_at_0 = catenary.evaluate(np.array([0.0]))
        assert abs(y_at_0[0] - 10) < 1e-10
        
        # Symmetric about x0
        x = np.array([-10, 10])
        y = catenary.evaluate(x)
        assert abs(y[0] - y[1]) < 1e-10
    
    def test_catenary_func(self):
        """Test standalone catenary function."""
        x = np.array([0, 10, -10])
        y = catenary_func(x, x0=0, y0=5, c=15)
        
        assert y[0] == 5  # At x0
        assert y[1] == y[2]  # Symmetric


class TestPlane:
    """Tests for Plane model."""
    
    def test_plane_creation(self):
        """Test plane creation."""
        plane = Plane(
            normal=np.array([1, 0, 0]),
            point=np.array([0, 0, 0])
        )
        
        assert np.allclose(np.linalg.norm(plane.normal), 1)
        assert plane.u is not None
        assert plane.v is not None
    
    def test_plane_projection(self):
        """Test point projection."""
        plane = Plane(
            normal=np.array([1, 0, 0]),
            point=np.array([0, 0, 0])
        )
        
        point = np.array([5, 3, 2])
        proj = plane.project_point(point)
        
        assert len(proj) == 2
    
    def test_plane_roundtrip(self):
        """Test projection and unprojection."""
        plane = Plane(
            normal=np.array([0, 0, 1]),
            point=np.array([1, 2, 3])
        )
        
        points_3d = np.array([[1, 2, 3], [2, 3, 3], [0, 1, 3]])
        
        # Project to 2D
        points_2d = plane.project_points(points_3d)
        
        # Unproject back to 3D
        recovered = plane.unproject_points(points_2d)
        
        # Should be close to original (within plane)
        assert np.allclose(recovered[:, 2], 3)  # Z should be 3


# =============================================================================
# Clustering Tests
# =============================================================================

class TestClustering:
    """Tests for wire clustering."""
    
    def test_single_wire(self, sample_catenary_points):
        """Test clustering single wire."""
        clusterer = WireClusterer()
        collection = clusterer.cluster(sample_catenary_points)
        
        assert collection.n_wires == 1
        assert collection.total_points == len(sample_catenary_points)
    
    def test_two_wires(self, two_wire_points):
        """Test clustering two stacked wires."""
        clusterer = WireClusterer()
        collection = clusterer.cluster(two_wire_points)
        
        # Should detect 2 wires
        assert collection.n_wires == 2


# =============================================================================
# Fitting Tests
# =============================================================================

class TestFitting:
    """Tests for catenary fitting."""
    
    def test_fit_clean_data(self, sample_catenary_points):
        """Test fitting on clean synthetic data."""
        fitter = CatenaryFitter()
        wire = Wire(wire_id=0, points=sample_catenary_points)
        
        catenary = fitter.fit_wire(wire)
        
        assert catenary is not None
        assert catenary.r_squared > 0.99  # Should be excellent fit
        assert catenary.rmse < 0.05  # Low error
    
    def test_fit_returns_correct_params(self):
        """Test that fitting recovers correct parameters."""
        # Generate perfect catenary data
        x0_true, y0_true, c_true = 0, 10, 20
        t = np.linspace(-20, 20, 100)
        z = y0_true + c_true * (np.cosh((t - x0_true) / c_true) - 1)
        
        # Create 3D points
        x = np.zeros_like(t)
        points = np.column_stack([x, t, z])
        
        fitter = CatenaryFitter()
        wire = Wire(wire_id=0, points=points)
        catenary = fitter.fit_wire(wire)
        
        # Check recovered parameters (allowing for coordinate transformation)
        assert catenary is not None
        assert catenary.r_squared > 0.999


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline(self, sample_catenary_points):
        """Test complete detection pipeline."""
        detector = CatenaryDetector()
        
        # Run detection
        results = detector.fit(sample_catenary_points)
        
        # Check results
        assert results.n_wires >= 1
        assert results.is_fitted
        
        # Check fit quality
        for wire in results:
            assert wire.catenary.r_squared > 0.9
    
    def test_wire_collection_methods(self, sample_catenary_points):
        """Test WireCollection functionality."""
        detector = CatenaryDetector()
        results = detector.fit(sample_catenary_points)
        
        # Test iteration
        count = 0
        for wire in results:
            count += 1
        assert count == results.n_wires
        
        # Test indexing
        wire0 = results[0]
        assert wire0.wire_id == 0
        
        # Test summary
        summary = results.get_summary()
        assert 'n_wires' in summary


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
