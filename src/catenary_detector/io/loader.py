"""
Point cloud loading utilities.

Supports loading point cloud data from various file formats:
- Parquet (.parquet)
- CSV (.csv)
- NumPy (.npy, .npz)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional, List, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PointCloud:
    """
    Container for point cloud data.
    
    Attributes:
        points: Array of shape (N, 3) with x, y, z coordinates
        source: Original file path or identifier
    """
    points: np.ndarray
    source: Optional[str] = None
    
    def __post_init__(self):
        """Validate and convert points array."""
        self.points = np.asarray(self.points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"Points must be (N, 3) array, got {self.points.shape}")
    
    @property
    def n_points(self) -> int:
        """Number of points."""
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
        """Z coordinates."""
        return self.points[:, 2]
    
    @property
    def bounds(self) -> Dict[str, tuple]:
        """Bounding box."""
        return {
            'x': (self.x.min(), self.x.max()),
            'y': (self.y.min(), self.y.max()),
            'z': (self.z.min(), self.z.max())
        }
    
    @property
    def centroid(self) -> np.ndarray:
        """Center point."""
        return np.mean(self.points, axis=0)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame(self.points, columns=['x', 'y', 'z'])
    
    def __len__(self) -> int:
        return self.n_points
    
    def __repr__(self) -> str:
        return f"PointCloud(n_points={self.n_points}, source='{self.source}')"


class PointCloudLoader:
    """
    Loader for point cloud data from various file formats.
    
    Example:
        >>> loader = PointCloudLoader()
        >>> cloud = loader.load("data/points.parquet")
        >>> print(cloud.n_points)
        1502
    """
    
    SUPPORTED_FORMATS = {'.parquet', '.csv', '.npy', '.npz', '.txt'}
    
    def __init__(
        self,
        x_col: str = 'x',
        y_col: str = 'y',
        z_col: str = 'z'
    ):
        """
        Initialize loader.
        
        Args:
            x_col: Column name for X coordinates
            y_col: Column name for Y coordinates  
            z_col: Column name for Z coordinates
        """
        self.x_col = x_col
        self.y_col = y_col
        self.z_col = z_col
    
    def load(
        self,
        source: Union[str, Path, np.ndarray, pd.DataFrame]
    ) -> PointCloud:
        """
        Load point cloud from various sources.
        
        Args:
            source: File path, numpy array, or DataFrame
            
        Returns:
            PointCloud object
        """
        # Handle numpy array
        if isinstance(source, np.ndarray):
            return self._from_array(source, source="array")
        
        # Handle DataFrame
        if isinstance(source, pd.DataFrame):
            return self._from_dataframe(source, source="dataframe")
        
        # Handle file path
        path = Path(source)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        suffix = path.suffix.lower()
        
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {suffix}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )
        
        logger.info(f"Loading point cloud from {path}")
        
        if suffix == '.parquet':
            return self._load_parquet(path)
        elif suffix == '.csv':
            return self._load_csv(path)
        elif suffix == '.txt':
            return self._load_txt(path)
        elif suffix in {'.npy', '.npz'}:
            return self._load_numpy(path)
    
    def _load_parquet(self, path: Path) -> PointCloud:
        """Load from Parquet file."""
        df = pd.read_parquet(path)
        return self._from_dataframe(df, source=str(path))
    
    def _load_csv(self, path: Path) -> PointCloud:
        """Load from CSV file."""
        df = pd.read_csv(path)
        return self._from_dataframe(df, source=str(path))
    
    def _load_txt(self, path: Path) -> PointCloud:
        """Load from text file (space or comma separated)."""
        try:
            data = np.loadtxt(path, delimiter=',')
        except ValueError:
            data = np.loadtxt(path)
        return self._from_array(data, source=str(path))
    
    def _load_numpy(self, path: Path) -> PointCloud:
        """Load from NumPy file."""
        if path.suffix == '.npy':
            data = np.load(path)
        else:  # .npz
            npz = np.load(path)
            if 'points' in npz:
                data = npz['points']
            else:
                data = npz[npz.files[0]]
        return self._from_array(data, source=str(path))
    
    def _from_dataframe(
        self,
        df: pd.DataFrame,
        source: Optional[str] = None
    ) -> PointCloud:
        """Create PointCloud from DataFrame."""
        required = {self.x_col, self.y_col, self.z_col}
        available = set(df.columns)
        missing = required - available
        
        if missing:
            raise ValueError(
                f"Missing columns: {missing}. "
                f"Available: {list(df.columns)}"
            )
        
        points = df[[self.x_col, self.y_col, self.z_col]].values
        return PointCloud(points=points, source=source)
    
    def _from_array(
        self,
        array: np.ndarray,
        source: Optional[str] = None
    ) -> PointCloud:
        """Create PointCloud from numpy array."""
        if array.ndim == 1:
            raise ValueError("1D array not supported. Expected (N, 3).")
        
        if array.shape[1] != 3:
            raise ValueError(f"Expected 3 columns, got {array.shape[1]}")
        
        return PointCloud(points=array, source=source)
    
    def load_multiple(
        self,
        paths: List[Union[str, Path]],
        combine: bool = False
    ) -> Union[List[PointCloud], PointCloud]:
        """
        Load multiple point clouds.
        
        Args:
            paths: List of file paths
            combine: If True, merge all into one PointCloud
            
        Returns:
            List of PointCloud objects, or single combined PointCloud
        """
        clouds = [self.load(p) for p in paths]
        
        if combine:
            combined = np.vstack([c.points for c in clouds])
            return PointCloud(points=combined, source="combined")
        
        return clouds
