"""
Data models for the catenary detector package.

This module provides:
- Plane: 3D plane for point projection
- CatenaryParams, Catenary2D, Catenary3D: Catenary curve models
- Wire, WireCollection: Detected wire representations
- FitMetrics: Quality metrics for curve fitting
"""

from catenary_detector.models.plane import Plane, fit_plane_to_wire
from catenary_detector.models.catenary import (
    CatenaryParams,
    Catenary2D,
    Catenary3D,
    catenary_func
)
from catenary_detector.models.wire import (
    Wire,
    WireCollection,
    FitMetrics
)

__all__ = [
    'Plane',
    'fit_plane_to_wire',
    'CatenaryParams',
    'Catenary2D',
    'Catenary3D',
    'catenary_func',
    'Wire',
    'WireCollection',
    'FitMetrics'
]
