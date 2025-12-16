"""
Visualization functions for catenary detection results.

Provides plotting utilities for:
- Point cloud visualization
- Fitted catenary curves
- Quality assessment plots
- Multi-view analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import Optional, Dict
import logging

from catenary_detector.config import VisualizationConfig, DEFAULT_CONFIG
from catenary_detector.models.wire import Wire, WireCollection

logger = logging.getLogger(__name__)


class CatenaryVisualizer:
    """Visualizer for catenary detection results."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or DEFAULT_CONFIG.visualization

    def plot_results(
        self,
        collection: WireCollection,
        title: str = "Catenary Detection Results",
        show_fitted: bool = True
    ) -> plt.Figure:
        fig = plt.figure(figsize=self.config.figure_size)
        fig.suptitle(title, fontsize=14, fontweight='bold')

        gs = GridSpec(2, 2, figure=fig)

        # 3D view
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        self._plot_3d(ax1, collection, show_fitted)
        ax1.set_title(f'3D View ({collection.n_wires} wires)')

        # Side view (Y-Z)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_2d(ax2, collection, 'y', 'z', show_fitted)
        ax2.set_title('Side View (Y-Z) - Catenary Shape')

        # Top view (X-Y)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_2d(ax3, collection, 'x', 'y', show_fitted=False)
        ax3.set_title('Top View (X-Y)')

        # Metrics or Front view
        ax4 = fig.add_subplot(gs[1, 1])
        if collection.is_fitted:
            self._plot_metrics(ax4, collection)
        else:
            self._plot_2d(ax4, collection, 'x', 'z', show_fitted=False)
            ax4.set_title('Front View (X-Z)')

        plt.tight_layout()
        return fig

    def _plot_3d(self, ax: plt.Axes, collection: WireCollection, show_fitted: bool):
        for i, wire in enumerate(collection):
            color = self.config.wire_colors[i % len(self.config.wire_colors)]
            ax.scatter(wire.x, wire.y, wire.z,
                       s=self.config.point_size, alpha=self.config.alpha,
                       c=color, label=f'Wire {wire.wire_id}')
            if show_fitted and wire.is_fitted:
                curve = wire.get_fitted_curve_3d(100)
                if curve is not None:
                    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2],
                            color='black', linewidth=self.config.line_width, linestyle='--')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend(fontsize=8)

    def _plot_2d(self, ax: plt.Axes, collection: WireCollection, x_axis: str, y_axis: str, show_fitted: bool = True):
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        xi, yi = axis_map[x_axis], axis_map[y_axis]
        for i, wire in enumerate(collection):
            color = self.config.wire_colors[i % len(self.config.wire_colors)]
            ax.scatter(wire.points[:, xi], wire.points[:, yi],
                       s=self.config.point_size, alpha=self.config.alpha, c=color, label=f'Wire {wire.wire_id}')
            if show_fitted and wire.is_fitted:
                curve = wire.get_fitted_curve_3d(100)
                if curve is not None:
                    ax.plot(curve[:, xi], curve[:, yi],
                            color='black', linewidth=self.config.line_width, linestyle='--')
        ax.set_xlabel(f'{x_axis.upper()} (m)')
        ax.set_ylabel(f'{y_axis.upper()} (m)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_metrics(self, ax: plt.Axes, collection: WireCollection):
        wire_ids, r_squared, colors = [], [], []
        for i, wire in enumerate(collection):
            if wire.is_fitted:
                wire_ids.append(f'Wire {wire.wire_id}')
                r_squared.append(wire.catenary.r_squared)
                colors.append(self.config.wire_colors[i % len(self.config.wire_colors)])
        if wire_ids:
            bars = ax.bar(wire_ids, r_squared, color=colors)
            ax.set_ylabel('R²')
            ax.set_ylim(0, 1.05)
            ax.axhline(0.95, color='green', linestyle='--', alpha=0.5, label='Good fit')
            ax.axhline(0.99, color='blue', linestyle='--', alpha=0.5, label='Excellent fit')
            ax.legend(fontsize=8)
            ax.set_title('Fit Quality')
            for bar, val in zip(bars, r_squared):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No fitted wires', ha='center', va='center', transform=ax.transAxes, fontsize=12)

    def save(self, fig: plt.Figure, filepath: str, dpi: Optional[int] = None):
        dpi = dpi or self.config.dpi
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved figure to {filepath}")

    def plot_wire_validation(self, collection: WireCollection, title: str = "Wire Cluster Validation", show_centroids: bool = True) -> plt.Figure:
        fig = plt.figure(figsize=self.config.figure_size)
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle(title, fontsize=14, fontweight='bold')
        colors = self.config.wire_colors

        print(f"Validation Report: {collection.n_wires} wires detected")
        for i, wire in enumerate(collection.wires):
            pts = wire.points
            centroid = wire.centroid
            print(f"Wire {wire.wire_id}: {len(pts)} points, centroid at {centroid}")
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       s=self.config.point_size, alpha=self.config.alpha,
                       c=colors[i % len(colors)], label=f'Wire {wire.wire_id}')
            if show_centroids:
                ax.scatter(centroid[0], centroid[1], centroid[2], s=50, c='black', marker='x')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend(fontsize=8)
        plt.tight_layout()
        return fig


# --- Module-level convenience functions ---
def plot_results(collection: WireCollection, output_path: Optional[str] = None, show: bool = True) -> plt.Figure:
    viz = CatenaryVisualizer()
    fig = viz.plot_results(collection)
    if output_path:
        viz.save(fig, output_path)
    if show:
        plt.show()
    return fig


def plot_wire_validation(collection: WireCollection, output_path: Optional[str] = None, show: bool = True) -> plt.Figure:
    viz = CatenaryVisualizer()
    fig = viz.plot_wire_validation(collection)
    if output_path:
        viz.save(fig, output_path)
    if show:
        plt.show()
    return fig

