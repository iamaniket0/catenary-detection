"""
Streamlit App: LiDAR Cable Catenary Detection Results
======================================================
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from catenary_detector import CatenaryDetector
from catenary_detector.models.catenary import catenary_func

# Page config
st.set_page_config(
    page_title="LiDAR Cable Detection",
    page_icon="⚡",
    layout="wide"
)

# Title
st.title("⚡ LiDAR Cable Catenary Detection")
st.markdown("**Automatic wire detection and catenary curve fitting from LiDAR point clouds**")

# Sidebar
st.sidebar.header("Settings")
dataset_choice = st.sidebar.selectbox(
    "Select Dataset",
    ["EASY", "MEDIUM", "HARD", "EXTRAHARD", "All Datasets"]
)

show_points = st.sidebar.checkbox("Show Raw Points", value=True)
show_fitted = st.sidebar.checkbox("Show Fitted Curves", value=True)
point_size = st.sidebar.slider("Point Size", 1, 10, 3)

# Dataset paths
DATASETS = {
    'EASY': ('data/lidar_cable_points_easy.parquet', 1),
    'MEDIUM': ('data/lidar_cable_points_medium.parquet', 2),
    'HARD': ('data/lidar_cable_points_hard.parquet', 1),
    'EXTRAHARD': ('data/lidar_cable_points_extrahard.parquet', 2)
}

@st.cache_resource
def load_and_fit(dataset_name):
    """Load dataset and fit catenaries."""
    path, expected = DATASETS[dataset_name]
    detector = CatenaryDetector()
    results = detector.fit(path)
    points = pd.read_parquet(path).values
    return points, results, expected

def create_3d_plot(points, results, title, show_pts=True, show_fit=True, pt_size=3):
    """Create 3D plotly figure."""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    if results.n_wires == 1:
        # Single wire
        if show_pts:
            fig.add_trace(go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='markers',
                marker=dict(size=pt_size, color='steelblue', opacity=0.6),
                name=f'Points ({len(points)})'
            ))
        
        if show_fit and results.wires[0].is_fitted:
            wire = results.wires[0]
            cat = wire.catenary
            
            # Generate fitted curve using generate_curve_3d
            curve_pts = cat.generate_curve_3d(n_points=200)
            
            fig.add_trace(go.Scatter3d(
                x=curve_pts[:, 0], y=curve_pts[:, 1], z=curve_pts[:, 2],
                mode='lines',
                line=dict(color='red', width=6),
                name=f'Fitted (R²={cat.r_squared:.4f})'
            ))
    else:
        # Multiple wires
        for i, wire in enumerate(results.wires):
            color = colors[i % len(colors)]
            
            if show_pts:
                fig.add_trace(go.Scatter3d(
                    x=wire.points[:, 0], y=wire.points[:, 1], z=wire.points[:, 2],
                    mode='markers',
                    marker=dict(size=pt_size, color=color, opacity=0.6),
                    name=f'Wire {i+1} ({len(wire.points)} pts)'
                ))
            
            if show_fit and wire.is_fitted:
                cat = wire.catenary
                
                # Generate fitted curve using generate_curve_3d
                curve_pts = cat.generate_curve_3d(n_points=200)
                
                fig.add_trace(go.Scatter3d(
                    x=curve_pts[:, 0], y=curve_pts[:, 1], z=curve_pts[:, 2],
                    mode='lines',
                    line=dict(color=color, width=6),
                    name=f'Fit {i+1} (R²={cat.r_squared:.4f})'
                ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_2d_profile(results, title):
    """Create 2D catenary profile plot."""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, wire in enumerate(results.wires):
        if not wire.is_fitted:
            continue
            
        cat = wire.catenary
        color = colors[i % len(colors)]
        
        # Project points to 2D using project_points
        plane = cat.plane
        pts_2d = plane.project_points(wire.points)
        
        fig.add_trace(go.Scatter(
            x=pts_2d[:, 0], y=pts_2d[:, 1],
            mode='markers',
            marker=dict(size=4, color=color, opacity=0.5),
            name=f'Wire {i+1} Points'
        ))
        
        # Fitted curve - use params attribute
        x_fit = np.linspace(pts_2d[:, 0].min(), pts_2d[:, 0].max(), 200)
        params = cat.params
        y_fit = catenary_func(x_fit, params.x0, params.y0, params.c)
        
        fig.add_trace(go.Scatter(
            x=x_fit, y=y_fit,
            mode='lines',
            line=dict(color=color, width=3),
            name=f'Wire {i+1} Fit (R²={cat.r_squared:.4f})'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Distance along wire (m)',
        yaxis_title='Height (m)',
        height=400,
        yaxis=dict(scaleanchor='x', scaleratio=1)
    )
    
    return fig

# Main content
if dataset_choice == "All Datasets":
    # Show all datasets in grid
    st.header("All Datasets Overview")
    
    cols = st.columns(2)
    
    results_summary = []
    
    for idx, (name, (path, expected)) in enumerate(DATASETS.items()):
        col = cols[idx % 2]
        
        with col:
            with st.spinner(f'Processing {name}...'):
                points, results, expected = load_and_fit(name)
            
            status = "✅" if results.n_wires == expected else "⚠️"
            st.subheader(f"{status} {name}")
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Wires Detected", results.n_wires)
            m2.metric("Expected", expected)
            
            if results.wires[0].is_fitted:
                avg_r2 = np.mean([w.catenary.r_squared for w in results.wires if w.is_fitted])
                m3.metric("Avg R²", f"{avg_r2:.4f}")
            
            # 3D plot
            fig = create_3d_plot(points, results, f"{name} - 3D View", 
                               show_points, show_fitted, point_size)
            st.plotly_chart(fig, use_container_width=True)
            
            # Store results
            for i, wire in enumerate(results.wires):
                if wire.is_fitted:
                    results_summary.append({
                        'Dataset': name,
                        'Wire': i + 1,
                        'Points': len(wire.points),
                        'R²': wire.catenary.r_squared,
                        'c (curvature)': wire.catenary.params.c
                    })
    
    # Summary table
    st.header("📊 Results Summary")
    df = pd.DataFrame(results_summary)
    st.dataframe(df, use_container_width=True)
    
else:
    # Single dataset view
    st.header(f"Dataset: {dataset_choice}")
    
    with st.spinner('Processing...'):
        points, results, expected = load_and_fit(dataset_choice)
    
    # Status
    if results.n_wires == expected:
        st.success(f"✅ Correctly detected {results.n_wires} wire(s)")
    else:
        st.warning(f"⚠️ Detected {results.n_wires} wire(s), expected {expected}")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Points", len(points))
    col2.metric("Wires Detected", results.n_wires)
    col3.metric("Expected Wires", expected)
    
    if results.wires[0].is_fitted:
        avg_r2 = np.mean([w.catenary.r_squared for w in results.wires if w.is_fitted])
        col4.metric("Average R²", f"{avg_r2:.4f}")
    
    # 3D Plot
    st.subheader("3D Visualization")
    fig_3d = create_3d_plot(points, results, f"{dataset_choice} - 3D Point Cloud & Fitted Curves",
                           show_points, show_fitted, point_size)
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # 2D Profile
    st.subheader("2D Catenary Profile")
    fig_2d = create_2d_profile(results, f"{dataset_choice} - Catenary Profile (Projected)")
    st.plotly_chart(fig_2d, use_container_width=True)
    
    # Wire details
    st.subheader("Wire Details")
    
    for i, wire in enumerate(results.wires):
        with st.expander(f"Wire {i+1} Details", expanded=True):
            if wire.is_fitted:
                cat = wire.catenary
                params = cat.params
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Points", len(wire.points))
                c2.metric("R² Score", f"{cat.r_squared:.6f}")
                c3.metric("Curvature (c)", f"{params.c:.2f}")
                
                st.markdown("**Catenary Parameters:**")
                st.code(f"""
x₀ = {params.x0:.4f} m  (horizontal offset)
y₀ = {params.y0:.4f} m  (vertical offset)
c  = {params.c:.4f} m   (curvature parameter)

Equation: y = y₀ + c × (cosh((x - x₀) / c) - 1)
                """)
                
                # Z statistics
                z = wire.points[:, 2]
                st.markdown(f"**Height range:** {z.min():.2f}m to {z.max():.2f}m (span: {z.max()-z.min():.2f}m)")
            else:
                st.warning("Wire not fitted")

# Footer
st.markdown("---")
st.markdown("""
**Algorithm Summary:**
- **Z range > 3m** → Stacked wires (cluster by Z)
- **1100-1300 points** → Parallel wires at same height (cluster by X)  
- **Otherwise** → Single wire

*Note: The 1100-1300 point count is a dataset-specific heuristic.*
""")