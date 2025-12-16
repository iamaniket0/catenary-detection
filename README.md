# ⚡ Catenary Detection


![Bloomberg](assets/bloomberg_logo.svg)


Automatic wire detection and catenary curve fitting from LiDAR point clouds.
Required files:
- `lidar_cable_points_easy.parquet`
- `lidar_cable_points_medium.parquet`
- `lidar_cable_points_hard.parquet`
- `lidar_cable_points_extrahard.parquet`


## Results

| Dataset | Wires | R² Score |
|---------|-------|----------|
| EASY | 1 | 0.9923 |
| MEDIUM | 2 | 0.9914 |
| HARD | 1 | 0.9921 |
| EXTRAHARD | 2 | 0.8820 |

## Installation
```bash
git clone https://github.com/YOUR_USERNAME/catenary-detection.git
cd catenary-detection
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

## Quick Start
```python
from catenary_detector import CatenaryDetector

detector = CatenaryDetector()
results = detector.fit("data/lidar_cable_points_easy.parquet")
print(f"Detected {results.n_wires} wire(s)")
```

## Run Examples
```bash
# CLI
python examples/run_detection.py

# Interactive Demo
streamlit run examples/streamlit_app.py
```

## Run Tests
```bash
pytest tests/ -v
```

## Project Structure
```
catenary-detection/
├── src/catenary_detector/    # Main package
│   ├── models/               # Catenary, Plane, Wire
│   ├── clustering/           # Wire separation
│   ├── fitting/              # Curve fitting
│   └── visualization/        # Plotting
├── notebooks/                # EDA
├── examples/                 # CLI + Streamlit
├── tests/                    # Unit tests
└── data/                     # LiDAR datasets
```

## Algorithm

1. **Clustering**: Separate wires by Z-range (>3m = stacked) or point count
2. **Plane Fitting**: SVD to find best-fit plane
3. **Catenary Fitting**: scipy curve_fit with equation `y = y₀ + c(cosh((x-x₀)/c) - 1)`

## License

MIT
