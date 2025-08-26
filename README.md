# Spatial Axis

Define relative positioning of cells along a spatial axis in spatial biology data.

## Overview

Spatial Axis is a Python package for analyzing the relative positioning of cells along anatomical or spatial axes. It's particularly useful for spatial biology applications where you need to understand how cells are distributed along defined gradients or anatomical structures.

The package allows you to:
- Define spatial axes using anatomical annotations
- Calculate relative positions of cells along these axes
- Handle various data formats (AnnData, GeoDataFrame, numpy arrays)
- Visualize spatial relationships

## Installation from source
```bash
git clone https://github.com/callum-jpg/spatial_axis.git
cd spatial_axis
pip install -e .
```

## Quick Start

```python
import numpy as np
import geopandas as gpd
from spatial_axis import spatial_axis

# Create example cell data
cells = gpd.GeoDataFrame(...)  # Your cell geometries

# Define anatomical annotations
annotations = gpd.GeoDataFrame(...)  # Your anatomical regions

# Calculate spatial axis
annotation_order = [0, 1, 2, 3, 4]  # Order along the axis
cells['spatial_axis'] = spatial_axis(
    cells, 
    annotations, 
    annotation_order, 
    k_neighbours=5
)
```

## Key Features

### Multiple Data Format Support
- **AnnData**: Integration with scanpy/squidpy workflows
- **GeoDataFrame**: Direct spatial data analysis
- **NumPy arrays**: Label-based analysis

### Flexible Annotation Methods
- Polygon-based anatomical regions
- Label array-based annotations
- Exclusion of specific regions
- Missing data handling

### Robust Analysis
- K-nearest neighbors for spatial smoothing
- Configurable weighting schemes
- Validation and preprocessing

## Examples

### Basic Usage with GeoDataFrame

```python
from spatial_axis import spatial_axis
from spatial_axis.utility import random_shapely_circles, create_broad_annotation_polygons
import geopandas as gpd
import numpy as np

# Generate example data
IMG_SHAPE = (256, 256)
cells = random_shapely_circles(IMG_SHAPE, num_circles=200, seed=42)
cells_gdf = gpd.GeoDataFrame(geometry=cells)

# Create anatomical annotations
annotations = create_broad_annotation_polygons(
    IMG_SHAPE, 
    annotation_shape="circle", 
    num_levels=5
)
annotations_gdf = gpd.GeoDataFrame(geometry=annotations)

# Calculate spatial axis
annotation_order = np.arange(5)
cells_gdf['spatial_axis'] = spatial_axis(
    cells_gdf, 
    annotations_gdf, 
    annotation_order, 
    k_neighbours=5
)
```

### Using Label Arrays

```python
# Use rasterized label arrays instead of polygons
from rasterio.features import rasterize

labeled_annotations = rasterize(
    [(poly, idx) for idx, poly in enumerate(annotations)],
    out_shape=IMG_SHAPE
)

cells_gdf['spatial_axis'] = spatial_axis(
    cells_gdf,
    labeled_annotations,
    annotation_order,
    k_neighbours=5
)
```

### Integration with AnnData

```python
import anndata as ad

# Load your spatial data
adata = ad.read_h5ad("spatial_data.h5ad")

# Calculate spatial axis
adata.obs['spatial_axis'] = spatial_axis(
    adata,
    annotations,
    annotation_order,
    annotation_column='region_id'
)
```

## API Reference

## Advanced Usage

### Excluding Regions

```python
# Exclude specific annotation regions
cells['spatial_axis'] = spatial_axis(
    cells,
    annotations,
    annotation_order,
    broad_annotations_to_exclude=[-1, 999]  # Exclude background/artifact regions
)
```

### Custom Weighting

```python
# Apply custom weights to annotations
weights = [1.0, 1.5, 2.0, 1.5, 1.0]  # Emphasize central regions
cells['spatial_axis'] = spatial_axis(
    cells,
    annotations,
    annotation_order,
    weights=weights
)
```

### Missing Data Handling

```python
# Handle cells not assigned to any annotation
cells['spatial_axis'] = spatial_axis(
    cells,
    annotations,
    annotation_order,
    missing_annotation_method="knn",  # Use k-nearest neighbors
    k_neighbours=10
)
```

### Development Setup

```bash
git clone https://github.com/callum-jpg/spatial_axis.git
cd spatial_axis
pip install -e ".[dev,test]"
```

### Running Tests

```bash
pytest tests/
```

### Code Style

We use `black`, `isort`, and `ruff` for code formatting:

```bash
black src/
isort src/
ruff check src/
```