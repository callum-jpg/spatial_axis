import typing

import geopandas
import numpy
from rasterio.features import rasterize


def spatial_axis_to_labelmap(
    shape_gdf: geopandas.GeoDataFrame,
    image_shape: typing.Tuple[int, int],
    geometry_column: str = "geometry",
    spatial_axis_column: str = "spatial_axis",
    background_value: int = 0,
) -> numpy.ndarray:
    """Convert spatial axis values to a rasterized label map for visualization.
    
    This function takes a GeoDataFrame with spatial axis values and converts it 
    to a rasterized array where each pixel is colored according to the spatial 
    axis value of the overlapping geometry.

    Parameters
    ----------
    shape_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing geometries and spatial axis values.
        Must contain columns specified by geometry_column and spatial_axis_column.
    image_shape : Tuple[int, int]
        Output image dimensions as (height, width) for rasterization.
    geometry_column : str, default="geometry"
        Name of the column containing Shapely geometry objects.
    spatial_axis_column : str, default="spatial_axis"
        Name of the column containing spatial axis values to use for labeling.
    background_value : int, default=0
        Value to assign to pixels not covered by any geometry.

    Returns
    -------
    numpy.ndarray
        2D array with shape `image_shape` where each pixel contains either:
        - The spatial_axis value of the overlapping geometry
        - `background_value` if no geometry overlaps that pixel

    Examples
    --------
    Convert spatial axis results to a visualization-ready array:
    
    >>> import geopandas as gpd
    >>> from spatial_axis import spatial_axis_to_labelmap
    >>> 
    >>> # Assume cells_gdf has spatial_axis values computed
    >>> label_map = spatial_axis_to_labelmap(
    ...     cells_gdf, 
    ...     image_shape=(512, 512),
    ...     background_value=-1
    ... )
    >>> 
    >>> # Visualize with matplotlib
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(label_map, cmap='viridis')
    >>> plt.colorbar(label='Spatial Axis Position')

    Notes
    -----
    The resulting array can be directly used with matplotlib's imshow() 
    for visualization. The spatial axis values provide a continuous 
    color scale representing relative position along the defined axis.
    """

    spatial_axis_polygons = shape_gdf.apply(
        lambda row: (row[geometry_column], row[spatial_axis_column]), axis=1
    ).tolist()

    rasterized_polygons = rasterize(
        [(poly, label) for poly, label in spatial_axis_polygons],
        out_shape=image_shape,
        fill=background_value,
    )

    return rasterized_polygons
