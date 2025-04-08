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
    """
    Convert a GeoPandas DataFrame containing a spatial axis column of relative
    object positionings into a label map for plotting.

    Args:
        shape_gdf (geopandas.GeoDataFrame): GeoDataFrame containing a geometry
        and spatial_axis column image_shape (typing.Tuple[int, int]): Shape of
        image to rasterize polygons onto. geometry_column (str, optional): Name
        of column containing Shapely polygons. Defaults to "geometry".
        spatial_axis_column (str, optional): Name of column containing relative
        positioning spatial axis values. Defaults to "spatial_axis".
        background_value (int, optional): Background value. Defaults to 0.

    Returns:
        numpy.ndarray: Rasterized labelmap with objects colored based upon their
        relative positioning.
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
