import typing

import geopandas
import anndata
import numpy
import scipy
import shapely
import skimage
from rasterio.features import rasterize

from .validation import validate_input


def spatial_axis(
    data: typing.Union[anndata.AnnData, geopandas.GeoDataFrame, numpy.ndarray],
    broad_annotation_order: typing.List[int],
    k_neighbours=5,
    
    annotation_column: str = None,
    broad_annotations: typing.Union[geopandas.GeoDataFrame, numpy.ndarray] = None,
    
    missing_annotation_method: typing.Literal["none", "replace", "knn"] = "none",
    replace_value: int = 1,
    broad_annotations_to_exclude: typing.Optional[typing.Union[int, typing.List[int]]] = None,
    exclusion_value: typing.Optional[typing.Union[int, float]] = numpy.nan,
    # broad_annotation_weights, # TODO: add this
) -> numpy.ndarray:
    """Find the relative positioning of uniquely labelled objects across
    broad annotations with a specific order.

    This enables the consideration of individual cells within the wider context
    of anatomical tissue. For example, broad annotations may include the cortex
    and medulla. It's useful to know that if a cell exists in the medulla region
    of the tissue, is it closer or further away from the cortex than other
    cells? ie. what is it's relative positioning?

    Args:
        instance_objects (typing.Union[geopandas.GeoDataFrame, numpy.ndarray]):
        Individual objects that will have realtive positioning calculated.
        Either a GeoPandas DataFrame with a Geometry column containing Shapely
        polygons, or a NumPy array with instance labels. 
        broad_annotations
        (typing.Union[geopandas.GeoDataFrame, numpy.ndarray]): Contextual
        annotations that will be used to determine the relative positioning of
        instance objects. 
        broad_annotation_order (typing.List[int]): Known
        organisational order of the broad annotations. For example: cortex ->
        medulla. Annotation order values must match the values in
        broad_annotations. 
        broad_annotations_to_exclude
        (typing.Optional[typing.List[int]], optional): Broad annotations to
        exclude. Defaults to None. 
        exclusion_value
        (typing.Optional[typing.Union[int, float]], optional): If a broad
        annotation has been excluded, their relative positioning will be
        replaced with this value. Defaults to numpy.nan. 
        k_neighbours (int,
        optional): Number of K neighbours used to calculate relative positioning
        for each individual object. Higher values will give a "smoother"
        relative positioning value. Defaults to 5.

    Returns:
        numpy.ndarray: 1D numpy array containing the relative positioning values
        for each object
    """
    validate_input(data, broad_annotations)

    if isinstance(data, geopandas.GeoDataFrame):
        assert broad_annotations is not None or annotation_column is not None, """Must provide either 
    broad_annotations GeoDataFrame or an annotation column ID
    """

    shape_centroids = _get_shape_centroids(data)

    if broad_annotations is not None:
        # Broad annotations are provided. Find which centroids are inside
        # each broad annotation
        centroid_class = _get_centroid_class(shape_centroids, broad_annotations)
    elif annotation_column is not None:
        # Directly load centroid class from anndata
        centroid_class = data.obs[annotation_column].numpy()

    all_dist = _spatial_axis(
        centroids = centroids,
        centroid_class = centroid_class,
        class_order = class_order,
        k_neighbours=k_neighbours,
    )

    # How to handle NaN values
    if missing_annotation_method.casefold() == "replace":
        assert replace_value >= -1 and replace_value <= 1, "replace_value must be >= -1 and <= 1"
        all_dist = numpy.nan_to_num(all_dist, nan = replace_value)
    elif missing_annotation_method.casefold() == "knn":
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=k_neighbours)
        all_dist = imputer.fit_transform(all_dist)

    relative_distance = compute_relative_positioning(all_dist)

    if broad_annotations_to_exclude is not None:
        if isinstance(broad_annotations_to_exclude, int):
            broad_annotations_to_exclude = [broad_annotations_to_exclude]

        # We will use distances that are of a broad annotation
        # to exclude
        floored_shape_centroids = numpy.floor(shape_centroids).astype(int)
        # Create a mask for regions of the image that contain
        # values to exclude
        exclusion_mask = numpy.isin(broad_annotations, broad_annotations_to_exclude)

        # Get XY coordinates for the centroids
        x, y = floored_shape_centroids[:, 0], floored_shape_centroids[:, 1]
        # Determine which centroids to include
        inclusion_mask = ~exclusion_mask[x, y]
        # Mask the original (non-floored) centroids
        # shape_centroids = shape_centroids[inclusion_mask]
        relative_distance[~inclusion_mask] = exclusion_value

    return relative_distance

def _get_shape_centroids(data):
    if isinstance(data, anndata.AnnData):
        shape_centroids = adata.obms["spatial"]
        return shape_centroids
    elif isinstance(data, geopandas.GeoDataFrame):
        shape_centroids = (
            instance_objects["geometry"].apply(lambda x: get_shapely_centroid(x)).to_numpy()
        )
        shape_centroids = numpy.stack(shape_centroids)
        return shape_centroids
    else:
        raise ValueError(f"Shape type {type(data)} not supported.")

def _get_centroid_class(centroids, broad_annotations):
    if isinstance(broad_annotations, geopandas.GeoDataFrame):
        # Convert centroids to shapely points.
        # This enables determining where points are found in
        # the broad_annotation GeoDataFrame
        centroid_points = geopandas.GeoDataFrame(
            geometry=[shapely.Point(x, y) for x, y in centroids],
            crs=broad_annotations.crs
        )
        # Find annotation polygons that contains each centroid
        joined = geopandas.sjoin(centroid_points, broad_annotations, how="left", predicate="within")
        def select_annotation(annotations):
            unique_matches = annotations.dropna().unique()
            return unique_matches[0] if len(unique_matches) == 1 else None

        # Group by the original point index (from points_gdf) and apply the function.
        annotations_grouped = joined.groupby(joined.index)['index_right'].apply(select_annotation)

        # Ensure the result is in the same order as the original points.
        annotations_grouped = annotations_grouped.reindex(centroid_points.index)
        # Get annotation indices
        centroid_class = annotations_grouped.tolist()
        centroid_class = numpy.array(centroid_class)
    elif isinstance(broad_annotations, numpy.ndarray):
        floored_shape_centroids = numpy.floor(centroids).astype(int)
        x, y = floored_centroids[:, 0], floored_centroids[:, 1]
        centroid_class = broad_annotations[x, y]
    else:
        ValueError(
            f"Do not support broad_annotations of type {type(broad_annotations)}"
        )

    return centroid_class


def _spatial_axis(
    shape_centroids,
    centroid_annotation,
    ordered_annotations,
    k_neighbours,
):

    # List to contain sample distance information
    all_dist = []

    # Iterate over each broad annotation class and create a
    # cKDTree for each group of centroids that are within
    # that class
    for broad_annotation_class in ordered_annotations:
        class_centroids = shape_centroids[
            numpy.where(centroid_annotation == broad_annotation_class)
        ]

        # Ensure the class has centroids
        if class_centroids.shape[0] != 0:
            # If there are less then k_neighbours in a class group, set k_neighbours to this value. Otherwise,
            # tree.query will return inf, since, for example, a group of 3 points cannot have 5 neighbours.
            tree_k_neighbours = (
                k_neighbours
                if class_centroids.shape[0] > k_neighbours
                else class_centroids.shape[0]
            )

            # Create a tree
            tree = scipy.spatial.cKDTree(class_centroids)

            # Based on all centroids, query their nearest neighbours. 0 if closest
            # to themselves, otherwise the euclidean distance to the class of interest
            distances, _ = tree.query(shape_centroids, k=tree_k_neighbours)

            if not tree_k_neighbours == 1:
                # Now we have the distances to nearest neighbours depending on the value of k
                # (first distance will be 0 if the class an item was in is queried). Now, we can
                # calculate the mean for all neighbours. Presumably, this will identify cells that
                # are distinctly within an annotation class, and then those that are "between" classes
                distances = numpy.mean(distances, axis=1)
            all_dist.append(distances)
        # No centroids found for this class, so distances are NaN.
        else:
            
            distances = numpy.empty(len(shape_centroids))
            distances[:] = numpy.nan
            all_dist.append(distances)

    all_dist = numpy.array(all_dist).T

    return all_dist


def get_shapely_centroid(polygon: shapely.Polygon) -> numpy.ndarray:
    """Get a Shapely polygon centroid coordinates.

    Args:
        polygon (shapely.Polygon): Shapely polygon.

    Returns:
        numpy.ndarray: Coordinates of a polygon centroid.
    """
    return numpy.array(polygon.centroid.coords[0])


def compute_relative_positioning(
    distances: numpy.ndarray,
) -> numpy.ndarray:
    """
    This function is intended to normalize the distances between N number of
    classes.

    eg. distances of cells to their nearest annotation neighbour. For all of
    these distances, we can compute a relative positioning "score" as follows:

    x1: distnace to annotation 1 x2: distance to annotation 2

    (x1 - x2) / (x1 + x2)

    Annotation 1...N is defined in a heirarchical scanning window, so only those
    spatially adjacent annotation classes are considered.

    eg. classes [1, 2, 3, 4] would be considered in scanning window pairs [1,
    2], [2, 3], [3, 4].

    Args:
        distances (numpy.ndarray): Distances computed with cKDTree for each cell
        across all broad annotations

    Returns:
        numpy.ndarray: Normalised distances across broad annotation classes.
    """

    inter_class_distances = []
    # Iterate over .shape[1] (the columns), which corresponds to the number of
    # broad annotation classes
    for col_idx in numpy.arange(distances.shape[1] - 1):
        if numpy.isnan(distances[:, col_idx]).any():
            a = numpy.empty(distances[:, col_idx].shape)
            a[:] = 1

            # a = (distances[:, col_idx] - distances[:, col_idx + 1]) / (
            #     distances[:, col_idx] + distances[:, col_idx + 1]
            # )
        else:
            a = (distances[:, col_idx] - distances[:, col_idx + 1]) / (
                distances[:, col_idx] + distances[:, col_idx + 1]
            )
        inter_class_distances.append(a)

    inter_class_distances = numpy.array(numpy.nansum(inter_class_distances, axis=0))

    return inter_class_distances


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


def get_label_centroids(
    instance_shapes: numpy.ndarray,
) -> typing.Union[numpy.ndarray, list]:
    """
    Get the centorids, cell labels, and their associated broad annotation ID.

    Args:
        instance_shapes (numpy.ndarray): Array containing instance labels.

    Returns:
        typing.Union[numpy.ndarray, list]: Returns a NumPy array of centroid
        coordinates and a list of label values.
    """
    # Get centroid and label values.
    props = skimage.measure.regionprops_table(
        instance_shapes, properties=["centroid", "label"]
    )

    # Pop label, leaving only centroid information in dict
    labels = props.pop("label")

    centroids = numpy.vstack([*props.values()]).T

    return centroids, labels
