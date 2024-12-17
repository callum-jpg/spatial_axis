import typing

import geopandas
import numpy
import scipy
import shapely
import skimage
from rasterio.features import rasterize


def spatial_axis(
    instance_objects: typing.Union[geopandas.GeoDataFrame, numpy.ndarray],
    broad_annotations: typing.Union[geopandas.GeoDataFrame, numpy.ndarray],
    broad_annotation_order: typing.List[int],
    broad_annotations_to_exclude: typing.Optional[
        typing.Union[int, typing.List[int]]
    ] = None,
    exclusion_value: typing.Optional[typing.Union[int, float]] = numpy.nan,
    k_neighbours=5,
    # broad_annotation_weights,
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
        polygons, or a NumPy array with instance labels. broad_annotations
        (typing.Union[geopandas.GeoDataFrame, numpy.ndarray]): Contextual
        annotations that will be used to determine the relative positioning of
        instance objects. broad_annotation_order (typing.List[int]): Known
        organisational order of the broad annotations. For example: cortex ->
        medulla. Annotation order values must match the values in
        broad_annotations. broad_annotations_to_exclude
        (typing.Optional[typing.List[int]], optional): Broad annotations to
        exclude. Defaults to None. exclusion_value
        (typing.Optional[typing.Union[int, float]], optional): If a broad
        annotation has been excluded, their relative positioning will be
        replaced with this value. Defaults to numpy.nan. k_neighbours (int,
        optional): Number of K neighbours used to calculate relative positioning
        for each individual object. Higher values will give a "smoother"
        relative positioning value. Defaults to 5.

    Returns:
        numpy.ndarray: 1D numpy array containing the relative positioning values
        for each object
    """
    all_dist = []

    if not isinstance(instance_objects, list):
        instance_objects = [instance_objects]

    if not isinstance(broad_annotations, list):
        broad_annotations = [broad_annotations]

    for io, ba in zip(instance_objects, broad_annotations):
        # Get centroids for each polygon
        shape_centroids = (
            io["geometry"].apply(lambda x: get_shapely_centroid(x)).to_numpy()
        )
        shape_centroids = numpy.stack(shape_centroids)

        if isinstance(ba, geopandas.GeoDataFrame):
            # Convert centroids to shapely points.
            # This enables determining where points are found in
            # the broad_annotation GeoDataFrame
            centroid_broad_annotation_class = []
            # TODO: vectorize this
            shape_centroid_points = [shapely.Point(x, y) for x, y in shape_centroids]
            for shp_ctrd in shape_centroid_points:
                centroid_broad_annotation_class.append(
                    ba.index[ba["geometry"].contains(shp_ctrd)].values[0]
                )
            centroid_broad_annotation_class = numpy.array(
                centroid_broad_annotation_class
            )
        elif isinstance(ba, numpy.ndarray):
            floored_shape_centroids = numpy.floor(shape_centroids).astype(int)
            x, y = floored_shape_centroids[:, 0], floored_shape_centroids[:, 1]
            centroid_broad_annotation_class = ba[x, y]
        else:
            ValueError(f"Do not support broad_annotations of type {type(ba)}")

        # List to contain sample distance information
        loop_dist = []

        # Iterate over each broad annotation class and create a
        # cKDTree for each group of centroids that are within
        # that class
        for broad_annotation_class in broad_annotation_order:
            class_centroids = shape_centroids[
                numpy.where(centroid_broad_annotation_class == broad_annotation_class)
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
                loop_dist.append(distances)
            else:
                distances = numpy.empty(len(shape_centroids))
                distances[:] = numpy.nan
                loop_dist.append(distances)
        loop_dist = numpy.array(loop_dist).T
        all_dist.append(loop_dist)

    all_dist = numpy.vstack(all_dist)

    relative_distance = compute_relative_positioning(all_dist)

    if broad_annotations_to_exclude is not None:
        if isinstance(broad_annotations_to_exclude, int):
            broad_annotations_to_exclude = [broad_annotations_to_exclude]

        inclusion_mask = []

        for io, ba in zip(instance_objects, broad_annotations):
            shape_centroids = [
                centroid
                for centroid in io["geometry"]
                .apply(lambda x: get_shapely_centroid(x))
                .to_numpy()
            ]
            shape_centroids = numpy.array(shape_centroids)

            # We will use distances that are of a broad annotation
            # to exclude
            floored_shape_centroids = numpy.floor(shape_centroids).astype(int)
            # Create a mask for regions of the image that contain
            # values to exclude
            exclusion_mask = numpy.isin(ba, broad_annotations_to_exclude)

            # Get XY coordinates for the centroids
            x, y = floored_shape_centroids[:, 0], floored_shape_centroids[:, 1]
            # Determine which centroids to include
            inclusion_array = ~exclusion_mask[x, y]

            inclusion_mask.append(inclusion_array)

        inclusion_mask = numpy.hstack(inclusion_mask)

        # Mask the original (non-floored) centroids
        # shape_centroids = shape_centroids[inclusion_mask]
        relative_distance[~inclusion_mask] = exclusion_value

    return relative_distance


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
        a = normalised_difference(distances[:, col_idx], distances[:, col_idx + 1])
        inter_class_distances.append(a)

    inter_class_distances = numpy.array(inter_class_distances)

    inter_class_distances = numpy.nansum(inter_class_distances, axis=0)

    return inter_class_distances


def normalise_min_max(data, epsilon=1e-10):
    """Min-max normalisation to scale data to [-1, 1]."""
    data_min = numpy.nanmin(data)
    data_max = numpy.nanmax(data)
    range_ = max(data_max - data_min, epsilon)
    return 2 * ((data - data_min) / range_ - 0.5)


def normalised_difference(
    array1, array2, norm_method: typing.Literal["minmax"] = "minmax"
):
    if norm_method.casefold() == "minmax":
        array1 = normalise_min_max(array1)
        array2 = normalise_min_max(array2)

    # Ignore NaN values during summation
    sum_array1 = numpy.nansum(array1)
    sum_array2 = numpy.nansum(array2)

    # Compute the denominator (global scale normalization)
    denominator = abs(sum_array1 + sum_array2)

    # Calculate the normalized difference for each element
    result = (array1 - array2) / denominator

    # Handle NaN values in the input arrays
    result = numpy.where(numpy.isnan(array1) | numpy.isnan(array2), numpy.nan, result)

    # result = 2 * (result - numpy.nanmin(result)) / (numpy.nanmax(result) - numpy.nanmin(result)) - 1
    result = numpy.multiply(result, 100)

    return result


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
