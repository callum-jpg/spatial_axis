import typing
import warnings

import geopandas
import numpy
import scipy
import skimage
from rasterio.features import rasterize


def spatial_axis(
    instance_polygons: geopandas.GeoDataFrame,
    broad_annotations: geopandas.GeoDataFrame,
    broad_annotation_order,
    k_neighbours=5,
    # broad_annotation_weights,
):
    # Get centroids for each polygon
    shape_centroids = (
        instance_polygons["geometry"]
        .apply(lambda x: get_shapely_centroid(x))
        .to_numpy()
    )
    shape_centroids = numpy.stack(shape_centroids)

    # For each polygon centroid, find which broad annotation
    # it is contained within
    centroid_broad_annotation_class = (
        instance_polygons.loc[:, "geometry"]
        .apply(lambda x: assign_broad_annotation(x.centroid, broad_annotations))
        .values
    )

    # List to contain sample distance information
    all_dist = []

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

            all_dist.append(distances)

    all_dist = numpy.array(all_dist).T

    relative_distance = compute_relative_positioning(all_dist)

    return relative_distance


def assign_broad_annotation(x, lookup):
    try:
        return lookup.index[lookup["geometry"].contains(x)].values[0]
    except IndexError:
        warnings.warn(f"Centroid {x} not found in broad annotation")
        return numpy.nan


def get_shapely_centroid(polygon):
    """Get a shapely centroid"""
    return numpy.array(polygon.centroid.coords[0])


def compute_relative_positioning(
    distances,
):
    """
    This function is intended to normalize the distances between N number of classes.

    eg. distances of cells to their nearest annotation neighbour.
    For all of these distances, we can compute a relative positioning
    "score" as follows:

    x1: distnace to annotation 1
    x2: distance to annotation 2

    (x1 - x2) / (x1 + x2)

    Annotation 1...N is defined in a heirarchical scanning window,
    so only those spatially adjacent annotation classes are considered.

    eg. classes [1, 2, 3, 4] would be considered in scanning window pairs
    [1, 2], [2, 3], [3, 4].
    """

    inter_class_distances = []
    # Iterate over .shape[1] (the columns), which corresponds to the number of
    # broad annotation classes
    for col_idx in numpy.arange(distances.shape[1] - 1):
        a = (distances[:, col_idx] - distances[:, col_idx + 1]) / (
            distances[:, col_idx] + distances[:, col_idx + 1]
        )
        inter_class_distances.append(a)

    inter_class_distances = numpy.array(sum(inter_class_distances))

    return inter_class_distances


def spatial_axis_to_labelmap(
    shape_gdf: geopandas.GeoDataFrame,
    image_shape: typing.Tuple[int, int],
    geometry_column: str = "geometry",
    spatial_axis_column: str = "spatial_axis",
    background_value: int = 0,
):

    spatial_axis_polygons = shape_gdf.apply(
        lambda row: (row[geometry_column], row[spatial_axis_column]), axis=1
    ).tolist()

    rasterized_polygons = rasterize(
        [(poly, label) for poly, label in spatial_axis_polygons],
        out_shape=image_shape,
        fill=background_value,
    )

    return rasterized_polygons


def get_label_centroids(instance_shapes):
    """Get the centorids, cell labels, and their associated broad
    annotation ID."""

    props = skimage.measure.regionprops_table(
        instance_shapes, properties=["centroid", "label"]
    )

    # Pop label, leaving only centroid information in dict
    labels = props.pop("label")

    centroids = numpy.vstack([*props.values()]).T

    return centroids, labels
