import logging
import typing

import anndata
import geopandas
import numpy
import scipy
import shapely
import skimage

from .constants import SpatialAxisConstants
from .validation import validate_input
from .preprocessing import spatial_celltype_filter

log = logging.getLogger(__name__)


def spatial_axis(
    data: typing.Union[anndata.AnnData, geopandas.GeoDataFrame, numpy.ndarray],
    annotation_order: typing.List[int],
    k_neighbours=5,
    annotation_column: str = None,
    broad_annotations: typing.Union[geopandas.GeoDataFrame, numpy.ndarray] = None,
    missing_annotation_method: typing.Literal[None, "replace", "knn"] = None,
    replace_value: int = 1,
    class_to_exclude: (
        typing.Optional[typing.Union[int, typing.List[int]]] | None
    ) = None,
    auxiliary_class: str = None,
    weights: numpy.ndarray | list[float] | None = None,
    exclusion_value: typing.Optional[typing.Union[int, float, numpy.nan]] = numpy.nan,
    normalise: bool = True,
    reference_cell_type: str = None,
    distance_threshold: float = None,
    distance_k_neighbors: int = None,
    scaling_factor: float = None,
    min_counts: int = 50,
    # broad_annotation_weights, # TODO: add this
) -> numpy.ndarray:
    """Calculate relative positioning of cells along a spatial axis.

    This function defines the relative positioning of uniquely labeled objects 
    across broad anatomical annotations with a specific order, enabling analysis 
    of individual cells within the wider context of tissue anatomy.

    Parameters
    ----------
    data : Union[anndata.AnnData, geopandas.GeoDataFrame, numpy.ndarray]
        Cell data containing spatial information. Can be:
        - AnnData object with spatial coordinates
        - GeoDataFrame with cell geometries
        - NumPy array with coordinates
    annotation_order : List[int]
        List defining the order of annotations along the spatial axis.
        For example, [0, 1, 2, 3, 4] defines progression from annotation 0 to 4.
    k_neighbours : int, default=5
        Number of nearest neighbors to consider for spatial smoothing.
    annotation_column : str, optional
        Column name in AnnData.obs containing annotation labels.
        Required if using AnnData without broad_annotations.
    broad_annotations : Union[geopandas.GeoDataFrame, numpy.ndarray], optional
        Anatomical annotation data. Can be:
        - GeoDataFrame with annotation polygons
        - NumPy array with annotation labels
    missing_annotation_method : {"replace", "knn", None}, optional
        Method to handle cells without annotation assignments:
        - "replace": Replace with replace_value
        - "knn": Use k-nearest neighbors
        - None: Exclude from analysis
    replace_value : int, default=1
        Value to use when missing_annotation_method="replace".
    class_to_exclude : Union[int, List[int]], optional
        Annotation class(es) to exclude from analysis.
    auxiliary_class : str, optional
        Additional cell type column for filtering.
    weights : Union[numpy.ndarray, list[float]], optional
        Custom weights for each annotation in annotation_order.
    exclusion_value : Union[int, float, numpy.nan], default=numpy.nan
        Value assigned to excluded cells.
    normalise : bool, default=True
        Whether to normalize the spatial axis values.
    reference_cell_type : str, optional
        Reference cell type for spatial filtering.
    distance_threshold : float, optional
        Distance threshold for spatial filtering.
    distance_k_neighbors : int, optional
        If provided, only keep query cells if they have at least distance_k_neighbors
        reference cells within the distance_threshold. If None, at least 1 neighbor 
        within threshold is required to keep the query cell.

    Returns
    -------
    numpy.ndarray
        Array of spatial axis values for each cell, indicating relative 
        position along the defined axis. Values closer to -1 indicate 
        proximity to early annotations in the order, while values closer 
        to +1 indicate proximity to later annotations.

    Examples
    --------
    Calculate spatial axis for cells with anatomical annotations:

    >>> import geopandas as gpd
    >>> import numpy as np
    >>> from spatial_axis import spatial_axis
    >>> 
    >>> # Create example data
    >>> cells = gpd.GeoDataFrame(...)  # Cell geometries
    >>> annotations = gpd.GeoDataFrame(...)  # Anatomical regions
    >>> annotation_order = [0, 1, 2, 3, 4]  # Cortex to medulla progression
    >>> 
    >>> # Calculate spatial axis
    >>> spatial_values = spatial_axis(
    ...     cells, annotation_order, 
    ...     broad_annotations=annotations,
    ...     k_neighbours=5
    ... )

    Using AnnData with annotation column:

    >>> import anndata as ad
    >>> adata = ad.read_h5ad("spatial_data.h5ad")
    >>> spatial_values = spatial_axis(
    ...     adata, [0, 1, 2], 
    ...     annotation_column="region_id",
    ...     k_neighbours=10
    ... )

    Notes
    -----
    For single annotation (len(annotation_order) == 1), only distance 
    to that annotation is calculated rather than relative positioning.

    The spatial axis represents a continuous measure of position along 
    anatomical gradients, useful for studying developmental processes, 
    tissue organization, and spatial gene expression patterns.
    """

    validate_input(data, broad_annotations)

    if len(annotation_order) == 1:
        log.warn(
            f"Annotation order is of length {len(annotation_order)}. Only the distance to the annotation will be calculated."
        )

    if isinstance(data, geopandas.GeoDataFrame):
        assert (
            broad_annotations is not None or annotation_column is not None
        ), """Must provide either 
    broad_annotations GeoDataFrame or an annotation column ID
    """

    if isinstance(data, anndata.AnnData) and min_counts is not None:
        import scanpy
        log.info(f"Filtering cells with less than {min_counts} transcripts.")
        scanpy.pp.filter_cells(data, min_counts=min_counts)

    if broad_annotations is not None and annotation_column is not None:
        raise ValueError(
            "Got values for both broad_annotations and annotation_column. Provide only one."
        )

    centroids = _get_centroids(data)

    if scaling_factor is not None:
        centroids = centroids * scaling_factor

    if broad_annotations is not None:
        # Broad annotations are provided. Find which centroids are inside
        # each broad annotation and assign this as the centroid class
        centroid_class = _get_centroid_class(centroids, broad_annotations)
    elif annotation_column is not None:
        if reference_cell_type is not None or distance_threshold is not None:
            assert len(annotation_order) == 1, f"Cell filtering only supports annotation_order of len 1. Got {len(annotation_order)}"
            
            annotation_column = spatial_celltype_filter(
                adata=data,
                celltype_col=annotation_column,
                query_cell=annotation_order[0],
                reference_cell = reference_cell_type or annotation_order[0],
                new_celltype = annotation_order[0],
                distance_threshold = distance_threshold,
                distance_k_neighbors = distance_k_neighbors,
            )
        # Directly load centroid class from anndata
        centroid_class = data.obs[annotation_column].to_numpy()

    assert len(list(set(annotation_order).intersection(set(centroid_class)))) > 0, "No elements of annotation_order were found in cell classes."

    assert (
        centroid_class is not None
    ), "Cannot define centroid class. Ensure you provide either broad_annotations or annotation_column."

    all_dist = _spatial_axis(
        centroids=centroids,
        centroid_class=centroid_class,
        class_order=annotation_order,
        k_neighbours=k_neighbours,
        auxiliary_class=auxiliary_class,
    )

    if missing_annotation_method is not None:
        # Handle NaN values
        if missing_annotation_method.casefold() == "replace":
            assert (
                replace_value >= -1 and replace_value <= 1
            ), "replace_value must be >= -1 and <= 1"
            all_dist = numpy.nan_to_num(all_dist, nan=replace_value)
        elif missing_annotation_method.casefold() == "knn":
            from sklearn.impute import KNNImputer

            imputer = KNNImputer(n_neighbors=k_neighbours)
            all_dist = imputer.fit_transform(all_dist)

    relative_distance = compute_relative_positioning(all_dist, weights=weights, normalise=normalise)

    if class_to_exclude is not None:
        relative_distance = _class_exclusion(
            distances=relative_distance,
            centroid_class=centroid_class,
            class_to_exclude=class_to_exclude,
            exclusion_value=exclusion_value,
        )

    return relative_distance


def _class_exclusion(
    distances,
    centroid_class,
    class_to_exclude,
    exclusion_value,
):
    if isinstance(class_to_exclude, int):
        class_to_exclude = [class_to_exclude]

    exclusion_mask = numpy.isin(centroid_class, class_to_exclude)
    distances[exclusion_mask] = exclusion_value

    return distances


def _get_centroids(data):
    if isinstance(data, anndata.AnnData):
        shape_centroids = data.obsm[SpatialAxisConstants.spatial_key]
        return shape_centroids
    elif isinstance(data, geopandas.GeoDataFrame):
        shape_centroids = (
            data[SpatialAxisConstants.geometry_key]
            .apply(lambda x: get_shapely_centroid(x))
            .to_numpy()
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
            crs=broad_annotations.crs,
        )
        # Find annotation polygons that contains each centroid
        joined = geopandas.sjoin(
            centroid_points, broad_annotations, how="left", predicate="within"
        )

        def select_annotation(annotations):
            unique_matches = annotations.dropna().unique()
            return unique_matches[0] if len(unique_matches) == 1 else None

        # Group by the original point index (from points_gdf) and apply the function.
        annotations_grouped = joined.groupby(joined.index)["index_right"].apply(
            select_annotation
        )

        # Ensure the result is in the same order as the original points.
        annotations_grouped = annotations_grouped.reindex(centroid_points.index)
        # Get annotation indices
        centroid_class = annotations_grouped.tolist()
        centroid_class = numpy.array(centroid_class)
    elif isinstance(broad_annotations, numpy.ndarray):
        floored_shape_centroids = numpy.floor(centroids).astype(int)
        x, y = floored_shape_centroids[:, 0], floored_shape_centroids[:, 1]
        centroid_class = broad_annotations[x, y]
    else:
        ValueError(
            f"Do not support broad_annotations of type {type(broad_annotations)}"
        )

    return centroid_class


def _spatial_axis(
    centroids,
    centroid_class,
    class_order,
    k_neighbours,
    auxiliary_class=None,
):
    # List to contain sample distance information
    all_dist = []

    if auxiliary_class is not None:
        auxiliary_centroids = centroids[numpy.where(centroid_class == auxiliary_class)]
        if len(auxiliary_centroids) == 0:
            log.warning(
                "No auxillary class centroids found. Calculating spatial_axis without auxiliary_class"
            )
            auxiliary_class = None

    # Iterate over each broad annotation class and create a
    # cKDTree for each group of centroids that are within
    # that class
    for class_id in class_order:
        class_centroids = centroids[numpy.where(centroid_class == class_id)]

        # Ensure the class has centroids
        if class_centroids.shape[0] != 0:
            # If there are less then k_neighbours in a class group, set k_neighbours to this value. Otherwise,
            # tree.query will return inf, since, for example, a group of 3 points cannot have 5 neighbours.
            tree_k_neighbours = (
                k_neighbours
                if class_centroids.shape[0] > k_neighbours
                else class_centroids.shape[0]
            )

            # Find the distance to the current class and to the auxillary structure
            # Mean aggregate these distances
            if auxiliary_class is not None:
                # Get the distance of all centroids to the auxiliary class
                distance_to_aux = get_centroid_distances(
                    centroids, auxiliary_centroids, tree_k_neighbours
                )
                # Get the distance of all centroids to the current centroid class iteration
                centroid_distances = get_centroid_distances(
                    centroids, class_centroids, tree_k_neighbours
                )
                # Take the mean of the distances to both classes
                distances = numpy.mean([centroid_distances, distance_to_aux], axis=0)
            else:
                distances = get_centroid_distances(
                    centroids, class_centroids, tree_k_neighbours
                )

            all_dist.append(distances)
        # No centroids found for this class, so distances are NaN.
        else:
            distances = numpy.empty(len(centroids))
            distances[:] = numpy.nan
            all_dist.append(distances)

    all_dist = numpy.array(all_dist).T

    return all_dist


def get_centroid_distances(centroids, class_centroids, tree_k_neighbours):
    # Create a tree
    tree = scipy.spatial.cKDTree(class_centroids)

    # Based on all centroids, query their nearest neighbours. 0 if closest
    # to themselves, otherwise the euclidean distance to the class of interest
    distances, _ = tree.query(centroids, k=tree_k_neighbours)

    if not tree_k_neighbours == 1:
        # Now we have the distances to nearest neighbours depending on the value of k
        # (first distance will be 0 if the class an item was in is queried). Now, we can
        # calculate the mean for all neighbours. Presumably, this will identify cells that
        # are distinctly within an annotation class, and then those that are "between" classes
        distances = numpy.mean(distances, axis=1)

    return distances


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
    weights: numpy.ndarray | None = None,
    eps: float = 1e-6,
    normalise: bool = False,
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
        normalise (bool): If True and n_classes == 1, min-max normalise. Otherwise, 
            don't.

    Returns:
        numpy.ndarray: Normalised distances across broad annotation classes.
    """

    # Weight matrix as n_class-1 x n_cells
    _weights = numpy.ones((distances.shape[1] - 1, distances.shape[0]))

    if weights is not None:
        if isinstance(weights, list):
            weights = numpy.array(weights)
        assert weights.ndim == 1, "weights should be 1-dimensional."
        assert (
            len(weights) == distances.shape[1] - 1
        ), f"weights should be the `num_annotations - 1` (got {distances.shape[1] - 1} annotations and {len(weights)} weights)."

        _weights = _weights * weights[..., numpy.newaxis]

    # Only a single class has been used, so just return the distance to this class.
    if distances.shape[1] == 1:
        distances = distances[..., 0]
        if normalise:
            # Min-max scale data
            distances = (distances - distances.min()) / (distances.max() - distances.min())
        else:
            log.info(f"Only a single discrete annotation was provided, so the euclidean distance this annotation will be returned.")
        return distances

    inter_class_distances = []
    # Iterate over .shape[1] (the columns), which corresponds to the number of
    # broad annotation classes
    for col_idx in numpy.arange(distances.shape[1] - 1):
        if numpy.isnan(distances[:, col_idx]).any():
            # A distance may be nan due to a missing annotation
            # In this scenario, we set the relative distance to 1.
            # This therefore asumes that the missing annotation is
            # close/inside its neighbour (col_idx + 1).
            # This is an assumption, but offers as a form of imputation
            # for missing data.
            relative_dist = numpy.empty(distances[:, col_idx].shape)
            relative_dist[:] = 1
        else:
            difference = distances[:, col_idx] - distances[:, col_idx + 1]
            summation = distances[:, col_idx] + distances[:, col_idx + 1]

            # Compute the normalised difference
            relative_dist = numpy.divide(
                difference,
                summation,
                # Avoid zero division error
                where=(difference != 0) | (summation != 0),
                # Keep values as 0 where the condition is met
                out=numpy.zeros_like(difference, dtype=float),
            )

        inter_class_distances.append(relative_dist)

    inter_class_distances = numpy.array(inter_class_distances)

    inter_class_distances = inter_class_distances * _weights

    inter_class_distances = numpy.nansum(inter_class_distances, axis=0)

    return inter_class_distances


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
