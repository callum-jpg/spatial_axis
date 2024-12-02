from typing import List, Literal, Tuple

import numpy
import skimage
import scipy
import geopandas
import shapely
import warnings

class SpatialAxis:
    def __init__(
        self,
        instance_shapes: geopandas.GeoDataFrame,
        broad_annotation_shapes: geopandas.GeoDataFrame,
        broad_annotation_order: List,
        label_to_broad_annotation_method: Literal["centroid"] = "centroid",
    ):
        """
        For a series of instance_shapes (cells) in an image, find their position relative
        to another series of broad_annotation_shapes (manual/anatomical annotations).
        """

        self.instance_shapes = instance_shapes
        self.broad_annotation_shapes = broad_annotation_shapes
        self.broad_annotation_order = broad_annotation_order
        self.label_to_broad_annotation_method = label_to_broad_annotation_method

        if isinstance(self.instance_shapes, numpy.ndarray):
            # Extract regionprop centroids
            self.centroids, self.labels = self.get_label_centroids()
            self.centroid_broad_annotations = self.get_label_broad_annotation()
        # elif isinstance(self.instance_shapes, shapely.geometry.polygon.Polygon):
        elif isinstance(self.instance_shapes, geopandas.GeoDataFrame):
            # get polygon centroids
            self.centroids = (
                self.instance_shapes["geometry"]
                .apply(lambda x: self.get_shapely_centroid(x))
                .values
            )
            self.centroids = numpy.vstack(self.centroids)
            self.centroid_broad_annotations = self.instance_shapes.loc[
                :, "geometry"
            ].apply(
                lambda x: self.assign_broad_annotation(x, self.broad_annotation_shapes)
            )
        elif isinstance(self.instance_shapes, list):
            # Process a list of labelled images?
            raise NotImplementedError
        else:
            raise NotImplementedError

    def get_relative_distances(self, k_neighbours: int):
        """
        This function should take in centroids. You can then vectorize 'em.

        Associated with these cnetroids should be their broad annotation ID,
        based upon where the centroid (or some other metric) lays. One centroid,
        one broad annotation.
        """
        
        if self.centroids.shape[0] <=  k_neighbours + 1:
            raise ValueError(
                f"Can't find k_neighbours {k_neighbours} for only {len(self.centroids)} centroids. Increase number of centroids or decrease k_neighbours"
            )

        all_dist = list()

        # [1:] to skip 0, which should be the background
        for broad_annotation_class in self.broad_annotation_order:
            # Get the centroids for a specific class that's defined by
            # their broad annotation.
            # Ie. only return centroid coordinates that are inside a broad annotation

            class_centroids = self.centroids[
                numpy.where(self.centroid_broad_annotations == broad_annotation_class)
            ]

            # Create a tree
            tree = scipy.spatial.cKDTree(class_centroids)

            # Based on all centroids, query their nearest neighbours. 0 if closest
            # to themselves, otherwise the euclidean distance to the class of interest
            distances, _ = tree.query(self.centroids, k=k_neighbours)

            if not k_neighbours == 1:
                # Now we have the distances to nearest neighbours depending on the value of k
                # (first distance will be 0 if the class an item was in is queried). Now, we can
                # calculate the mean for all neighbours. Presumably, this will identify cells that
                # are distinctly within an annotation class, and then those that are "between" classes
                distances = numpy.mean(distances, axis=1)

            all_dist.append(distances)

        # Create all_dist, which will contain a column of cell distances
        # in the order defined by broad_annotation_order
        all_dist = numpy.array(all_dist).T

        # For all broad annotations, calculate the relative positioning
        relative_distance = self.compute_relative_positioning(all_dist)

        return relative_distance
    
    def get_relative_distance_labelmap(self, relative_distance: numpy.array):
        """
        Return a labelmap where instance labels are labelled with their
        relative position in an image.
        
        Used 
        """
        # Create a new label map where the cells are labelled
        # based on their position in the broad annotations
        assert isinstance(self.instance_shapes, numpy.ndarray), "Can only generate a label map if input was initially a label map"
        relative_labelmap = self.instance_shapes.copy()
        for cell_lab in self.labels:
            relative_labelmap = numpy.where(
                relative_labelmap == cell_lab,
                relative_distance[numpy.argwhere(self.labels == cell_lab)],
                relative_labelmap,
            )

        return relative_labelmap
        

    def assign_broad_annotation(self, x, lookup):
        try:
            return lookup.index[lookup["geometry"].contains(x)].values[0]
        except:
            warnings.warn("Centroid not found in broad annotation")
            return numpy.nan

    def get_shapely_centroid(self, polygon):
        """Get a shapely centroid"""
        return numpy.array(polygon.centroid.coords[0])

    def compute_relative_positioning(
        self,
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

    def get_label_centroids(self):
        """Get the centorids, cell labels, and their associated broad
        annotation ID."""

        props = skimage.measure.regionprops_table(
            self.instance_shapes, properties=["centroid", "label"]
        )

        # Pop label, leaving only centroid information in dict
        labels = props.pop("label")

        centroids = numpy.vstack([*props.values()]).T

        return centroids, labels

    # Combine this function with above? Prevent centroids class attribute double thing
    def get_label_broad_annotation(self):
        """Assign a label to a broad annotation.

        centroid: the broad_annotation where the centroid of a
        label is what's used to define it's broad_annotation class
        """

        if self.label_to_broad_annotation_method.casefold() == "centroid":
            # Indexer as a tuple to allow for N dimensional indexing of
            # self.broad_annotation_image (ie. 2D or 3D centroids)
            label_broad_annotation = self.broad_annotation_shapes[
                tuple(self.centroids.astype(int).T)
            ]
        else:
            raise NotImplementedError

        return label_broad_annotation
