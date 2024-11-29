from spatial_axis import SpatialAxis
import numpy
import scipy
import shapely
import geopandas


def random_shapely_circles(
    image_shape, num_circles, min_radius=15, max_radius=35, seed=None
):
    """Generate a list of randomly sized shapely Polygons"""
    circles = []

    numpy.random.seed(seed=seed)

    for _ in range(num_circles):
        x_center = numpy.random.uniform(0, image_shape[1])
        y_center = numpy.random.uniform(0, image_shape[0])
        radius = numpy.random.uniform(min_radius, max_radius)

        circle = shapely.geometry.Point(x_center, y_center).buffer(radius)

        circles.append(circle)

    return circles


def create_broad_annotation_polygons(image_shape: tuple):
    """Generate all-encompassing (ie. cover the whole image) polygons
    with smaller and smaller sizes. Used as a type of broad human annotation"""

    minx, miny, maxx, maxy = 0, 0, image_shape[1], image_shape[0]

    outer = shapely.geometry.box(minx, miny, maxx, maxy)

    middle = shapely.affinity.scale(outer, 0.8, 0.8)

    inner = shapely.affinity.scale(outer, 0.4, 0.4)

    edge = shapely.difference(outer, middle)
    cortex = shapely.difference(middle, inner)
    medulla = inner

    return edge, cortex, medulla


class TestSpatialAxis:
    def test_hierarchical_labels(self):
        expected = numpy.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        labels = numpy.eye(4)
        # Uniquely label non-zero diagonal elements
        labels = scipy.ndimage.label(labels)[0]

        # Define broad annotations
        broad_annotations = numpy.zeros_like(labels)
        broad_annotations[:, :2] = 1
        broad_annotations[:, 2:] = 2

        me = SpatialAxis(
            instance_shapes=labels,
            broad_annotation_shapes=broad_annotations,
            broad_annotation_order=[1, 2],
        )
        rel_distances = me.get_relative_distances(k_neighbours=1)

        rel_labs = me.get_relative_distance_labelmap(rel_distances)

        numpy.testing.assert_array_equal(rel_labs, expected)

    def test_3d_hierarchical_labels(self):
        expected = numpy.array(
            [
                [[-1, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, -1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
            ]
        )

        labels = numpy.zeros((3, 3, 3)).astype(int)

        broad_annotations = labels.copy()

        labels[0, 0, 0] = 1
        labels[1, 1, 1] = 2
        labels[2, 2, 2] = 3

        broad_annotations[:, :, :2] = 1
        broad_annotations[:, :, 2:] = 2

        me = SpatialAxis(
            instance_shapes=labels,
            broad_annotation_shapes=broad_annotations,
            broad_annotation_order=[1, 2],
        )
        rel_distances = me.get_relative_distances(k_neighbours=1)

        rel_labs = me.get_relative_distance_labelmap(rel_distances)

        numpy.testing.assert_array_equal(rel_labs, expected)

    def test_shapely_labels(self):
        instance_polygons = [
            shapely.geometry.box(0, 0, 1, 1),
            shapely.geometry.box(1, 1, 2, 2),
            shapely.geometry.box(2, 2, 3, 3),
            shapely.geometry.box(3, 3, 4, 4),
            shapely.geometry.box(4, 4, 5, 5),
            shapely.geometry.box(5, 5, 6, 6),
            shapely.geometry.box(6, 6, 7, 7),
        ]

        broad_annotation_polygons = create_broad_annotation_polygons((10, 10))

        cells_df = geopandas.GeoDataFrame(
            {
                "geometry": instance_polygons,
                "annotation_id": numpy.arange(1, len(instance_polygons) + 1),
            }
        )

        broad_df = geopandas.GeoDataFrame(
            {
                "geometry": broad_annotation_polygons,
                "broad_annotation_id": ["edge", "cortex", "medulla"],
            }
        )
        # SpatialData uses label IDs as the index, so we do to
        cells_df = cells_df.set_index("annotation_id")
        broad_df = broad_df.set_index("broad_annotation_id")

        me = SpatialAxis(
            cells_df, broad_df, broad_annotation_order=["edge", "cortex", "medulla"]
        )

        observed = me.get_relative_distances(k_neighbours=1)

        expected = numpy.array([-1.5, 0.0, 0.0, 1.5, 1.33333333, 1.25, 1.2])

        numpy.testing.assert_almost_equal(observed, expected)

    def check_heirarchy_order(self):
        """
        Ensure that the heirarchy is connected in
        a way that is expected.

        ie.
        """
        pass
