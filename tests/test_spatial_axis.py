import geopandas
import numpy
import shapely

from spatial_axis import spatial_axis
from spatial_axis.utility import create_broad_annotation_polygons
from spatial_axis.data import toy_anndata

def test_spatial_axis_anndata():
    adata = toy_anndata(
        n_samples = 4,
        class_id = [0, 0, 1, 1],
    )

    observed = spatial_axis(
        adata,
        annotation_column="class_id", 
        annotation_order=[0, 1],
        k_neighbours=1
    )

    """
    Explanation why the expected is [-1, -1, 1, 1]:

    The generated anndata has spatial coordinates
    [
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
    ]

    In euclidean space, these coordinates are have a disntace of 
    ~1.41 between each.

    When we calculate the KNN graph based on the annotation class,
    the distance of class 0 to class 0 (ie. 0th and 1st samples) is 0. 

    The distance for the 0th and 1st samples is ~2.82 and ~1.41 respectively
    (remember, each sample is ~1.41 from its neighbour).

    When we compute the relative positioning, we take the distance
    difference divided by the distance sum. For the 0th sample, this looks like:
    (0 - 2.82) / (0 + 2.82) = -1
    For the 1st sample:
    (0 - 1.41) / (0 + 1.41) = -1
    etc.
    """

    expected = numpy.array([-1, -1, 1, 1])

    numpy.testing.assert_equal(observed, expected)

# class TestSpatialAxis:
#     #### TODO: reinstate labelmap support (this allows for 3D to be compatible. Otherwise,
#     #### shapely does not support 3D polygons)
#     # def test_hierarchical_labels(self):
#     #     expected = numpy.array(
#     #         [
#     #             [-1.0, 0.0, 0.0, 0.0],
#     #             [0.0, -1.0, 0.0, 0.0],
#     #             [0.0, 0.0, 1.0, 0.0],
#     #             [0.0, 0.0, 0.0, 1.0],
#     #         ]
#     #     )

#     #     labels = numpy.eye(4)
#     #     # Uniquely label non-zero diagonal elements
#     #     labels = scipy.ndimage.label(labels)[0]

#     #     # Define broad annotations
#     #     broad_annotations = numpy.zeros_like(labels)
#     #     broad_annotations[:, :2] = 1
#     #     broad_annotations[:, 2:] = 2

#     #     me = SpatialAxis(
#     #         instance_shapes=labels,
#     #         broad_annotation_shapes=broad_annotations,
#     #         broad_annotation_order=[1, 2],
#     #     )
#     #     rel_distances = me.get_relative_distances(k_neighbours=1)

#     #     rel_labs = me.get_relative_distance_labelmap(rel_distances)

#     #     numpy.testing.assert_array_equal(rel_labs, expected)

#     # def test_3d_hierarchical_labels(self):
#     #     expected = numpy.array(
#     #         [
#     #             [[-1, 0, 0], [0, 0, 0], [0, 0, 0]],
#     #             [[0, 0, 0], [0, -1, 0], [0, 0, 0]],
#     #             [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
#     #         ]
#     #     )

#     #     labels = numpy.zeros((3, 3, 3)).astype(int)

#     #     broad_annotations = labels.copy()

#     #     labels[0, 0, 0] = 1
#     #     labels[1, 1, 1] = 2
#     #     labels[2, 2, 2] = 3

#     #     broad_annotations[:, :, :2] = 1
#     #     broad_annotations[:, :, 2:] = 2

#     #     me = SpatialAxis(
#     #         instance_shapes=labels,
#     #         broad_annotation_shapes=broad_annotations,
#     #         broad_annotation_order=[1, 2],
#     #     )
#     #     rel_distances = me.get_relative_distances(k_neighbours=1)

#     #     rel_labs = me.get_relative_distance_labelmap(rel_distances)

#     #     numpy.testing.assert_array_equal(rel_labs, expected)

#     def test_shapely_labels(self):
#         instance_polygons = [
#             shapely.geometry.box(0, 0, 1, 1),
#             shapely.geometry.box(1, 1, 2, 2),
#             shapely.geometry.box(2, 2, 3, 3),
#             shapely.geometry.box(3, 3, 4, 4),
#             shapely.geometry.box(4, 4, 5, 5),
#             shapely.geometry.box(5, 5, 6, 6),
#             shapely.geometry.box(6, 6, 7, 7),
#         ]

#         broad_annotation_polygons = create_broad_annotation_polygons(
#             (10, 10),
#             num_levels=3,
#             downscale_factor=0.8,
#         )

#         cells_df = geopandas.GeoDataFrame(
#             {
#                 "geometry": instance_polygons,
#                 "annotation_id": numpy.arange(1, len(instance_polygons) + 1),
#             }
#         )

#         broad_df = geopandas.GeoDataFrame(
#             {
#                 "geometry": broad_annotation_polygons,
#                 "broad_annotation_id": [1, 2, 3],
#             }
#         )
#         # SpatialData uses label IDs as the index, so we do to
#         cells_df = cells_df.set_index("annotation_id")
#         broad_df = broad_df.set_index("broad_annotation_id")

#         observed = spatial_axis(
#             instance_objects=cells_df,
#             broad_annotations=broad_df,
#             broad_annotation_order=[1, 2, 3],
#             k_neighbours=1,
#         )

#         expected = numpy.array(
#             [-1.3333333, 0.0, 1.3333333, 1.2, 1.1428571, 1.1111111, 1.0909091]
#         )

#         numpy.testing.assert_almost_equal(observed, expected)

#     def test_anndata(self):
#         pass
