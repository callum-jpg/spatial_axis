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

def test_spatial_axis_anndata_replace():
    adata = toy_anndata(
        n_samples = 4,
        class_id = [0, 0, 1, 1],
    )

    # Replace has no impact since all classes are present
    observed = spatial_axis(
        adata,
        annotation_column="class_id", 
        annotation_order=[0, 1],
        missing_annotation_method="replace",
        k_neighbours=1
    )

    expected = numpy.array([-1, -1, 1, 1])

    numpy.testing.assert_equal(observed, expected)

    # Class not present, so replace nan with 0
    observed = spatial_axis(
        adata,
        annotation_column="class_id", 
        annotation_order=[0, 2],
        missing_annotation_method="replace",
        replace_value=0,
        k_neighbours=1
    )

    expected = numpy.array([0, 0, 1, 1])

    numpy.testing.assert_equal(observed, expected)

def test_spatial_axis_auxiliary_class():
    adata = toy_anndata(
        n_samples = 3,
        class_id = [0, 1, 2],
    )

    observed = spatial_axis(
        adata,
        annotation_column="class_id", 
        annotation_order=[0, 1],
        auxiliary_class=2,
        k_neighbours=1,
    )

    """Explanation of expected:

    We are also now computing the distance between class
    centroids AND to a single auxiliary class. The resulting
    distances are mean aggregated.

    For centroid 1, which is at position [0, 0], the distances to 
    all centroids are: [0, 1.41, 2.82] for class 0.

    For centroid 1, the distance to the auxiliary class is:
    [2.82, 1.41, 0].

    The mean distances for each centroid is: [1.41, 1.41, 1.41] for
    for class 0.

    For class 2:
    distance to centroids: [1.41, 0, 1.41]
    distance to aux: [2.82, 1.41, 0]
    mean distances: [2.115, 0.705, 0.705]

    Relative position is computed as before (example for the first
    centroid only. ie. 0th postion):
    (1.41 - 2.115) / (1.41 + 2.115) = -0.2
    
    There is only two classes, so only one relative position per
    centroid is calculcated (ie. no sum step)
    """

    expected = numpy.array([-0.2, 1/3, 1/3])

    numpy.testing.assert_almost_equal(observed, expected)


