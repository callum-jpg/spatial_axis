import numpy

from spatial_axis import spatial_axis
from spatial_axis.data import toy_anndata

import pytest
import numpy
import pandas
import geopandas
import anndata
from shapely.geometry import Point, Polygon
from unittest.mock import patch, MagicMock
import scipy.spatial

from spatial_axis.spatial_axis import (
    spatial_axis, 
    _get_centroids,
    _get_centroid_class,
    _spatial_axis,
    get_centroid_distances,
    get_shapely_centroid,
    compute_relative_positioning,
    get_label_centroids,
    _class_exclusion
)
from spatial_axis.constants import SpatialAxisConstants


def test_spatial_axis_anndata():
    adata = toy_anndata(
        n_samples=4,
        class_id=[0, 0, 1, 1],
    )

    observed = spatial_axis(
        adata, annotation_column="class_id", annotation_order=[0, 1], k_neighbours=1
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
        n_samples=4,
        class_id=[0, 0, 1, 1],
    )

    # Replace has no impact since all classes are present
    observed = spatial_axis(
        adata,
        annotation_column="class_id",
        annotation_order=[0, 1],
        missing_annotation_method="replace",
        k_neighbours=1,
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
        k_neighbours=1,
    )

    expected = numpy.array([0, 0, 1, 1])

    numpy.testing.assert_equal(observed, expected)


def test_spatial_axis_auxiliary_class():
    adata = toy_anndata(
        n_samples=3,
        class_id=[0, 1, 2],
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

    expected = numpy.array([-0.2, 1 / 3, 1 / 3])

    numpy.testing.assert_almost_equal(observed, expected)

@pytest.fixture
def sample_anndata():
    """Create a simple anndata object with spatial coordinates."""
    obs = pandas.DataFrame(index=[f"cell_{i}" for i in range(5)])
    var = pandas.DataFrame(index=[f"gene_{i}" for i in range(3)])
    X = numpy.random.rand(5, 3)
    
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.obsm[SpatialAxisConstants.spatial_key] = numpy.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 1.0],
        [5.0, 5.0]
    ])
    
    # Add an annotation column
    adata.obs['annotation'] = pandas.Series([1, 1, 2, 2, 3], index=adata.obs.index)
    
    return adata


@pytest.fixture
def sample_gdf():
    """Create a GeoDataFrame with some polygons."""
    geometry = [
        Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        Polygon([(2, 0), (4, 0), (4, 3), (2, 3)]),
        Polygon([(4, 3), (6, 3), (6, 6), (4, 6)])
    ]
    return geopandas.GeoDataFrame(geometry=geometry, crs="EPSG:4326")


@pytest.fixture
def broad_annotations_array():
    """Create a simple numpy array for broad annotations."""
    arr = numpy.zeros((6, 6), dtype=int)
    arr[0:2, 0:2] = 1  # Top left: class 1
    arr[2:4, 0:3] = 2  # Middle: class 2
    arr[4:6, 3:6] = 3  # Bottom right: class 3
    return arr


@pytest.fixture
def sample_centroids():
    """Create centroids for testing."""
    return numpy.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 1.0],
        [5.0, 5.0]
    ])


@pytest.fixture
def sample_centroid_class():
    """Create centroid classes."""
    return numpy.array([1, 1, 2, 3])


def test_spatial_axis_with_anndata_and_annotation_column(sample_anndata):
    """Test spatial_axis with AnnData and annotation column."""
    result = spatial_axis(
        data=sample_anndata,
        annotation_order=[1, 2, 3],
        k_neighbours=2,
        annotation_column='annotation'
    )
    
    assert isinstance(result, numpy.ndarray)
    assert len(result) == 5  # One value per cell


def test_spatial_axis_with_gdf_and_broad_annotations(sample_gdf):
    """Test spatial_axis with GeoDataFrame and broad annotations."""
    result = spatial_axis(
        data=sample_gdf,
        annotation_order=[0, 1, 2],
        k_neighbours=1,
        broad_annotations=sample_gdf
    )
    
    assert isinstance(result, numpy.ndarray)
    assert len(result) == 3  # One value per polygon


def test_spatial_axis_with_missing_annotation_replace(sample_anndata):
    """Test handling missing annotations with replace method."""
    # Create test data with some NaN annotations
    sample_anndata.obs['sparse_annotation'] = pandas.Series([1, numpy.nan, 2, numpy.nan, 3], index=sample_anndata.obs.index)
    
    result = spatial_axis(
        data=sample_anndata,
        annotation_order=[1, 2, 3],
        k_neighbours=2,
        annotation_column='sparse_annotation',
        missing_annotation_method='replace',
        replace_value=0.5
    )
    
    assert isinstance(result, numpy.ndarray)
    assert len(result) == 5
    assert not numpy.isnan(result).any()  # No NaN values after replacement


def test_spatial_axis_with_missing_annotation_knn(sample_anndata):
    """Test handling missing annotations with KNN imputation."""
    # Create test data with some NaN annotations
    sample_anndata.obs['sparse_annotation'] = pandas.Series([1, numpy.nan, 2, numpy.nan, 3], index=sample_anndata.obs.index)
    
    result = spatial_axis(
        data=sample_anndata,
        annotation_order=[1, 2, 3],
        k_neighbours=2,
        annotation_column='sparse_annotation',
        missing_annotation_method='knn'
    )
    
    assert isinstance(result, numpy.ndarray)
    assert len(result) == 5
    assert not numpy.isnan(result).any()  # No NaN values after KNN imputation


def test_spatial_axis_with_class_exclusion(sample_anndata):
    """Test class exclusion functionality."""
    result = spatial_axis(
        data=sample_anndata,
        annotation_order=[1, 2, 3],
        k_neighbours=2,
        annotation_column='annotation',
        class_to_exclude=2,
        exclusion_value=numpy.nan
    )
    
    assert isinstance(result, numpy.ndarray)
    assert len(result) == 5
    # Check that values for class 2 are excluded (set to NaN)
    mask = sample_anndata.obs['annotation'] == 2
    assert numpy.isnan(result[mask.values]).all()
    assert not numpy.isnan(result[~mask.values]).any()


def test_spatial_axis_with_auxiliary_class(sample_anndata):
    """Test spatial_axis with auxiliary class parameter."""
    result = spatial_axis(
        data=sample_anndata,
        annotation_order=[1, 2],
        k_neighbours=2,
        annotation_column='annotation',
        auxiliary_class=3
    )
    
    assert isinstance(result, numpy.ndarray)
    assert len(result) == 5


def test_get_centroids_from_anndata(sample_anndata):
    """Test extracting centroids from AnnData."""
    centroids = _get_centroids(sample_anndata)
    
    assert isinstance(centroids, numpy.ndarray)
    assert centroids.shape == (5, 2)
    numpy.testing.assert_array_equal(centroids, sample_anndata.obsm[SpatialAxisConstants.spatial_key])


def test_get_centroids_from_gdf(sample_gdf):
    """Test extracting centroids from GeoDataFrame."""
    with patch('spatial_axis.spatial_axis.get_shapely_centroid', side_effect=lambda x: numpy.array(x.centroid.coords[0])):
        centroids = _get_centroids(sample_gdf)
        
        assert isinstance(centroids, numpy.ndarray)
        assert centroids.shape == (3, 2)


def test_get_centroids_invalid_type():
    """Test _get_centroids with invalid data type."""
    with pytest.raises(ValueError):
        _get_centroids("invalid_data_type")


def test_get_centroid_class_from_gdf():
    """Test getting centroid classes from GeoDataFrame."""
    # Create test points and polygons
    points = numpy.array([[1.0, 1.0], [3.0, 1.0], [5.0, 5.0]])
    
    # Create mock polygons for broad annotations
    polygons = [
        Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),  # Class 0
        Polygon([(2, 0), (4, 0), (4, 3), (2, 3)]),  # Class 1
        Polygon([(4, 3), (6, 3), (6, 6), (4, 6)])   # Class 2
    ]
    
    broad_annotations = geopandas.GeoDataFrame(
        {'class_id': [0, 1, 2]},
        geometry=polygons,
        crs="EPSG:4326"
    )
    
    with patch('geopandas.sjoin') as mock_sjoin:
        # Mock the spatial join result
        result_df = pandas.DataFrame({
            'index_right': [0, 1, 2],
        }, index=[0, 1, 2])
        mock_sjoin.return_value = result_df
        
        centroid_class = _get_centroid_class(points, broad_annotations)
        
        assert isinstance(centroid_class, numpy.ndarray)
        assert len(centroid_class) == 3


def test_get_centroid_class_from_array(broad_annotations_array):
    """Test getting centroid classes from numpy array."""
    points = numpy.array([[1, 1], [3, 1], [5, 5]])
    
    centroid_class = _get_centroid_class(points, broad_annotations_array)
    
    assert isinstance(centroid_class, numpy.ndarray)
    assert len(centroid_class) == 3
    # Check if classes match the expected values based on the array regions
    assert centroid_class[0] == 1  # Point at [1,1] is in class 1
    assert centroid_class[1] == 2  # Point at [3,1] is in class 2
    assert centroid_class[2] == 3  # Point at [5,5] is in class 3


def test_spatial_axis_internal(sample_centroids, sample_centroid_class):
    """Test the internal _spatial_axis function."""
    result = _spatial_axis(
        centroids=sample_centroids,
        centroid_class=sample_centroid_class,
        class_order=[1, 2, 3],
        k_neighbours=1
    )
    
    assert isinstance(result, numpy.ndarray)
    assert result.shape == (4, 3)  # 4 centroids, 3 classes


def test_spatial_axis_with_auxiliary(sample_centroids, sample_centroid_class):
    """Test _spatial_axis with auxiliary class."""
    result = _spatial_axis(
        centroids=sample_centroids,
        centroid_class=sample_centroid_class,
        class_order=[1, 2],
        k_neighbours=1,
        auxiliary_class=3
    )
    
    assert isinstance(result, numpy.ndarray)
    assert result.shape == (4, 2)  # 4 centroids, 2 classes


def test_spatial_axis_empty_class(sample_centroids, sample_centroid_class):
    """Test _spatial_axis with an empty class."""
    result = _spatial_axis(
        centroids=sample_centroids,
        centroid_class=sample_centroid_class,
        class_order=[1, 2, 4],  # Class 4 doesn't exist
        k_neighbours=1
    )
    
    assert isinstance(result, numpy.ndarray)
    assert result.shape == (4, 3)

def test_spatial_axis_weights(sample_anndata, sample_centroid_class):
    """Test _spatial_axis with an empty class."""
    result_none = spatial_axis(
        data=sample_anndata,
        annotation_order=[1, 2, 3],
        annotation_column='annotation',
        weights=None
    )

    result_weights = spatial_axis(
        data=sample_anndata,
        annotation_order=[1, 2, 3],
        annotation_column='annotation',
        weights=[10, 10]
    )
    
    numpy.testing.assert_almost_equal(result_weights, result_none * 10)

