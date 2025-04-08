import anndata
import geopandas


def validate_input(data, broad_annotations=None):
    if isinstance(data, anndata.AnnData):
        assert "spatial" in data.obsm, "AnnData object must have 'spatial' in obsm"

    if isinstance(data, geopandas.GeoDataFrame):
        assert "geometry" in data.columns

    if broad_annotations is not None:
        if isinstance(broad_annotations, geopandas.GeoDataFrame):
            assert "geometry" in broad_annotations.columns
