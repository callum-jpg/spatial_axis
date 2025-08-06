import numpy
from scipy.spatial.distance import cdist
import anndata
import pandas
import logging

log = logging.getLogger(__name__)

def spatial_celltype_filter(
    adata, 
    celltype_col: str, 
    query_cell: str,
    reference_cell: str = None, 
    new_celltype: str = None, 
    distance_threshold=20,
    spatial_obsm_key="spatial",
):
    """
    Create a new cell type based on spatial proximity conditions.
    
    Parameters:
    -----------
    adata : AnnData object
        The annotated data object
    celltype_col : str
        Name of the obs column containing cell types
    query_cell : str
        Source cell type to check
    reference_cell : str
        Target cell type to check proximity to
    new_celltype : str
        Name of the new cell type to create
    distance_threshold : float
        Distance threshold in pixels (default: 20)
    spatial_coords : list
        Names of spatial coordinate columns in adata.obs (default: ['x', 'y'])
    
    Returns:
    --------
    None (modifies adata.obs in place)
    """

    if isinstance(query_cell, str) or isinstance(query_cell, int):
        query_cell = [query_cell]
    else:
        assert len(query_cell) == 1, f"Can only support one query cell type. Got {len(query_cell)}"
    
    if isinstance(reference_cell, str) or isinstance(reference_cell, int):
        reference_cell = [reference_cell]
    else:
        assert len(reference_cell) == 1, f"Can only support one reference cell type. Got {len(reference_cell)}"

    if new_celltype is None:
        new_celltype = query_cell[0]
    
    # Create a copy of the original cell type column for the new assignment
    new_celltype_col = f"{celltype_col}_spatial_filter"
    
    # Get the XY spatial coords
    coords = adata.obsm[spatial_obsm_key]
    
    # Get indices of query_cell and reference_cell
    query_cell_mask = adata.obs[celltype_col].isin(query_cell)
    
    # If there's a reference_cell, get the mask for this
    if reference_cell is not None:
        reference_cell_mask = adata.obs[celltype_col].isin(reference_cell)
    # Otherwise, we will be looking at distances to other cells
    else:
        reference_cell_mask = adata.obs[celltype_col].isin(query_cell)
    
    if not query_cell_mask.any():
        log.info(f"Warning: No cells found with cell type '{query_cell}'")
        return
    
    if not reference_cell_mask.any():
        log.info(f"Warning: No cells found with cell type '{reference_cell}'")
        return
    
    # Get coordinates for each cell type
    coords_a = coords[query_cell_mask]
    coords_b = coords[reference_cell_mask]
    
    # Calculate distances between query_cell and reference_cell
    distances = cdist(coords_a, coords_b)

    # Remove distances of cells to self from consideration
    if numpy.array_equal(coords_a, coords_b):
        numpy.fill_diagonal(distances, numpy.nan)
    
    # Find query_cell cells that are within threshold distance of any reference_cell
    min_distances = numpy.nanmin(distances, axis=1)
    close_cells_mask = min_distances <= distance_threshold
    
    # Get the indices of query_cell cells that should become new_celltype
    query_cell_indices = numpy.where(query_cell_mask)[0]
    cells_to_change = query_cell_indices[close_cells_mask]

    # Define the new annotations using the cells_to_change mask
    new_annotations = pandas.Series(
        data=numpy.nan,
        index=adata.obs.index,
        dtype="object"
    )
    new_annotations.iloc[cells_to_change] = new_celltype

    # Update adata with new annotations
    adata.obs[new_celltype_col] = new_annotations
    
    log.info(f"Created new cell type '{new_celltype}' for {len(cells_to_change)} cells in adata.obs['{new_celltype_col}']")
    
    return new_celltype_col