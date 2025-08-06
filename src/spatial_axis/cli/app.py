import pathlib
import tomllib

import anndata
import typer
from InquirerPy import inquirer

from spatial_axis import spatial_axis
from spatial_axis.data import subset_dataframe

from ._search import search
import logging

import numpy

# from spatial_axis.validation import validate_spatial_axis_config

log = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def calculate(config_file: str):
    config = pathlib.Path(config_file)

    if config.is_dir():
        config_path = inquirer.fuzzy(
            "Select a config file", choices=search(config, ".toml")
        ).execute()
    elif config.suffix == ".toml":
        config_path = config
    else:
        raise ValueError

    config = tomllib.load(open(config_path, "rb"))

    # validate_spatial_axis_config(config)

    data_path = config.get("data_path")
    data_path = pathlib.Path(data_path)

    save_path = config.get("save_path")
    assert save_path is not None, "Please provide save_path"

    added_spatial_axis_key = config.get("save_column", "spatial_axis")

    batch_id = config.get("batch_id_column")

    subset_filter = config.get("subset_filter", None)

    if data_path.suffix == ".zarr":
        # SpatialData
        import spatialdata

        sdata = spatialdata.read_zarr(
            data_path
            )
        data = sdata.tables["table"]
    elif data_path.suffix == ".h5ad":
        # AnnData
        data = anndata.io.read_h5ad(data_path)
    else:
        raise ValueError(
            f"Cannot determine a reader for {data_path.suffix}. Expected a .zarr or .h5ad file."
        )

    if subset_filter is not None:
        # Subset the DataFrame
        # Since data is an AnnData object, find indices from the obs.
        subset_idx = subset_dataframe(data.obs, subset_filter = subset_filter).index.astype(int)
        log.info(f"Subsetting input data. Subset is {round((len(subset_idx) / len(data.obs.index)), 3) * 100}% of original dataset.")
    else:
        # TODO: this doesn't work if batched. Ensure XOR
        subset_idx = slice(None)

    if batch_id is not None:
        # Find the indices for batches of data
        # Optionally, subset the data first
        batched_data = data[subset_idx].obs.groupby(batch_id).indices.items()

        # Create an empty array to store spatial_axis values
        spatial_data = numpy.empty(len(data))
        spatial_data[:] = numpy.nan
        
        assert added_spatial_axis_key not in data.obs.columns, f"save_column {added_spatial_axis_key} found in adata.obs. Exiting."

        for batch_key, batch_idx in batched_data:
            log.info(f"Computing spatial_axis for: {batch_key}")
            # Convert batch_idx (which are positions in the subset) 
            # back to original data indices
            original_indices = numpy.array(subset_idx[batch_idx])

            # Calculate spatial axis for the batch
            sp_ax = spatial_axis(
                # batch_idx was found for the subset_idx, if requested
                # so we can use these idx to subset the original DF
                data=data[original_indices],
                annotation_order=config.get("annotation_order"),
                k_neighbours=config.get("k_neighbours"),
                annotation_column=config.get("annotation_column"),
                broad_annotations=config.get("broad_annotations"),
                missing_annotation_method=config.get("missing_annotation_method"),
                replace_value=config.get("replace_value"),
                class_to_exclude=config.get("class_to_exclude"),
                exclusion_value=config.get("exclusion_value"),
                auxiliary_class=config.get("auxiliary_class"),
                normalise=config.get("normalise", True),
                reference_cell_type=config.get("reference_cell_type", None),
                distance_threshold=config.get("distance_threshold", None),
            )

            spatial_data[original_indices] = sp_ax

        data.obs[added_spatial_axis_key] = spatial_data

    else:
        log.info("Computing spatial_axis")
        sp_ax = spatial_axis(
            data=data,
            annotation_order=config.get("annotation_order"),
            k_neighbours=config.get("k_neighbours"),
            annotation_column=config.get("annotation_column"),
            broad_annotations=config.get("broad_annotations"),
            missing_annotation_method=config.get("missing_annotation_method"),
            replace_value=config.get("replace_value"),
            class_to_exclude=config.get("class_to_exclude"),
            exclusion_value=config.get("exclusion_value"),
            auxiliary_class=config.get("auxiliary_class"),
            normalise=config.get("normalise", True),
        )
        data.obs[added_spatial_axis_key] = sp_ax
        
    log.info(f"Saving {save_path}")

    if data_path.suffix == ".zarr":
        if str(save_path) == str(data_path):
            # From: https://github.com/scverse/spatialdata/blob/main/tests/io/test_readwrite.py
            # Write element to existing store
            sdata["table_new"] = sdata["table"]
            sdata.write_element("table_new")
            sdata.delete_element_from_disk("table")
            sdata.write_element("table")
            del sdata["table_new"]
            sdata.delete_element_from_disk("table_new")
        else: 
            # We have modified the SpatialData in place, 
            # so we can save as is.
            sdata.write(save_path, overwrite=True)

    elif data_path.suffix == ".h5ad":
        data.write_h5ad(save_path)
