import pathlib
import tomllib

import typer
from InquirerPy import inquirer

from spatial_axis import spatial_axis
import anndata
# from spatial_axis.validation import validate_spatial_axis_config

from ._search import search

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

    if data_path.suffix == ".zarr":
        # SpatialData
        import spatialdata
        sdata = spatialdata.read_zarr(data_path)
        data = sdata.tables["table"]
    elif data_path.suffix == ".h5ad":
        # AnnData
        data = anndata.io.read_h5ad(data_path)
    else:
        raise ValueError(
            f"Cannot determine a reader for {data_path.suffix}. Expected a .zarr or .h5ad file."
        )


    if batch_id is not None:
        all_batch_data = []
        batched_data = data.obs.groupby(batch_id).indices.items()
        import numpy
        spatial_data = numpy.zeros(len(data))
        for batch_key, batch_idx in batched_data:
            # Calculate spatial axis for the batch
            sp_ax = spatial_axis(
                data=data[batch_idx],
                annotation_order=config.get("annotation_order"),
                k_neighbours=config.get("k_neighbours"),
                annotation_column=config.get("annotation_column"),
                broad_annotations=config.get("broad_annotations"),
                missing_annotation_method=config.get("missing_annotation_method"),
                replace_value=config.get("replace_value"),
                class_to_exclude=config.get("class_to_exclude"),
                exclusion_value=config.get("exclusion_value"),
            )

            spatial_data[batch_idx] = sp_ax

        data.obs[added_spatial_axis_key] = spatial_data

    else:
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
        )
        data.obs[added_spatial_axis_key] = sp_ax

    if data_path.suffix == ".zarr":
        sdata.write(save_path, overwrite=True)

    elif data_path.suffix == ".h5ad":
        data.write_h5ad(save_path)
