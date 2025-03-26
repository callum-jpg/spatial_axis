import pathlib
import tomllib

import typer
from InquirerPy import inquirer

from spatial_axis import spatial_axis
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

    if data_path.suffix == ".zarr":
        import spatialdata

        # SpatialData
        sdata = spatialdata.read_zarr(data_path, selection=["table", "tables"])
        data = sdata.tables["table"]
    elif data_path.suffix == ".h5ad":
        import anndata

        data = anndata.io.read_h5ad(data_path)
    else:
        raise ValueError(
            f"Cannot determine a reader for {data_path.suffix}. Expected a .zarr or .h5ad file."
        )

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

    data["spatial_axis"] = sp_ax

    save_path = config.get("save_path")

    if data_path.suffix == ".zarr":
        sdata.write(save_path)

    elif data_path.suffix == ".h5ad":
        data.write_h5ad(save_path)
