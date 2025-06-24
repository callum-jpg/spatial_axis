import anndata
import numpy
import pandas


def toy_anndata(n_samples=10, class_id=None):
    if class_id is not None:
        assert len(class_id) == n_samples, "Must provide a class_id for every sample."
    else:
        class_id = numpy.random.randint(0, 4, size=(n_samples))

    adata = anndata.AnnData(
        X=numpy.random.normal(size=(n_samples, 10)),
        obs=pandas.DataFrame(class_id, columns=["class_id"]),
        obsm={
            # "spatial": numpy.random.randint(0, 10, size=(N_SAMPLES, 2))
            "spatial": numpy.dstack(numpy.where(numpy.eye(n_samples))).squeeze()
        },
    )

    return adata


def subset_dataframe(df: pandas.DataFrame, subset_filter: dict[str, str] | list[dict[str, str]]) -> pandas.DataFrame:
    """
    Subsets a DataFrame based on a dictionary of filtering conditions.
    
    Args:
        df (pandas.DataFrame): The DataFrame to be filtered.
        subset_filter (dict): A dictionary where keys are column names (str),
                        and values are the values that should be in the respective columns.
    
    Returns:
        pandas.DataFrame: The filtered DataFrame.
    """
    # Start with all True values (ie. all rows kept)
    if isinstance(subset_filter, dict):
        subset_filter = [subset_filter]  # Convert single dictionary to list
    
    combined_mask = pandas.Series(False, index=df.index)  # Start with all False values to apply OR condition
    
    for filter_dict in subset_filter:
        mask = pandas.Series(True, index=df.index)  # Start with all True values for AND condition
        
        for key, value in filter_dict.items():
            if key in df.columns:
                if isinstance(value, list):  # If value is a list, check if any match
                    value = [v.lower() if isinstance(v, str) else v for v in value]
                    col_values = df[key].astype(str).str.lower() if df[key].dtype == 'O' else df[key]
                    mask &= col_values.isin(value)
                else:  # Single value case
                    value = value.lower() if isinstance(value, str) else value
                    col_values = df[key].astype(str).str.lower() if df[key].dtype == 'O' else df[key]
                    mask &= col_values == value
        
        combined_mask |= mask  # Combine masks using OR condition
    
    return df[combined_mask]