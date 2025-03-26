import anndata
import numpy
import pandas

def toy_anndata(n_samples = 10):
    adata = anndata.AnnData(
        X=numpy.random.normal(size=(n_samples, 10)), 
        obs=pandas.DataFrame(
            numpy.random.randint(0, 4, size=(n_samples)), 
            columns=["class_id"]
            ),
        obsm = {
            # "spatial": numpy.random.randint(0, 10, size=(N_SAMPLES, 2))
            "spatial":numpy.dstack(
                numpy.where(
                    numpy.eye(n_samples)
                    )
                ).squeeze()
            },
        )

    return adata