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
