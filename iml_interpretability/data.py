"""Dataset loading.

Every dataset is returned as a :class:`Dataset` with numeric, imputed features so that all
models in :mod:`iml_interpretability.models` can be fit on it directly. Add further loaders
(e.g. other OpenML datasets) following the pattern of :func:`load_titanic`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
# OpenML downloads are cached here, relative to the current working directory.
DATA_HOME = Path("data")


@dataclass(frozen=True)
class Dataset:
    """A preprocessed tabular classification dataset.

    :param name: Human-readable dataset name, used in result tables.
    :param X: Numeric feature matrix without missing values.
    :param y: Target vector.
    """

    name: str
    X: pd.DataFrame
    y: pd.Series

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)




def load_dataset(data_home: str | Path = DATA_HOME) -> Dataset:
    """Load the Breast Cancer dataset.

    :param data_home: Cache directory for the download, relative paths are resolved against the
        current working directory.
    :return: The preprocessed Titanic dataset.
    :rtype: Dataset
    """
    # The default pandas ARFF parser crashes on Titanic's integer-valued categorical columns.
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.data
    y = bunch.target.rename("diagnosis")

    return Dataset(name="breast_cancer", X=X, y=y)

def load_digits_dataset() -> Dataset:
    """Load the handwritten digits classification dataset."""
    bunch = load_digits(as_frame=True)

    return Dataset(
        name="digits",
        X=bunch.data,
        y=bunch.target.rename("digit"),
    )

def load_iris_dataset() -> Dataset:
    """Load the Iris classification dataset."""
    bunch = load_iris(as_frame=True)

    return Dataset(
        name="iris",
        X=bunch.data,
        y=bunch.target.rename("species"),
    )

def load_wine_dataset() -> Dataset:
    """Load the Wine classification dataset."""
    bunch = load_wine(as_frame=True)

    return Dataset(
        name="wine",
        X=bunch.data,
        y=bunch.target.rename("wine_type"),
    )
      
    
