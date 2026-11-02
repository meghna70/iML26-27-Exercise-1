"""Dataset loading.

Every dataset is returned as a :class:`Dataset` with numeric, imputed features so that all
models in :mod:`iml_interpretability.models` can be fit on it directly. Add further loaders
(e.g. other OpenML datasets) following the pattern of :func:`load_titanic`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

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


# Columns that leak the target (boat = lifeboat, body = body identification number) or are
# free-text identifiers without a meaningful encoding.
_TITANIC_DROP_COLUMNS = ["boat", "body", "home.dest", "name", "ticket", "cabin"]


def load_titanic(data_home: str | Path = DATA_HOME) -> Dataset:
    """Load the Titanic dataset from OpenML (data id 40945).

    The download is cached in ``data_home``, so network access is only needed on the first call.

    :param data_home: Cache directory for the download, relative paths are resolved against the
        current working directory.
    :return: The preprocessed Titanic dataset.
    :rtype: Dataset
    """
    # The default pandas ARFF parser crashes on Titanic's integer-valued categorical columns.
    bunch = fetch_openml("titanic", version=1, as_frame=True, parser="liac-arff", data_home=data_home)
    X = bunch.data.drop(columns=_TITANIC_DROP_COLUMNS)
    y = bunch.target.astype(int).rename("survived")

    numeric_columns = X.select_dtypes(include="number").columns
    X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())
    X = pd.get_dummies(X, drop_first=True, dtype=float)

    return Dataset(name="titanic", X=X, y=y)
