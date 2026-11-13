import numpy as np
import pandas as pd


def extract_meta_features(dataset):

    X = dataset.X
    y = dataset.y

    n_samples, n_features = X.shape

    # Number of classes
    class_counts = y.value_counts(normalize=True)
    n_classes = len(class_counts)

    # Normalised class entropy
    if n_classes > 1:
        class_entropy = -np.sum(
            class_counts * np.log2(class_counts)
        ) / np.log2(n_classes)
    else:
        class_entropy = 0.0

    # Class imbalance
    majority_class_ratio = class_counts.max()
    minority_class_ratio = class_counts.min()

    # Average absolute feature correlation
    correlations = X.corr().abs()

    upper_triangle = correlations.where(
        np.triu(
            np.ones(correlations.shape),
            k=1
        ).astype(bool)
    )

    mean_correlation = upper_triangle.stack().mean()

    if pd.isna(mean_correlation):
        mean_correlation = 0.0

    # Feature sparsity
    sparsity = (X == 0).sum().sum() / X.size

    return {
        "dataset": dataset.name,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "samples_per_feature": n_samples / n_features,
        "class_entropy": class_entropy,
        "majority_class_ratio": majority_class_ratio,
        "minority_class_ratio": minority_class_ratio,
        "mean_feature_correlation": mean_correlation,
        "feature_sparsity": sparsity,
    }