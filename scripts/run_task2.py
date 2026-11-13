from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_digits,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------
# Project setup
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "task2_plots"

RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Make the project package importable when running this script
sys.path.insert(0, str(PROJECT_ROOT))

from iml_interpretability.metric import DecisionTreeComplexityScore


# ---------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------

RANDOM_SEEDS = [0, 21, 42, 63, 84]

MAX_DEPTHS = [1, 3, 5, 7, 10, None]

TEST_SIZE = 0.25

# The same DTCCS implementation is used throughout Task 2.
metric = DecisionTreeComplexityScore()


# ---------------------------------------------------------
# 1. Load datasets
# ---------------------------------------------------------

def load_datasets():
    """Load the four classification datasets used in Task 2."""

    datasets = {}

    sklearn_datasets = {
        "iris": load_iris(),
        "wine": load_wine(),
        "breast_cancer": load_breast_cancer(),
        "digits": load_digits(),
    }

    for name, data in sklearn_datasets.items():
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")

        datasets[name] = {
            "X": X,
            "y": y,
        }

    return datasets


# ---------------------------------------------------------
# 2. Extract dataset meta-features
# ---------------------------------------------------------

def extract_meta_features(X, y):
    """
    Extract descriptive characteristics of a classification dataset.

    These features describe dataset size, dimensionality,
    class distribution, and feature characteristics.
    """

    n_samples, n_features = X.shape
    n_classes = y.nunique()

    # Class distribution
    class_counts = y.value_counts()

    class_probabilities = class_counts / class_counts.sum()

    class_entropy = -np.sum(
        class_probabilities * np.log2(class_probabilities)
    )

    if n_classes > 1:
        normalized_class_entropy = class_entropy / np.log2(n_classes)
    else:
        normalized_class_entropy = 0.0

    majority_class_ratio = class_probabilities.max()
    minority_class_ratio = class_probabilities.min()

    imbalance_ratio = (
        class_counts.max() / class_counts.min()
    )

    # Feature correlations
    if n_features > 1:
        correlation_matrix = X.corr().abs()

        upper_triangle = correlation_matrix.where(
            np.triu(
                np.ones(correlation_matrix.shape),
                k=1
            ).astype(bool)
        )

        correlations = upper_triangle.stack().dropna()

        mean_absolute_correlation = (
            correlations.mean()
            if len(correlations) > 0
            else 0.0
        )
    else:
        mean_absolute_correlation = 0.0

    # Fraction of feature values equal to zero
    feature_sparsity = float((X == 0).to_numpy().mean())

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "samples_per_feature": n_samples / n_features,
        "class_entropy": class_entropy,
        "normalized_class_entropy": normalized_class_entropy,
        "majority_class_ratio": majority_class_ratio,
        "minority_class_ratio": minority_class_ratio,
        "imbalance_ratio": imbalance_ratio,
        "mean_absolute_correlation": mean_absolute_correlation,
        "feature_sparsity": feature_sparsity,
    }


# ---------------------------------------------------------
# 3. Evaluate decision trees
# ---------------------------------------------------------

def run_experiments(datasets, meta_features):
    """
    Train decision trees with different maximum depths
    and evaluate their accuracy and DTCCS.

    Each experiment is repeated using multiple random seeds.
    """

    results = []

    for dataset_name, dataset in datasets.items():

        X = dataset["X"]
        y = dataset["y"]

        print(f"\nDataset: {dataset_name}")
        print(f"Samples: {len(X)}, Features: {X.shape[1]}")

        for seed in RANDOM_SEEDS:

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                random_state=seed,
                stratify=y,
            )

            for max_depth in MAX_DEPTHS:

                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    random_state=seed,
                )

                model.fit(X_train, y_train)

                # Predict on the test set
                accuracy = model.score(X_test, y_test)

                # Calculate DTCCS on the test set.
                # The metric expects an object with an X attribute.
                class TestDataset:
                    pass

                test_dataset = TestDataset()
                test_dataset.X = X_test
                test_dataset.y = y_test

                dtccs = metric.score(model, test_dataset)

                depth_label = (
                    "unrestricted"
                    if max_depth is None
                    else str(max_depth)
                )

                results.append({
                    "dataset": dataset_name,
                    "seed": seed,
                    "max_depth": depth_label,
                    "accuracy": accuracy,
                    "dtccs": dtccs,
                    "actual_tree_depth": model.get_depth(),
                    "number_of_leaves": model.get_n_leaves(),
                    "number_of_nodes": model.tree_.node_count,
                    "number_of_features_used": len(
                        np.unique(
                            model.tree_.feature[
                                model.tree_.feature >= 0
                            ]
                        )
                    ),
                    **meta_features[dataset_name],
                })

                print(
                    f"  Seed={seed:2d}, "
                    f"max_depth={depth_label:12s}, "
                    f"accuracy={accuracy:.3f}, "
                    f"DTCCS={dtccs:.2f}"
                )

    return pd.DataFrame(results)


# ---------------------------------------------------------
# 4. Summarise results
# ---------------------------------------------------------

def create_summary(results):
    """
    Calculate mean and standard deviation of accuracy and DTCCS
    for each dataset and tree configuration.
    """

    summary = (
        results
        .groupby(["dataset", "max_depth"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_dtccs=("dtccs", "mean"),
            std_dtccs=("dtccs", "std"),
            mean_actual_depth=("actual_tree_depth", "mean"),
            mean_leaves=("number_of_leaves", "mean"),
        )
    )

    return summary


# ---------------------------------------------------------
# 5. Correlation analysis
# ---------------------------------------------------------

def calculate_correlations(summary, meta_features):
    """
    Calculate Spearman correlations between dataset meta-features
    and mean DTCCS, separately for each tree configuration.

    Each correlation uses one observation per dataset.
    """

    feature_names = [
        "n_samples",
        "n_features",
        "n_classes",
        "samples_per_feature",
        "class_entropy",
        "normalized_class_entropy",
        "majority_class_ratio",
        "minority_class_ratio",
        "imbalance_ratio",
        "mean_absolute_correlation",
        "feature_sparsity",
    ]

    correlation_results = []

    # One row of metadata per dataset
    metadata_df = pd.DataFrame.from_dict(
        meta_features,
        orient="index"
    )

    metadata_df.index.name = "dataset"
    metadata_df = metadata_df.reset_index()

    for depth in summary["max_depth"].unique():

        depth_results = summary[
            summary["max_depth"] == depth
        ].merge(
            metadata_df,
            on="dataset",
            how="left",
        )

        for feature in feature_names:

            rho, p_value = spearmanr(
                depth_results[feature],
                depth_results["mean_dtccs"],
            )

            correlation_results.append({
                "max_depth": depth,
                "meta_feature": feature,
                "spearman_rho": rho,
                "p_value": p_value,
                "number_of_datasets": len(depth_results),
            })

    return pd.DataFrame(correlation_results)


# ---------------------------------------------------------
# 6. Visualise relationships
# ---------------------------------------------------------

def create_plots(summary, meta_features):
    """
    Create scatter plots showing the relationship between
    dataset meta-features and mean DTCCS.

    Each line represents a different tree configuration.
    """

    metadata_df = pd.DataFrame.from_dict(
        meta_features,
        orient="index"
    )

    metadata_df.index.name = "dataset"
    metadata_df = metadata_df.reset_index()

    features_to_plot = [
        "n_samples",
        "n_features",
        "n_classes",
        "normalized_class_entropy",
        "imbalance_ratio",
        "mean_absolute_correlation",
        "feature_sparsity",
    ]

    merged = summary.merge(
        metadata_df,
        on="dataset",
        how="left",
    )

    for feature in features_to_plot:

        plt.figure(figsize=(9, 6))

        for depth in merged["max_depth"].unique():

            subset = merged[
                merged["max_depth"] == depth
            ]

            plt.scatter(
                subset[feature],
                subset["mean_dtccs"],
                label=f"Max depth: {depth}",
                s=70,
            )

        plt.xlabel(feature.replace("_", " ").title())
        plt.ylabel("Mean DTCCS")
        plt.title(f"{feature.replace('_', ' ').title()} vs DTCCS")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        filename = f"{feature}_vs_dtccs.png"

        plt.savefig(
            PLOTS_DIR / filename,
            dpi=150,
        )

        plt.close()


# ---------------------------------------------------------
# 7. Main execution
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("TASK 2: DATASET META-FEATURES AND MODEL INTERPRETABILITY")
    print("=" * 60)

    # Load datasets
    datasets = load_datasets()

    # Extract meta-features
    meta_features = {
        name: extract_meta_features(
            dataset["X"],
            dataset["y"],
        )
        for name, dataset in datasets.items()
    }

    meta_features_df = pd.DataFrame.from_dict(
        meta_features,
        orient="index",
    )

    meta_features_df.index.name = "dataset"
    meta_features_df = meta_features_df.reset_index()

    print("\nDataset meta-features:")
    print(meta_features_df.to_string(index=False))

    meta_features_df.to_csv(
        RESULTS_DIR / "task2_dataset_meta_features.csv",
        index=False,
    )

    # Run experiments
    results = run_experiments(
        datasets,
        meta_features,
    )

    results.to_csv(
        RESULTS_DIR / "task2_model_results.csv",
        index=False,
    )

    # Summarise results
    summary = create_summary(results)

    summary.to_csv(
        RESULTS_DIR / "task2_summary.csv",
        index=False,
    )

    # Correlation analysis
    correlations = calculate_correlations(
        summary,
        meta_features,
    )

    correlations.to_csv(
        RESULTS_DIR / "task2_spearman_correlations.csv",
        index=False,
    )

    # Visualisations
    create_plots(
        summary,
        meta_features,
    )

    print("\n" + "=" * 60)
    print("TASK 2 COMPLETED")
    print("=" * 60)

    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\nSpearman correlations:")
    print(
        correlations.sort_values(
            by="spearman_rho",
            key=lambda x: x.abs(),
            ascending=False,
        ).to_string(index=False)
    )

    print("\nGenerated files:")
    print(f"  {RESULTS_DIR / 'task2_dataset_meta_features.csv'}")
    print(f"  {RESULTS_DIR / 'task2_model_results.csv'}")
    print(f"  {RESULTS_DIR / 'task2_summary.csv'}")
    print(f"  {RESULTS_DIR / 'task2_spearman_correlations.csv'}")
    print(f"  {PLOTS_DIR}/")


if __name__ == "__main__":
    main()