"""Visualize decision tree interpretability experiment results."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Paths
INPUT = Path("results/breast_cancer_dtccs.csv")
OUTPUT_DIR = Path("results/plots")

# Dataset names and their display order
DATASETS = ["breast_cancer", "digits", "iris", "wine"]

# Model configuration order
MODEL_ORDER = [
    "decision_tree_1",
    "decision_tree_3",
    "decision_tree_4",
    "decision_tree_5",
    "decision_tree_7",
    "decision_tree_10",
    "decision_tree_15",
    "decision_tree_unrestricted",
]


def load_results() -> pd.DataFrame:
    """Load experiment results from CSV."""
    if not INPUT.exists():
        raise FileNotFoundError(f"Results file not found: {INPUT}")

    return pd.read_csv(INPUT)


def plot_accuracy_vs_dtccs(df: pd.DataFrame) -> None:
    """
    Plot predictive accuracy against interpretability score.

    Higher DTCCS = greater estimated interpretability.
    """
    plt.figure(figsize=(9, 6))

    for dataset in DATASETS:
        subset = df[df["dataset"] == dataset]

        plt.scatter(
            subset["dtccs"],
            subset["accuracy"] * 100,
            label=dataset.replace("_", " ").title(),
            s=80,
        )

        for _, row in subset.iterrows():
            plt.annotate(
                row["model"].replace("decision_tree_", ""),
                (row["dtccs"], row["accuracy"] * 100),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    plt.xlabel("DTCCS (higher = more interpretable)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Predictive Accuracy vs Interpretability")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "accuracy_vs_dtccs.png", dpi=300)
    plt.show()


def plot_complexity_vs_dtccs(df: pd.DataFrame) -> None:
    """
    Plot tree configurations against DTCCS.

    This shows how the proposed interpretability metric
    responds to increasing tree complexity.
    """
    plt.figure(figsize=(11, 6))

    for dataset in DATASETS:
        subset = df[df["dataset"] == dataset].copy()

        subset["model"] = pd.Categorical(
            subset["model"],
            categories=MODEL_ORDER,
            ordered=True,
        )

        subset = subset.sort_values("model")

        plt.plot(
            subset["model"].astype(str),
            subset["dtccs"],
            marker="o",
            linewidth=2,
            label=dataset.replace("_", " ").title(),
        )

    plt.xlabel("Decision Tree Configuration")
    plt.ylabel("DTCCS (higher = more interpretable)")
    plt.title("Tree Complexity vs Interpretability")
    plt.xticks(rotation=35, ha="right")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "complexity_vs_dtccs.png", dpi=300)
    plt.show()


def plot_complexity_vs_accuracy(df: pd.DataFrame) -> None:
    """
    Plot tree configurations against predictive accuracy.

    This shows whether increasing tree complexity
    improves predictive performance.
    """
    plt.figure(figsize=(11, 6))

    for dataset in DATASETS:
        subset = df[df["dataset"] == dataset].copy()

        subset["model"] = pd.Categorical(
            subset["model"],
            categories=MODEL_ORDER,
            ordered=True,
        )

        subset = subset.sort_values("model")

        plt.plot(
            subset["model"].astype(str),
            subset["accuracy"] * 100,
            marker="o",
            linewidth=2,
            label=dataset.replace("_", " ").title(),
        )

    plt.xlabel("Decision Tree Configuration")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Tree Complexity vs Predictive Accuracy")
    plt.xticks(rotation=35, ha="right")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "complexity_vs_accuracy.png", dpi=300)
    plt.show()


def main() -> None:
    """Generate all experiment visualizations."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results()

    print("\nLoaded experiment results:")
    print(df.head())

    print("\nGenerating visualizations...")

    plot_accuracy_vs_dtccs(df)
    plot_complexity_vs_dtccs(df)
    plot_complexity_vs_accuracy(df)

    print(f"\nAll plots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()