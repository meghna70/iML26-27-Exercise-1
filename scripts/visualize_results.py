"""Visualize decision tree interpretability experiment results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INPUT = Path("results/dtccs.csv")
OUTPUT_DIR = Path("results/plots")

DATASETS = [
    "breast_cancer",
    "digits",
    "iris",
    "wine",
]

DATASET_LABELS = {
    "breast_cancer": "Breast Cancer",
    "digits": "Digits",
    "iris": "Iris",
    "wine": "Wine",
}

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

MODEL_LABELS = {
    "decision_tree_1": "1",
    "decision_tree_3": "3",
    "decision_tree_4": "4",
    "decision_tree_5": "5",
    "decision_tree_7": "7",
    "decision_tree_10": "10",
    "decision_tree_15": "15",
    "decision_tree_unrestricted": "No limit",
}


# ---------------------------------------------------------
# Load and aggregate results
# ---------------------------------------------------------

def load_results() -> pd.DataFrame:
    """Load experiment results from CSV."""

    if not INPUT.exists():
        raise FileNotFoundError(f"Results file not found: {INPUT}")

    df = pd.read_csv(INPUT)

    required_columns = {
        "dataset",
        "model",
        "accuracy",
        "dtccs",
        "seed",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean and standard deviation across random seeds."""

    summary = (
        df.groupby(["dataset", "model"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_dtccs=("dtccs", "mean"),
            std_dtccs=("dtccs", "std"),
        )
    )

    summary[["std_accuracy", "std_dtccs"]] = (
        summary[["std_accuracy", "std_dtccs"]].fillna(0)
    )

    summary["model"] = pd.Categorical(
        summary["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    return summary.sort_values(["dataset", "model"])


# ---------------------------------------------------------
# Shared plotting utilities
# ---------------------------------------------------------

def setup_figure(title: str):
    """Create a consistent 2x2 subplot layout."""

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
    )

    fig.suptitle(title, fontsize=16, fontweight="bold")

    return fig, axes.flatten()


def format_axis(ax):
    """Apply consistent axis formatting."""

    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def get_dataset_results(df, dataset):
    """Get ordered results for one dataset."""

    subset = df[df["dataset"] == dataset].copy()

    subset["model"] = pd.Categorical(
        subset["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    return subset.sort_values("model")


# ---------------------------------------------------------
# 1. Accuracy vs DTCCS
# ---------------------------------------------------------

def plot_accuracy_vs_dtccs(df: pd.DataFrame) -> None:
    """
    Plot predictive accuracy against interpretability.

    Each dataset receives its own subplot.
    Points are labelled by maximum tree depth.
    Vertical error bars represent accuracy variation across seeds.
    """

    fig, axes = setup_figure(
        "Predictive Accuracy vs Interpretability"
    )

    for ax, dataset in zip(axes, DATASETS):

        subset = get_dataset_results(df, dataset)

        x = subset["mean_dtccs"].to_numpy()
        y = (subset["mean_accuracy"] * 100).to_numpy()
        yerr = (subset["std_accuracy"] * 100).to_numpy()

        labels = [
            MODEL_LABELS[m]
            for m in subset["model"].astype(str)
        ]

        # Connect configurations in increasing tree-depth order
        ax.plot(
            x,
            y,
            linestyle="--",
            linewidth=1.2,
            alpha=0.5,
            zorder=1,
        )

        # Plot points with vertical error bars
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            markersize=8,
            capsize=4,
            linewidth=1.5,
            zorder=2,
        )

        # Label each point with its maximum depth
        for i, (xi, yi, label) in enumerate(zip(x, y, labels)):

            # Alternate label positions to reduce overlap
            offsets = [
                (0, 11),
                (0, -17),
                (10, 12),
                (-10, -18),
                (12, 0),
                (-12, 14),
                (12, -15),
                (-12, 0),
            ]

            offset = offsets[i % len(offsets)]

            ax.annotate(
                label,
                (xi, yi),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                ha="center",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc="white",
                    ec="none",
                    alpha=0.75,
                ),
                zorder=3,
            )

        ax.set_title(DATASET_LABELS[dataset], fontweight="bold")
        ax.set_xlabel("Mean DTCCS (higher = more interpretable)")
        ax.set_ylabel("Mean Test Accuracy (%)")
        ax.set_ylim(0, 105)

        format_axis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(
        OUTPUT_DIR / "accuracy_vs_dtccs.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

# ---------------------------------------------------------
# 2. Tree Complexity vs DTCCS
# ---------------------------------------------------------

def plot_complexity_vs_dtccs(df: pd.DataFrame) -> None:
    """
    Plot tree configurations against interpretability.

    Each dataset receives its own subplot.
    """

    fig, axes = setup_figure(
        "Tree Complexity vs Interpretability"
    )

    for ax, dataset in zip(axes, DATASETS):

        subset = get_dataset_results(df, dataset)

        ax.errorbar(
            subset["model"].astype(str).replace(MODEL_LABELS),
            subset["mean_dtccs"],
            yerr=subset["std_dtccs"],
            marker="o",
            markersize=7,
            linewidth=2,
            capsize=4,
        )

        ax.set_title(DATASET_LABELS[dataset], fontweight="bold")
        ax.set_xlabel("Maximum Tree Depth")
        ax.set_ylabel("Mean DTCCS")

        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels(
            [MODEL_LABELS[m] for m in MODEL_ORDER]
        )

        format_axis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(
        OUTPUT_DIR / "complexity_vs_dtccs.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


# ---------------------------------------------------------
# 3. Tree Complexity vs Accuracy
# ---------------------------------------------------------

def plot_complexity_vs_accuracy(df: pd.DataFrame) -> None:
    """
    Plot tree configurations against predictive accuracy.

    Each dataset receives its own subplot.
    """

    fig, axes = setup_figure(
        "Tree Complexity vs Predictive Accuracy"
    )

    for ax, dataset in zip(axes, DATASETS):

        subset = get_dataset_results(df, dataset)

        ax.errorbar(
            subset["model"].astype(str).replace(MODEL_LABELS),
            subset["mean_accuracy"] * 100,
            yerr=subset["std_accuracy"] * 100,
            marker="o",
            markersize=7,
            linewidth=2,
            capsize=4,
        )

        ax.set_title(DATASET_LABELS[dataset], fontweight="bold")
        ax.set_xlabel("Maximum Tree Depth")
        ax.set_ylabel("Mean Test Accuracy (%)")
        ax.set_ylim(0, 100)

        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels(
            [MODEL_LABELS[m] for m in MODEL_ORDER]
        )

        format_axis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(
        OUTPUT_DIR / "complexity_vs_accuracy.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    """Generate all experiment visualizations."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results()

    print("\nLoaded experiment results:")
    print(df.head())

    print("\nExperiments per dataset:")
    print(df.groupby("dataset").size())

    print("\nAggregating results across random seeds...")

    summary = aggregate_results(df)

    summary.to_csv(
        OUTPUT_DIR / "aggregated_results.csv",
        index=False,
    )

    print("\nGenerating visualizations...")

    plot_accuracy_vs_dtccs(summary)
    plot_complexity_vs_dtccs(summary)
    plot_complexity_vs_accuracy(summary)

    print(f"\nAll plots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()