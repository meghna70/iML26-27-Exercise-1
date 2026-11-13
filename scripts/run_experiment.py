
"""Run the DTCCS interpretability experiment and save results as CSV.

Usage:
    uv run python scripts/run_experiment.py
"""

from __future__ import annotations

from pathlib import Path

from iml_interpretability import evaluate, get_models
from iml_interpretability.data import load_dataset, load_digits_dataset, load_wine_dataset, load_iris_dataset
from iml_interpretability.metric import DecisionTreeComplexityScore


OUTPUT = Path("results/breast_cancer_dtccs.csv")
SEED = 42


def main() -> None:

    # Load Breast Cancer Wisconsin dataset
    datasets = [load_dataset(), load_digits_dataset(), load_iris_dataset(), load_wine_dataset()]

    # Initialize proposed interpretability metric
    metric = DecisionTreeComplexityScore()

    # Run evaluation
    results = evaluate(
        datasets=datasets,
        models=get_models(seed=SEED),
        metrics=[metric],
        seed=SEED,
        test_size=0.25,
    )

    # Save results
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)

    print("\n=== DTCCS Experiment Results ===\n")
    print(results.to_string(index=False))

    print(f"\nResults saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
