
"""Run the DTCCS interpretability experiment and save results as CSV.

Usage:
    uv run python scripts/run_experiment.py
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from iml_interpretability import evaluate, get_models
from iml_interpretability.data import load_dataset, load_digits_dataset, load_wine_dataset, load_iris_dataset
from iml_interpretability.metric import DecisionTreeComplexityScore


OUTPUT = Path("results/dtccs.csv")
SEEDS = [0, 21, 42, 63, 84]


def main() -> None:

    # Load Breast Cancer Wisconsin dataset
    datasets = [load_dataset(), load_digits_dataset(), load_iris_dataset(), load_wine_dataset()]

    # Initialize proposed interpretability metric
    metric = DecisionTreeComplexityScore()

    all_results = []

    # run evaluation 
    for seed in SEEDS:
        results = evaluate(
            datasets=datasets,
            models=get_models(seed=seed),
            metrics=[metric],
            seed=seed,
            test_size=0.25,
        )

        results["seed"] = seed
        all_results.append(results)


    # Save results

    final_results = pd.concat(all_results, ignore_index=True)
    
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    final_results.to_csv(OUTPUT, index=False)

    print("\n=== DTCCS Experiment Results ===\n")
    print(final_results.to_string(index=False))

    print(f"\nResults saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
