# iML 26/27 – Exercise 1: Quantifying Interpretability

This project investigates how the structural complexity of decision trees affects their interpretability. It introduces a custom metric, the **Decision Tree Complexity Score (DTCCS)**, and evaluates its behaviour across multiple datasets and tree configurations.

The project addresses both Task 1 (developing an interpretability metric) and Task 2 (investigating relationships between dataset meta-features and interpretability).

## 1. Task 1 – Decision Tree Complexity Score (DTCCS)

### 1.1 Motivation

Decision trees are considered inherently interpretable because their predictions can be explained through sequences of feature-based decisions. However, as a tree grows in depth and structural complexity, understanding its overall decision logic becomes increasingly difficult.

Common measures such as tree depth, number of leaves, or number of features used capture only individual aspects of complexity. Therefore, this project proposes a composite metric that considers three structural characteristics of a decision tree.

### 1.2 Proposed metric

The **Decision Tree Complexity Score (DTCCS)** is designed specifically for `sklearn.tree.DecisionTreeClassifier`.

It combines three components:

**1. Average Decision Path Length (D)**

Measures the average number of decision nodes visited when making predictions.

Longer paths require users to process more sequential decisions.

The component is normalized using a reference depth of 10:

\[
D = \min\left(\frac{\text{Average Path Length}}{10}, 1\right)
\]

**2. Branching Complexity (B)**

Measures how evenly the dataset's samples are distributed across the tree's leaves using Shannon entropy.

\[
H = -\sum_{i=1}^{L} p_i \log_2(p_i)
\]

\[
B = \frac{H}{\log_2(L)}
\]

Here, \(p_i\) represents the proportion of samples reaching leaf \(i\), and \(L\) is the number of leaves. For a tree with only one leaf, \(B=0\).

A higher value indicates a more evenly distributed set of active decision paths.

**3. Feature Switching (S)**

Measures how frequently the decision feature changes between consecutive nodes along a prediction path.

\[
S = \frac{\text{Number of Feature Switches}}
{\text{Number of Consecutive Feature Comparisons}}
\]

The score is averaged across all evaluated samples. Paths with fewer than two decision nodes have a switching score of zero.

**Final DTCCS formula**

\[
\boxed{
DTCCS = 100 \left(1 - (0.4D + 0.3B + 0.3S)\right)
}
\]

The weights are design choices that assign 40% importance to path length and 30% each to branching complexity and feature switching.

**Interpretation:**

- Higher DTCCS indicates lower estimated structural complexity and greater interpretability.
- Lower DTCCS indicates greater estimated structural complexity.
- The score ranges from 0 to 100.

DTCCS is a proposed structural proxy for interpretability, not a validated measurement of human cognitive effort.

### 1.3 Experimental evaluation

The metric is evaluated using two classification datasets from scikit-learn:

| Dataset | Samples | Features | Classes |
|---|---:|---:|---:|
| Breast Cancer Wisconsin | 569 | 30 | 2 |
| Digits | 1,797 | 64 | 10 |

Decision trees are trained with different maximum depth configurations:

- 1
- 3
- 4
- 5
- 7
- 10
- 15
- Unrestricted

The experiments examine the relationship between predictive accuracy and DTCCS, as well as how the score changes as tree complexity increases.

### 1.4 Investigating the interpretability threshold

The experiments explore whether increasing tree complexity leads to a point where a decision tree becomes difficult to interpret.

The analysis considers changes in DTCCS alongside tree depth, number of leaves, and predictive accuracy.

The threshold should be understood as context-dependent rather than universal. A formal human evaluation is needed to establish whether the proposed score corresponds to people's actual ability to understand and follow tree decisions.

---

## 2. Task 2 – Dataset Meta-features and Interpretability

### 2.1 Objective

Task 2 investigates whether characteristics of a dataset are associated with the interpretability of decision trees trained on it.

The same DTCCS metric from Task 1 is used to measure interpretability.

### 2.2 Datasets

Four classification datasets are included:

| Dataset | Samples | Features | Classes |
|---|---:|---:|---:|
| Iris | 150 | 4 | 3 |
| Wine | 178 | 13 | 3 |
| Breast Cancer Wisconsin | 569 | 30 | 2 |
| Digits | 1,797 | 64 | 10 |

### 2.3 Extracted meta-features

The following dataset characteristics are extracted:

| Meta-feature | Description |
|---|---|
| Number of samples | Total number of observations |
| Number of features | Number of input variables |
| Number of classes | Number of target classes |
| Samples per feature | Ratio of samples to features |
| Class entropy | Shannon entropy of the class distribution |
| Normalized class entropy | Class entropy normalized by the maximum possible entropy |
| Majority class ratio | Proportion of samples belonging to the largest class |
| Minority class ratio | Proportion of samples belonging to the smallest class |
| Imbalance ratio | Ratio of the largest to the smallest class |
| Mean absolute feature correlation | Average absolute pairwise feature correlation |
| Feature sparsity | Fraction of feature values equal to zero |

### 2.4 Experimental procedure

For each dataset:

1. Extract its meta-features.
2. Split the data into training and test sets using stratified sampling (75:25).
3. Train decision trees using maximum depths of 1, 3, 5, 7, 10, and unrestricted.
4. Repeat the experiments with five random seeds: 0, 21, 42, 63, and 84.
5. Evaluate predictive accuracy and DTCCS on the test set.
6. Calculate mean DTCCS and accuracy for each dataset and tree configuration.

### 2.5 Statistical analysis

Spearman rank correlation is used to explore associations between dataset meta-features and mean DTCCS.

Correlations are calculated separately for each tree configuration, using one observation per dataset.

Scatter plots are generated to visualize relationships between selected meta-features and DTCCS.

Since only four datasets are included, the correlation analysis is exploratory. The results cannot establish causality or statistically reliable general relationships. Additional datasets and controlled experiments would be needed for stronger conclusions.

---

## 3. Reproducing the Experiments

### Requirements

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/)
- Git

### Installation

Clone your repository and navigate to the project directory:

```bash
git clone <your-repository-url>
cd iML26-27-Exercise-1
```

Install the project dependencies:

```bash
uv sync --extra dev
```

### Run Task 1

Execute the Task 1 experiment:

```bash
uv run python scripts/run_experiment.py
```

This generates the Task 1 experiment results.

### Run Task 2

Execute the Task 2 experiments:

```bash
uv run python scripts/run_task2.py
```

This extracts dataset meta-features, trains decision trees across datasets and random seeds, evaluates DTCCS, and performs correlation analysis.

### Generate visualizations

Generate the Task 1 plots:

```bash
uv run python scripts/visualize_results.py
```

Generate the Task 2 plots:

```bash
uv run python scripts/visualize_task2.py
```

The generated CSV files and plots are saved in the `results/` directory.

### Run tests

```bash
uv run pytest
```

---

## 4. Repository Structure

```text
iML26-27-Exercise-1/
│
├── iml_interpretability/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── metric.py
│   ├── evaluation.py
│   └── meta_features.py
│
├── scripts/
│   ├── run_experiment.py
│   ├── visualize_results.py
│   ├── run_task2.py
│   └── visualize_task2.py
│
├── notebooks/
│   └── interpretability_analysis.ipynb
│
├── results/
│   ├── breast_cancer_dtccs.csv
│   ├── task2_dataset_meta_features.csv
│   ├── task2_model_results.csv
│   ├── task2_summary.csv
│   ├── task2_spearman_correlations.csv
│   ├── plots/
│   └── task2_plots/
│
├── presentation/
│   └── presentation.pdf
│
├── tests/
│
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 5. Limitations and Future Work

The proposed DTCCS metric has several limitations:

- The reference depth of 10 and component weights are manually selected design choices.
- Feature switching is assumed to contribute to cognitive complexity, but this assumption requires human validation.
- Branching entropy measures the distribution of samples across leaves rather than directly measuring the difficulty of understanding a tree.
- The metric does not account for feature semantics, domain knowledge, or the difficulty of understanding individual split conditions.
- The experiments use a limited number of datasets and do not establish a universal interpretability threshold.

Future work could include human evaluation studies, sensitivity analysis of the metric's weights and normalization, and experiments with a larger and more diverse collection of datasets.

---

## 6. AI Statement

AI tools were used during the development of this project, primarily ChatGPT (OpenAI). They assisted with brainstorming and refining the proposed metric, understanding relevant machine learning and interpretability concepts, and debugging Python code.

The proposed DTCCS formulation, experimental choices, and interpretation of the results were reviewed and refined as part of the project work. Generated code and explanations were checked against the project requirements and the implemented experiments. AI-generated suggestions were not treated as automatically correct.

I take responsibility for reviewing the submitted code, results, analysis, and presentation, and for the final content of this repository.