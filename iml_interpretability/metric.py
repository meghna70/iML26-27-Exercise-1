
"""Interpretability metrics.

DTCCS (Decision Tree Complexity Score) measures the cognitive
complexity of a decision tree using three components:

D: Average decision-path length.
B: Normalized entropy of the distribution of samples across leaves.
S: Feature switching frequency along decision paths.

Scores range from 0 to 100.
Higher scores indicate greater interpretability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

from iml_interpretability.data import Dataset


class InterpretabilityMetric(ABC):
    """Base class for interpretability metrics."""

    name: str

    @abstractmethod
    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        """Calculate interpretability score. Higher means more interpretable."""


class DecisionTreeComplexityScore(InterpretabilityMetric):
    """Decision Tree Complexity Score (DTCCS).

    Components:
        D: Average decision-path length.
        B: Normalized entropy across terminal leaves.
        S: Average feature switching frequency.

    Higher scores indicate greater interpretability.
    Score range: [0, 100].
    """

    name = "dtccs"

    # Normalization references
    DEPTH_REFERENCE = 10

    # Component weights
    WEIGHTS = {
        "depth": 0.4,
        "branching": 0.3,
        "switching": 0.3, #tested out different weights and found that this combination gives the best results
    }

    def components(
        self,
        model: ClassifierMixin,
        dataset: Dataset,
    ) -> dict[str, float]:

        if not isinstance(model, DecisionTreeClassifier):
            raise TypeError(
                f"DTCCS only supports DecisionTreeClassifier, "
                f"got {type(model).__name__}"
            )

        X = dataset.X

        if len(X) == 0:
            raise ValueError("Dataset cannot be empty.")

        tree = model.tree_

        # ------------------------------------------------
        # 1. Average decision-path length (D)
        # ------------------------------------------------

        decision_paths = model.decision_path(X)

        path_lengths = []
        switch_ratios = []

        for i in range(decision_paths.shape[0]):

            start = decision_paths.indptr[i]
            end = decision_paths.indptr[i + 1]

            node_indices = decision_paths.indices[start:end]

            # Extract splitting features along the path.
            # Leaf nodes have feature = -2.
            features = [
                tree.feature[node]
                for node in node_indices
                if tree.feature[node] >= 0
            ]

            path_lengths.append(len(features))

            # --------------------------------------------
            # 2. Feature switching frequency (S)
            # --------------------------------------------

            if len(features) > 1:

                switches = sum(
                    features[j] != features[j - 1]
                    for j in range(1, len(features))
                )

                switch_ratio = switches / (len(features) - 1)

            else:
                switch_ratio = 0.0

            switch_ratios.append(switch_ratio)

        avg_path_length = float(np.mean(path_lengths))

        # Normalize average path length to [0, 1].
        D = min(
            avg_path_length / self.DEPTH_REFERENCE,
            1.0
        )

        # Average proportion of consecutive splits
        # that use different features.
        S = float(np.mean(switch_ratios))

        # --------------------------------------------
        # 3. Branching complexity (B)
        # --------------------------------------------

        leaf_ids = model.apply(X)

        _, counts = np.unique(
            leaf_ids,
            return_counts=True
        )

        probabilities = counts / counts.sum()

        # Shannon entropy.
        entropy = -np.sum(
            probabilities * np.log2(probabilities)
        )

        # Normalize by the maximum possible entropy
        # given the total number of leaves in the tree.
        n_leaves = model.get_n_leaves()

        if n_leaves > 1:
            B = float(entropy / np.log2(n_leaves))
        else:
            B = 0.0

        B = min(max(B, 0.0), 1.0)

        return {
            "depth_complexity": D,
            "branching_complexity": B,
            "feature_switching": S,
        }

    def score(
        self,
        model: ClassifierMixin,
        dataset: Dataset,
    ) -> float:

        components = self.components(model, dataset)

        # Weighted complexity.
        complexity = (
            self.WEIGHTS["depth"] * components["depth_complexity"]
            + self.WEIGHTS["branching"] * components["branching_complexity"]
            + self.WEIGHTS["switching"] * components["feature_switching"]
        )

        # Convert complexity into interpretability.
        return float(100 * (1 - complexity))
