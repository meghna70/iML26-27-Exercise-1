"""Interpretability metrics.

Implement your own metric by subclassing :class:`InterpretabilityMetric`. The metric must be
applicable to every model of the chosen model class (e.g. decision trees, random forests and
gradient-boosted trees), so that scores are comparable across models and datasets.
:class:`RelativeNumberOfLeaves` is a minimal example for a single decision tree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

from iml_interpretability.data import Dataset


class InterpretabilityMetric(ABC):
    """Base class for a metric that quantifies how interpretable a fitted model is."""

    #: Column name under which the score is reported.
    name: str

    @abstractmethod
    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        """Compute the interpretability score of a fitted model.

        :param model: A model fitted on ``dataset``.
        :param dataset: The dataset the model was fitted on.
        :return: The interpretability score. Document whether higher means more or less interpretable.
        :rtype: float
        """


class RelativeNumberOfLeaves(InterpretabilityMetric):
    """Number of leaves of a decision tree relative to the maximum possible for its depth.

    A binary tree of depth ``d`` has at most ``2 ** d`` leaves, so the score lies in ``(0, 1]``, where
    1 means a complete tree. Lower means the tree uses fewer of the decision rules its depth would
    allow. This is only an example: replace it with your own metric, which should go beyond such
    simple size measures, and argue why it captures interpretability.
    """

    name = "relative_n_leaves"

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        if not isinstance(model, DecisionTreeClassifier):
            raise TypeError(f"{type(self).__name__} only supports DecisionTreeClassifier, got {type(model).__name__}")
        return model.get_n_leaves() / 2 ** model.get_depth()
