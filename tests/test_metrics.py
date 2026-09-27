"""Proxy-metric helpers stay honest on tiny inputs."""

import numpy as np

from src.evaluation.metrics import (
    bootstrap_mean_ci,
    coverage_fraction,
    gini_coefficient,
    intra_list_diversity,
    jaccard,
    precision_at_k,
)


def test_precision_and_jaccard() -> None:
    assert precision_at_k(["Dramas", "Comedies"], [["Dramas"], ["Horror Movies"]]) == 0.5
    assert jaccard(["Dramas"], ["Dramas", "Comedies"]) == 0.5
    assert jaccard([], []) == 0.0


def test_gini_and_coverage() -> None:
    assert gini_coefficient(np.array([1, 1, 1, 1])) == 0.0
    assert gini_coefficient(np.array([0, 0, 0, 4])) > 0.5
    assert coverage_fraction(np.array([[0, 1], [1, -1]]), 4) == 0.5


def test_bootstrap_is_deterministic() -> None:
    values = np.array([0.2, 0.4, 0.6, 0.8])
    first = bootstrap_mean_ci(values, n_boot=50, seed=42)
    second = bootstrap_mean_ci(values, n_boot=50, seed=42)
    assert first == second
    assert first[1] <= first[2]


def test_intra_list_diversity_of_identical_vectors() -> None:
    identical = np.ones((3, 3))
    assert intra_list_diversity(identical) == 0.0
