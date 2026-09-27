"""Proxy metrics. This dataset has no watch labels, so none of these are accuracy."""

from __future__ import annotations

import numpy as np


def genre_hit(query_genres: list[str], recommended_genres: list[str]) -> float:
    """1 when the recommendation shares at least one genre with the query."""
    if not recommended_genres and not query_genres:
        return 0.0
    return 1.0 if set(query_genres) & set(recommended_genres) else 0.0


def jaccard(left: list[str], right: list[str]) -> float:
    """Jaccard similarity of two genre sets. Two empty sets score 0 (no evidence)."""
    a = set(left)
    b = set(right)
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def precision_at_k(query_genres: list[str], neighbour_genres: list[list[str]]) -> float:
    """Fraction of neighbours that share at least one genre."""
    if not neighbour_genres:
        return float("nan")
    hits = [genre_hit(query_genres, genres) for genres in neighbour_genres]
    return float(np.mean(hits))


def jaccard_at_k(query_genres: list[str], neighbour_genres: list[list[str]]) -> float:
    if not neighbour_genres:
        return float("nan")
    return float(np.mean([jaccard(query_genres, genres) for genres in neighbour_genres]))


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap interval. ``values`` are per-title scores."""
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for draw in range(n_boot):
        sample = clean[rng.integers(0, clean.size, size=clean.size)]
        means[draw] = float(sample.mean())
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(clean.mean()), float(low), float(high)


def gini_coefficient(counts: np.ndarray) -> float:
    """Gini of a non-negative frequency vector. 0 is equal, 1 is concentrated."""
    array = np.asarray(counts, dtype=np.float64)
    if array.size == 0 or np.all(array == 0):
        return 0.0
    array = np.sort(array)
    n = array.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * array) / (n * np.sum(array))) - (n + 1) / n)


def coverage_fraction(recommended_ids: np.ndarray, catalogue_size: int) -> float:
    """Fraction of the catalogue that appears at least once in the id array."""
    if catalogue_size <= 0:
        return 0.0
    valid = recommended_ids[recommended_ids >= 0]
    if valid.size == 0:
        return 0.0
    return float(np.unique(valid).size / catalogue_size)


def intra_list_diversity(pairwise_cosine: np.ndarray) -> float:
    """Mean pairwise dissimilarity (1 - cosine) of the upper triangle."""
    matrix = np.asarray(pairwise_cosine, dtype=np.float64)
    if matrix.shape[0] < 2:
        return float("nan")
    tri = matrix[np.triu_indices(matrix.shape[0], k=1)]
    if tri.size == 0:
        return float("nan")
    dissimilarity = 1.0 - np.clip(tri, 0.0, 1.0)
    return float(np.mean(dissimilarity))
