"""Fit a TF-IDF or count vectoriser and L2-normalise the rows.

Row-normalised vectors make a later dot product equal cosine similarity, so the
service never materialises an N×N similarity matrix.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from src.config import VectorizerConfig


def make_vectorizer(config: VectorizerConfig) -> TfidfVectorizer | CountVectorizer:
    """Build the configured vectoriser. ``kind`` is ``tfidf`` or ``count``."""
    common: dict[str, Any] = {
        "stop_words": "english",
        "ngram_range": config.ngram_range,
        "min_df": config.min_df,
        "max_features": config.max_features,
        "dtype": np.float32,
    }
    if config.kind == "tfidf":
        return TfidfVectorizer(**common, sublinear_tf=config.sublinear_tf)
    if config.kind == "count":
        return CountVectorizer(**common)
    raise ValueError(f"Unknown vectorizer kind '{config.kind}'. Use 'tfidf' or 'count'.")


def fit_matrix(
    texts: list[str],
    config: VectorizerConfig,
) -> tuple[TfidfVectorizer | CountVectorizer, sparse.csr_matrix]:
    """Fit on ``texts`` and return an L2-normalised float32 CSR matrix."""
    vectorizer = make_vectorizer(config)
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError as exc:
        raise ValueError(
            "Vectorizer produced no vocabulary. The documents are empty or every "
            "token was removed by stop words / min_df."
        ) from exc
    if matrix.shape[1] == 0:
        raise ValueError(
            "Vectorizer produced no vocabulary. The documents are empty or every "
            "token was removed by stop words / min_df."
        )
    matrix = normalize(matrix, norm="l2", axis=1, copy=False)
    matrix = matrix.astype(np.float32).tocsr()
    return vectorizer, matrix


def transform_query(
    vectorizer: TfidfVectorizer | CountVectorizer,
    text: str,
) -> sparse.csr_matrix:
    """Transform one query. An all-zero row means nothing survived the vocabulary."""
    matrix = vectorizer.transform([text])
    if matrix.nnz == 0:
        return matrix.tocsr()
    matrix = normalize(matrix, norm="l2", axis=1, copy=False)
    return matrix.astype(np.float32).tocsr()
