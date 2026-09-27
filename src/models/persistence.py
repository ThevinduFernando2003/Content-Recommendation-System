"""Save and load recommender artifacts plus a metadata sidecar.

The fitted vectoriser is a scikit-learn estimator (importable from sklearn),
the matrix is a scipy CSR ``.npz``, and the catalogue columns are parquet.
Nothing persisted is a lambda or a nested function, so a fresh process can load
the artifacts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from src.config import AppConfig, repo_root
from src.models.recommender import Recommender

logger = logging.getLogger(__name__)

VECTORIZER_NAME = "tfidf_vectorizer.joblib"
MATRIX_NAME = "item_matrix.npz"
INDEX_NAME = "title_index.parquet"
NN_NAME = "nn_index.joblib"
METADATA_NAME = "metadata.json"


def _display_path(path: Path) -> str:
    """Store a repo-relative POSIX path when the file lives inside the repo."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root().resolve()).as_posix()
    except ValueError:
        return str(path)


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, streamed so a large CSV is not held twice in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path | None = None) -> str:
    """Current commit, or ``unknown`` when git is not available."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root or repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def library_versions() -> dict[str, str]:
    import pandas as pandas_module

    return {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "pandas": pandas_module.__version__,
    }


def _frame_for_index(model: Recommender) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "show_id": model.show_ids.astype(str),
            "title": model.titles.astype(str),
            "type": model.types.astype(str),
            "release_year": model.years.astype(int),
            "rating": model.ratings,
            "genres": [json.dumps(items, ensure_ascii=False) for items in model.genres],
            "directors": [json.dumps(items, ensure_ascii=False) for items in model.directors],
            "cast_list": [json.dumps(items, ensure_ascii=False) for items in model.cast_list],
            "countries": [json.dumps(items, ensure_ascii=False) for items in model.countries],
            "description": model.descriptions,
            "date_added": model.date_added,
            "date_added_year": pd.array(
                [pd.NA if year is None else int(year) for year in model.date_added_year],
                dtype="Int64",
            ),
            "duration_minutes": pd.array(
                [pd.NA if value is None else float(value) for value in model.duration_minutes],
                dtype="Float64",
            ),
        }
    )


def save_recommender(
    model: Recommender,
    directory: Path,
    source_csv: Path,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write artifacts and ``metadata.json``. Returns the metadata dict."""
    if model.matrix is None or model.vectorizer is None:
        raise RuntimeError("Cannot save an unfitted recommender.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.vectorizer, directory / VECTORIZER_NAME)
    sparse.save_npz(directory / MATRIX_NAME, model.matrix)
    _frame_for_index(model).to_parquet(directory / INDEX_NAME, index=False)

    # Option B artifact. Request time uses the sparse matvec (option A); this
    # index is persisted so the comparison is reproducible and loadable.
    neighbors = min(11, int(model.matrix.shape[0]))
    nn = NearestNeighbors(n_neighbors=neighbors, metric="cosine", algorithm="brute")
    nn.fit(model.matrix)
    joblib.dump(nn, directory / NN_NAME)

    commit = git_commit()
    metadata: dict[str, Any] = {
        "trained_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": commit,
        "data_sha256": file_sha256(source_csv),
        "source_csv": _display_path(source_csv),
        "row_count": int(model.matrix.shape[0]),
        "n_features": int(model.matrix.shape[1]),
        "nnz": int(model.matrix.nnz),
        "random_seed": model.config.random_seed,
        "vectorizer_kind": model.config.vectorizer.kind,
        "feature_weights": dict(model.config.feature_weights),
        "library_versions": library_versions(),
        "metrics": metrics,
        "similarity": "sparse_dot_l2_normalised",
    }
    (directory / METADATA_NAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    short = commit[:7] if commit != "unknown" else "unknown"
    model.model_version = f"{short}@{metadata['trained_at']}"
    logger.info("Saved artifacts to %s (%s)", directory, model.model_version)
    return metadata


def load_metadata(directory: Path) -> dict[str, Any]:
    path = Path(directory) / METADATA_NAME
    if not path.exists():
        raise FileNotFoundError(f"metadata.json not found in {directory}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_recommender(directory: Path, config: AppConfig) -> Recommender:
    """Load artifacts into a new ``Recommender``. Safe to call in a fresh process."""
    directory = Path(directory)
    vectorizer_path = directory / VECTORIZER_NAME
    matrix_path = directory / MATRIX_NAME
    index_path = directory / INDEX_NAME
    for path in (vectorizer_path, matrix_path, index_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing model artifact: {path}")
    model = Recommender(config)
    model.vectorizer = joblib.load(vectorizer_path)
    model.matrix = sparse.load_npz(matrix_path).tocsr()
    model.feature_names = np.asarray(model.vectorizer.get_feature_names_out())
    frame = pd.read_parquet(index_path)
    for column in ("genres", "directors", "cast_list", "countries"):
        frame[column] = frame[column].map(
            lambda value: json.loads(value) if isinstance(value, str) else []
        )
    model._bind_frame(frame)
    model.fitted = True
    try:
        metadata = load_metadata(directory)
    except FileNotFoundError:
        metadata = {}
    commit = str(metadata.get("git_commit", "unknown"))
    trained_at = str(metadata.get("trained_at", "unknown"))
    short = commit[:7] if commit not in {"", "unknown"} else "unknown"
    model.model_version = f"{short}@{trained_at}"
    logger.info(
        "Loaded recommender %s rows=%s features=%s",
        model.model_version,
        metadata.get("row_count"),
        metadata.get("n_features"),
    )
    return model
