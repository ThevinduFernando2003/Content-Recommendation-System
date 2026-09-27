"""Load ``config.yaml`` and expose a single random seed."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

RANDOM_SEED = 42

_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PathConfig:
    raw_csv: Path
    interim_dir: Path
    processed_path: Path
    models_dir: Path
    reports_dir: Path
    figures_dir: Path


@dataclass(frozen=True)
class VectorizerConfig:
    kind: str
    ngram_range: tuple[int, int]
    min_df: int
    max_features: int
    sublinear_tf: bool


@dataclass(frozen=True)
class RecommenderConfig:
    default_n: int
    cast_cap: int
    fuzzy_score_threshold: float
    fuzzy_ambiguity_gap: float
    fuzzy_suggestion_limit: int
    franchise_prefix_chars: int
    franchise_token_set_ratio: float
    candidate_pool: int
    saturated_similarity: float
    weak_text_similarity: float


@dataclass(frozen=True)
class EvaluationConfig:
    ks: tuple[int, ...]
    bootstrap_samples: int
    latency_queries: int
    latency_budget_ms: float
    qualitative_titles: tuple[str, ...]


@dataclass(frozen=True)
class AppConfig:
    random_seed: int
    paths: PathConfig
    feature_weights: dict[str, int]
    vectorizer: VectorizerConfig
    recommender: RecommenderConfig
    evaluation: EvaluationConfig
    root: Path = field(default=_REPO_ROOT)

    @property
    def raw_csv(self) -> Path:
        return self.paths.raw_csv

    @property
    def reports_dir(self) -> Path:
        return self.paths.reports_dir

    @property
    def models_dir(self) -> Path:
        return self.paths.models_dir


def repo_root() -> Path:
    """Return the repository root (directory that contains ``config.yaml``)."""
    override = os.environ.get("REPO_ROOT")
    if override:
        return Path(override).resolve()
    return _REPO_ROOT


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_config(path: Path | None = None) -> AppConfig:
    """Read YAML config. Relative paths are resolved against the repo root."""
    root = repo_root()
    config_path = path or (root / "config.yaml")
    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    seed = int(raw["random_seed"])
    if seed != RANDOM_SEED:
        # The module-level constant is the canonical seed. The YAML value must match
        # so a reader of either place sees the same number.
        raise ValueError(f"config.yaml random_seed={seed} does not match RANDOM_SEED={RANDOM_SEED}")

    paths = raw["paths"]
    vec = raw["vectorizer"]
    rec = raw["recommender"]
    ev = raw["evaluation"]
    ngram = vec["ngram_range"]
    return AppConfig(
        random_seed=seed,
        paths=PathConfig(
            raw_csv=_resolve(root, paths["raw_csv"]),
            interim_dir=_resolve(root, paths["interim_dir"]),
            processed_path=_resolve(root, paths["processed_path"]),
            models_dir=_resolve(root, os.environ.get("MODELS_DIR", paths["models_dir"])),
            reports_dir=_resolve(root, paths["reports_dir"]),
            figures_dir=_resolve(root, paths["figures_dir"]),
        ),
        feature_weights={str(k): int(v) for k, v in raw["feature_weights"].items()},
        vectorizer=VectorizerConfig(
            kind=str(vec["kind"]),
            ngram_range=(int(ngram[0]), int(ngram[1])),
            min_df=int(vec["min_df"]),
            max_features=int(vec["max_features"]),
            sublinear_tf=bool(vec["sublinear_tf"]),
        ),
        recommender=RecommenderConfig(
            default_n=int(rec["default_n"]),
            cast_cap=int(rec["cast_cap"]),
            fuzzy_score_threshold=float(rec["fuzzy_score_threshold"]),
            fuzzy_ambiguity_gap=float(rec["fuzzy_ambiguity_gap"]),
            fuzzy_suggestion_limit=int(rec["fuzzy_suggestion_limit"]),
            franchise_prefix_chars=int(rec["franchise_prefix_chars"]),
            franchise_token_set_ratio=float(rec["franchise_token_set_ratio"]),
            candidate_pool=int(rec["candidate_pool"]),
            saturated_similarity=float(rec["saturated_similarity"]),
            weak_text_similarity=float(rec["weak_text_similarity"]),
        ),
        evaluation=EvaluationConfig(
            ks=tuple(int(k) for k in ev["ks"]),
            bootstrap_samples=int(ev["bootstrap_samples"]),
            latency_queries=int(ev["latency_queries"]),
            latency_budget_ms=float(ev["latency_budget_ms"]),
            qualitative_titles=tuple(str(t) for t in ev["qualitative_titles"]),
        ),
        root=root,
    )
