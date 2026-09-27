"""Evaluate the saved model and write ``reports/evaluation.md`` plus figures.

There are no watch or rating labels in this catalogue. The numbers below are
proxy metrics. Do not read them as accuracy.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import AppConfig, load_config  # noqa: E402
from src.data.clean import clean_frame  # noqa: E402
from src.data.load import load_raw_csv  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    bootstrap_mean_ci,
    coverage_fraction,
    gini_coefficient,
    intra_list_diversity,
    jaccard_at_k,
    precision_at_k,
)
from src.logging_conf import configure_logging  # noqa: E402
from src.models.persistence import load_metadata, load_recommender  # noqa: E402
from src.models.recommender import Recommender, TitleNotFoundError  # noqa: E402


def _lists_at(model: Recommender, indices: np.ndarray, row: int, k: int) -> list[list[str]]:
    return [model.genres[int(idx)] for idx in indices[row, :k] if int(idx) >= 0]


def _per_title(model: Recommender, indices: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    precision = np.empty(len(model.titles), dtype=np.float64)
    jaccard = np.empty(len(model.titles), dtype=np.float64)
    for row in range(len(model.titles)):
        neighbours = _lists_at(model, indices, row, k)
        precision[row] = precision_at_k(model.genres[row], neighbours)
        jaccard[row] = jaccard_at_k(model.genres[row], neighbours)
    return precision, jaccard


def _random_baseline(model: Recommender, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_items = len(model.titles)
    precision = np.empty(n_items, dtype=np.float64)
    jaccard = np.empty(n_items, dtype=np.float64)
    for row in range(n_items):
        banned = {int(idx) for idx in model.same_title_indices[row]}
        drawn: list[int] = []
        while len(drawn) < min(k, n_items - len(banned)):
            pick = int(rng.integers(0, n_items))
            if pick in banned or pick in drawn:
                continue
            drawn.append(pick)
        neighbours = [model.genres[idx] for idx in drawn]
        precision[row] = precision_at_k(model.genres[row], neighbours)
        jaccard[row] = jaccard_at_k(model.genres[row], neighbours)
    return precision, jaccard


def _intra_mean(model: Recommender, indices: np.ndarray, k: int) -> float:
    assert model.matrix is not None
    values: list[float] = []
    for row in range(indices.shape[0]):
        chosen = [int(idx) for idx in indices[row, :k] if int(idx) >= 0]
        if len(chosen) < 2:
            continue
        gram = np.asarray(model.matrix[chosen].dot(model.matrix[chosen].T).toarray())
        values.append(intra_list_diversity(gram))
    if not values:
        return float("nan")
    return float(np.mean(values))


def _fit_variant(frame, config: AppConfig, weights: dict[str, int], kind: str) -> Recommender | str:
    vectorizer = replace(config.vectorizer, kind=kind)
    variant = replace(config, feature_weights=weights, vectorizer=vectorizer)
    try:
        return Recommender(variant).fit(frame)
    except ValueError as exc:
        return str(exc)


def _fmt(value: float) -> str:
    if value != value:  # NaN
        return "n/a"
    return f"{value:.4f}"


def _load_verdicts(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in payload.items()}


def _recommendation_table(model: Recommender, title: str, exclude: bool) -> str:
    try:
        _resolved, results = model.recommend(title, n=5, exclude_same_franchise=exclude)
    except TitleNotFoundError as exc:
        return f"Title `{title}` is not in the catalogue. {exc}"
    rows = []
    for rank, item in enumerate(results, start=1):
        genres = ", ".join(item.genres)
        rows.append(
            f"| {rank} | {item.title} | {item.type} | {item.release_year} | "
            f"{item.similarity:.3f} | {genres} | {item.reason} |"
        )
    header = "| rank | title | type | year | similarity | genres | reason |\n| --- | --- | --- | --- | --- | --- | --- |"
    return header + "\n" + "\n".join(rows)


def _plots(
    figures: Path,
    ks: list[int],
    model_p: dict[int, float],
    base_p: dict[int, float],
    ablation: list[tuple[str, float]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "figure.facecolor": "white",
        }
    )
    figures.mkdir(parents=True, exist_ok=True)

    labels = [f"@{k}" for k in ks]
    x = np.arange(len(ks))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - width / 2, [model_p[k] for k in ks], width, label="Recommender", color="#b20710")
    ax.bar(x + width / 2, [base_p[k] for k in ks], width, label="Random baseline", color="#6b7280")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Genre precision (share of list)")
    ax.set_xlabel("Recommendation list length N")
    ax.set_title("Genre precision versus a random baseline")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "genre_precision_vs_baseline.png", dpi=140)
    plt.close(fig)

    names = [name for name, value in ablation if value == value]
    values = [value for _name, value in ablation if value == value]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    positions = np.arange(len(names))
    ax.barh(positions, values, color="#1f4e79")
    ax.set_yticks(positions, names)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Genre precision@10 (share of list)")
    ax.set_title("Ablation: genre precision@10")
    fig.tight_layout()
    fig.savefig(figures / "ablation_precision.png", dpi=140)
    plt.close(fig)


def _frequency_plot(figures: Path, counts: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(counts, bins=30, color="#1f4e79")
    ax.set_xlabel("Times a title appears in some top-10 list")
    ax.set_ylabel("Number of titles")
    ax.set_title("How concentrated are the recommendations?")
    fig.tight_layout()
    fig.savefig(figures / "recommendation_frequency.png", dpi=140)
    plt.close(fig)


def _latency(model: Recommender, n_queries: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_items = len(model.titles)
    chosen = rng.choice(n_items, size=min(n_queries, n_items), replace=False)
    elapsed = np.empty(len(chosen), dtype=np.float64)
    for position, index in enumerate(chosen):
        # show_id is unique. A display title such as "Esperando la carroza" is not.
        query = str(model.show_ids[int(index)])
        started = time.perf_counter()
        model.recommend(query, n=10, exclude_same_franchise=True)
        elapsed[position] = (time.perf_counter() - started) * 1000
    return float(np.quantile(elapsed, 0.50)), float(np.quantile(elapsed, 0.95))


def _load_time_ms() -> float:
    code = (
        "import time\n"
        "from src.config import load_config\n"
        "from src.models.persistence import load_recommender\n"
        "started = time.perf_counter()\n"
        "config = load_config()\n"
        "load_recommender(config.models_dir, config)\n"
        "print((time.perf_counter() - started) * 1000)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return float(completed.stdout.strip().splitlines()[-1])


def main() -> None:
    configure_logging()
    config = load_config()
    np.random.seed(config.random_seed)
    model = load_recommender(config.models_dir, config)
    raw = load_raw_csv(config.raw_csv)
    cleaned = clean_frame(raw)
    ks = list(config.evaluation.ks)
    max_k = max(ks)
    indices, _scores = model.neighbor_table(max_k, exclude_same_franchise=True)

    model_precision: dict[int, tuple[float, float, float]] = {}
    model_jaccard: dict[int, tuple[float, float, float]] = {}
    base_precision: dict[int, tuple[float, float, float]] = {}
    base_jaccard: dict[int, tuple[float, float, float]] = {}
    for k in ks:
        precision, jaccard = _per_title(model, indices, k)
        model_precision[k] = bootstrap_mean_ci(
            precision, config.evaluation.bootstrap_samples, config.random_seed
        )
        model_jaccard[k] = bootstrap_mean_ci(
            jaccard, config.evaluation.bootstrap_samples, config.random_seed + 1
        )
        random_p, random_j = _random_baseline(model, k, config.random_seed)
        base_precision[k] = bootstrap_mean_ci(
            random_p, config.evaluation.bootstrap_samples, config.random_seed + 2
        )
        base_jaccard[k] = bootstrap_mean_ci(
            random_j, config.evaluation.bootstrap_samples, config.random_seed + 3
        )

    top10 = indices[:, : min(10, indices.shape[1])]
    flat = top10.ravel()
    counts = np.bincount(flat[flat >= 0], minlength=len(model.titles))
    coverage = coverage_fraction(top10, len(model.titles))
    gini = gini_coefficient(counts)
    diversity = _intra_mean(model, indices, min(10, indices.shape[1]))

    weights = dict(config.feature_weights)
    variants: list[tuple[str, dict[str, int], str]] = [
        ("description only", {key: 0 for key in weights} | {"description": 1}, "tfidf"),
        ("genres only", {key: 0 for key in weights} | {"genre": 3}, "tfidf"),
        ("full soup (production)", weights, "tfidf"),
        ("genre weight 1", {**weights, "genre": 1}, "tfidf"),
        ("genre weight 6", {**weights, "genre": 6}, "tfidf"),
        ("country weight 1", {**weights, "country": 1}, "tfidf"),
        ("count vectorizer, full soup", weights, "count"),
    ]
    ablation_rows: list[tuple[str, float, float, str]] = []
    ablation_plot: list[tuple[str, float]] = []
    for name, variant_weights, kind in variants:
        fitted = _fit_variant(cleaned, config, variant_weights, kind)
        if isinstance(fitted, str):
            ablation_rows.append((name, float("nan"), float("nan"), fitted))
            ablation_plot.append((name, float("nan")))
            continue
        variant_index, _variant_scores = fitted.neighbor_table(10, exclude_same_franchise=True)
        precision, jaccard = _per_title(fitted, variant_index, min(10, variant_index.shape[1]))
        p_mean = float(np.nanmean(precision))
        j_mean = float(np.nanmean(jaccard))
        ablation_rows.append((name, p_mean, j_mean, ""))
        ablation_plot.append((name, p_mean))

    p50, p95 = _latency(model, config.evaluation.latency_queries, config.random_seed)
    load_ms = _load_time_ms()

    figures = config.paths.figures_dir
    _plots(
        figures,
        ks,
        {k: model_precision[k][0] for k in ks},
        {k: base_precision[k][0] for k in ks},
        ablation_plot,
    )
    _frequency_plot(figures, counts)

    verdicts = _load_verdicts(config.reports_dir / "qualitative_verdicts.json")
    lines: list[str] = [
        "# Evaluation",
        "",
        "Generated by `scripts/evaluate.py` from the artifacts in `models/`.",
        "Re-run the script to refresh every number in this file.",
        "",
        "## What this is not",
        "",
        "The catalogue has no user interactions, no ratings of preference, and no",
        "held-out relevance labels. **Genre precision is not accuracy.** Genres are",
        "also the strongest feature in the soup, so a high genre precision mostly",
        "shows that the vectoriser did what it was asked to do. It is a sanity check",
        "against a random baseline, not an estimate of whether a person would enjoy",
        "the recommendation. `cast` and `description` are absent from this extract,",
        "so many titles are similar only because they share genre tokens.",
        "",
        "## Genre agreement",
        "",
        "For every title, take the top N recommendations (franchise filter on, which",
        "is the product default) and average over the catalogue. Intervals are",
        f"percentile bootstrap 95% CIs with {config.evaluation.bootstrap_samples} resamples",
        f"and seed {config.random_seed}.",
        "",
        "| N | genre precision | 95% CI | random precision | random 95% CI | Jaccard | Jaccard 95% CI | random Jaccard |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for k in ks:
        mp, mlo, mhi = model_precision[k]
        bp, blo, bhi = base_precision[k]
        mj, mjlo, mjhi = model_jaccard[k]
        bj, _bjlo, _bjhi = base_jaccard[k]
        lines.append(
            f"| {k} | {_fmt(mp)} | {_fmt(mlo)}–{_fmt(mhi)} | {_fmt(bp)} | {_fmt(blo)}–{_fmt(bhi)} | "
            f"{_fmt(mj)} | {_fmt(mjlo)}–{_fmt(mjhi)} | {_fmt(bj)} |"
        )
    lines.extend(
        [
            "",
            "## Coverage and diversity",
            "",
            "Computed from each title's top 10.",
            "",
            f"- coverage (share of the catalogue that appears at least once): **{_fmt(coverage)}**",
            f"- Gini of the recommendation-frequency distribution (0 equal, 1 concentrated): **{_fmt(gini)}**",
            f"- intra-list diversity (mean pairwise cosine dissimilarity inside a top-10): **{_fmt(diversity)}**",
            "",
            "![Recommendation frequency](figures/recommendation_frequency.png)",
            "",
            "![Genre precision versus baseline](figures/genre_precision_vs_baseline.png)",
            "",
            "## Ablation",
            "",
            "Same metric, whole catalogue, franchise filter on. Production weights are",
            "the `feature_weights` block in `config.yaml` (country weight 0). Description-only",
            "has almost nothing to fit because this extract has no `description` column.",
            "",
            "| configuration | genre precision@10 | Jaccard@10 | note |",
            "| --- | --- | --- | --- |",
        ]
    )
    for name, precision_mean, jaccard_mean, note in ablation_rows:
        lines.append(f"| {name} | {_fmt(precision_mean)} | {_fmt(jaccard_mean)} | {note} |")
    lines.extend(
        [
            "",
            "![Ablation](figures/ablation_precision.png)",
            "",
            "The genres-only row is the useful comparison: if it matches the full soup,",
            "the other fields are not moving genre precision (which is expected when genre",
            "precision is computed from the same field the model was shown). Country weight 1",
            "is the sensitivity check for the corrupted Pakistan labels. Count versus TF-IDF",
            "is the second vectoriser. See `docs/DECISIONS.md` for the weight that is shipped.",
            "",
            "## Qualitative audit",
            "",
            "Top 5 with the franchise filter on. Verdicts are stored in",
            "`reports/qualitative_verdicts.json` after reading these lists; they are judgements,",
            "not computed scores.",
            "",
        ]
    )
    for title in config.evaluation.qualitative_titles:
        verdict = verdicts.get(title, "Verdict not written yet.")
        lines.append(f"### {title}")
        lines.append("")
        lines.append(_recommendation_table(model, title, exclude=True))
        lines.append("")
        lines.append(f"Verdict: {verdict}")
        lines.append("")

    lines.extend(
        [
            "## Franchise filter",
            "",
            "Breaking Bad's top 5 does not move when the filter is turned off. Better Call Saul",
            "and El Camino do not share its exact genre triple, so other crime dramas outrank them",
            "before the franchise rule ever runs.",
            "",
            "### Breaking Bad, filter off",
            "",
            _recommendation_table(model, "Breaking Bad", exclude=False),
            "",
            "The filter does change a sequel-heavy title. `Naruto Shippuden: The Movie` shares a",
            "long normalised prefix with `Naruto Shippuden the Movie: Bonds`, so the default",
            "filter drops that near-duplicate. The earlier Naruto films do not share a 10-character",
            "prefix with the Shippuden title, so they stay.",
            "",
            "### Naruto Shippuden: The Movie, filter on",
            "",
            _recommendation_table(model, "Naruto Shippuden: The Movie", exclude=True),
            "",
            "### Naruto Shippuden: The Movie, filter off",
            "",
            _recommendation_table(model, "Naruto Shippuden: The Movie", exclude=False),
            "",
            "## Latency",
            "",
            f"Timed `{config.evaluation.latency_queries}` calls to `recommend()` using randomly",
            "chosen `show_id` values (seed "
            f"{config.random_seed}), franchise filter on, N=10. "
            "show_id is unique; a few display titles are not.",
            "",
            f"- p50: **{p50:.2f} ms**",
            f"- p95: **{p95:.2f} ms**",
            f"- cold artifact load in a fresh process: **{load_ms:.1f} ms**",
            "",
            f"Budget for p95 is {config.evaluation.latency_budget_ms:.0f} ms.",
            "",
        ]
    )

    report_path = config.reports_dir / "evaluation.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    metadata = load_metadata(config.models_dir)
    metadata["metrics"] = {
        "genre_precision_at_5": model_precision[5][0] if 5 in model_precision else None,
        "genre_precision_at_10": model_precision[10][0] if 10 in model_precision else None,
        "genre_precision_at_10_ci95": (
            [model_precision[10][1], model_precision[10][2]] if 10 in model_precision else None
        ),
        "random_genre_precision_at_10": base_precision[10][0] if 10 in base_precision else None,
        "jaccard_at_10": model_jaccard[10][0] if 10 in model_jaccard else None,
        "coverage_at_10": coverage,
        "gini_at_10": gini,
        "intra_list_diversity_at_10": diversity,
        "recommend_p50_ms": p50,
        "recommend_p95_ms": p95,
        "artifact_load_ms": load_ms,
    }
    (config.models_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote {report_path}")
    print(f"genre precision@10 = {model_precision.get(10, (float('nan'),) )[0]:.4f}")
    print(f"random precision@10 = {base_precision.get(10, (float('nan'),) )[0]:.4f}")
    print(f"p95 = {p95:.2f} ms")
    if p95 > config.evaluation.latency_budget_ms:
        print("p95 exceeds the latency budget.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
