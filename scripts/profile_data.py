"""Profile the raw Netflix catalogue and write ``reports/data_profile.md``.

This script is the source of every count in that report. Re-run it after
replacing ``data/raw/netflix_titles.csv``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

# Allow ``python scripts/profile_data.py`` from the repo root without install.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import RANDOM_SEED, load_config  # noqa: E402
from src.data.repairs import EXCEL_TITLE_BY_SHOW_ID  # noqa: E402
from src.logging_conf import configure_logging  # noqa: E402

# Spec table the file was expected to match. The file on disk wins on conflict.
EXPECTED_COLUMNS = [
    "show_id",
    "type",
    "title",
    "director",
    "cast",
    "country",
    "date_added",
    "release_year",
    "rating",
    "duration",
    "listed_in",
    "description",
]
DURATION_MIN_RE = re.compile(r"^\d+\s*min$", re.IGNORECASE)
DURATION_SEASON_RE = re.compile(r"^\d+\s*seasons?$", re.IGNORECASE)
RATING_DURATION_RE = re.compile(r"^\d+\s*min$", re.IGNORECASE)


def _md_table(headers: list[str], rows: list[list[object]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def _series_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def build_profile(df: pd.DataFrame) -> str:
    """Return the markdown report for a raw catalogue frame."""
    n_rows, n_cols = df.shape
    columns = [str(c) for c in df.columns]
    missing_expected = [c for c in EXPECTED_COLUMNS if c not in columns]
    unexpected = [c for c in columns if c not in EXPECTED_COLUMNS]

    null_rows: list[list[object]] = []
    for col in columns:
        nulls = int(df[col].isna().sum())
        # Treat blank strings as missing for the text fields the spec calls out.
        blanks = int(_series_text(df[col]).eq("").sum()) if df[col].dtype == object else nulls
        null_rows.append(
            [f"`{col}`", str(df[col].dtype), nulls, f"{100.0 * nulls / n_rows:.2f}%", blanks]
        )

    dup_show = int(df["show_id"].duplicated().sum()) if "show_id" in df.columns else "n/a"
    if {"title", "type", "release_year"}.issubset(df.columns):
        dup_title = int(df.duplicated(subset=["title", "type", "release_year"]).sum())
    else:
        dup_title = "n/a"

    cat_rows: list[list[object]] = []
    for col in columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            cat_rows.append([f"`{col}`", int(df[col].nunique(dropna=True))])

    type_values = sorted(df["type"].dropna().astype(str).unique()) if "type" in df.columns else []
    rating_values = (
        sorted(df["rating"].dropna().astype(str).unique(), key=str)
        if "rating" in df.columns
        else []
    )

    genre_counts = pd.Series(dtype=int)
    if "listed_in" in df.columns:
        genres = (
            df["listed_in"]
            .dropna()
            .astype(str)
            .str.split(",")
            .explode()
            .str.strip()
            .replace("", pd.NA)
            .dropna()
        )
        genre_counts = genres.value_counts()

    year_section = "_Column `release_year` is absent._"
    if "release_year" in df.columns:
        years = pd.to_numeric(df["release_year"], errors="coerce").dropna()
        quantiles = years.quantile([0, 0.25, 0.5, 0.75, 1.0])
        year_section = "\n".join(
            [
                f"- non-null count: {int(years.shape[0])}",
                f"- min: {int(quantiles.loc[0.0])}",
                f"- p25: {quantiles.loc[0.25]:.1f}",
                f"- p50: {quantiles.loc[0.5]:.1f}",
                f"- p75: {quantiles.loc[0.75]:.1f}",
                f"- max: {int(quantiles.loc[1.0])}",
            ]
        )

    duration_fail_n = 0
    duration_fail_examples: list[str] = []
    if "duration" in df.columns:
        duration = _series_text(df["duration"])
        parsed = duration.str.match(DURATION_MIN_RE) | duration.str.match(DURATION_SEASON_RE)
        # Blank duration is a parse failure: there is nothing to put in either unit column.
        failed = ~parsed
        duration_fail_n = int(failed.sum())
        duration_fail_examples = duration[failed].head(8).tolist()

    rating_bad_n = 0
    rating_bad_examples: list[list[object]] = []
    if "rating" in df.columns:
        rating = _series_text(df["rating"])
        bad = rating.str.match(RATING_DURATION_RE)
        rating_bad_n = int(bad.sum())
        if rating_bad_n and "show_id" in df.columns:
            preview = df.loc[
                bad, [c for c in ("show_id", "title", "rating", "duration") if c in df.columns]
            ]
            for _, row in preview.head(10).iterrows():
                rating_bad_examples.append([str(row[c]) for c in preview.columns])

    lines: list[str] = [
        "# Data profile",
        "",
        "Generated by `scripts/profile_data.py`. Every figure below is computed from",
        "`data/raw/netflix_titles.csv`. Re-run the script to refresh this file.",
        "",
        "## Shape",
        "",
        f"- rows: **{n_rows}**",
        f"- columns: **{n_cols}**",
        "",
        "### Dtypes inferred by pandas",
        "",
        _md_table(["column", "dtype"], [[f"`{c}`", str(df[c].dtype)] for c in columns]),
        "",
        "## Contract comparison",
        "",
        "The build brief expected 12 columns (including `cast` and `description`) and about",
        "8,791 rows. **The file on disk wins** wherever it disagrees.",
        "",
        _md_table(
            ["check", "result"],
            [
                [
                    "expected columns missing",
                    ", ".join(f"`{c}`" for c in missing_expected) or "none",
                ],
                ["unexpected columns", ", ".join(f"`{c}`" for c in unexpected) or "none"],
                ["row count vs ~8791", f"{n_rows} (delta {n_rows - 8791})"],
            ],
        ),
        "",
    ]
    if missing_expected:
        lines.extend(
            [
                "### What the mismatch changes",
                "",
                "The pipeline treats `cast` and `description` as optional. When they are absent",
                "they are filled with an empty string for text features and are **not** replaced",
                "with the token `Unknown`. Similarity then rests on `listed_in`, `director`,",
                "`country`, `type` and `rating`. Cold-start text can only match tokens that",
                "actually occur in those fields.",
                "",
            ]
        )

    lines.extend(
        [
            "## Missing values",
            "",
            "`null` is pandas NA. `blank` counts empty strings after strip (equal to null for",
            "numeric columns).",
            "",
            _md_table(["column", "dtype", "nulls", "null %", "blank or null"], null_rows),
            "",
            "## Duplicates",
            "",
            f"- duplicate `show_id` rows (beyond the first): **{dup_show}**",
            f"- duplicate `(title, type, release_year)` rows (beyond the first): **{dup_title}**",
            "",
            "## Categorical cardinality",
            "",
            _md_table(["column", "nunique (dropna)"], cat_rows),
            "",
            "### Full value list: `type`",
            "",
            ", ".join(f"`{v}`" for v in type_values) or "_absent_",
            "",
            "### Full value list: `rating`",
            "",
            ", ".join(f"`{v}`" for v in rating_values) or "_absent_",
            "",
            "## Genre vocabulary (`listed_in` exploded on comma)",
            "",
            f"- distinct genres after strip: **{int(genre_counts.shape[0])}**",
            "",
        ]
    )
    if not genre_counts.empty:
        lines.append(
            _md_table(
                ["genre", "count"],
                [[genre, int(count)] for genre, count in genre_counts.items()],
            )
        )
        lines.append("")

    lines.extend(
        [
            "## `release_year`",
            "",
            year_section,
            "",
            "## Duration parse",
            "",
            "A duration parses when it matches `^\\d+\\s*min$` or `^\\d+\\s*seasons?$`",
            "(case-insensitive). Anything else, including blanks, is a parse failure.",
            "",
            f"- rows whose `duration` fails to parse: **{duration_fail_n}**",
            "",
        ]
    )
    if duration_fail_examples:
        lines.append("Examples of failing `duration` values:")
        lines.append("")
        for example in duration_fail_examples:
            shown = example if example else "(blank)"
            lines.append(f"- `{shown}`")
        lines.append("")

    lines.extend(
        [
            "## Rating values that look like a duration",
            "",
            "Pattern: `^\\d+\\s*min$`. These are the shifted-column defect: the duration",
            "landed in `rating`.",
            "",
            f"- rows: **{rating_bad_n}**",
            "",
        ]
    )
    if rating_bad_examples:
        headers = [c for c in ("show_id", "title", "rating", "duration") if c in df.columns]
        lines.append(_md_table(headers, rating_bad_examples))
        lines.append("")

    if "date_added" in df.columns:
        samples = _series_text(df["date_added"])
        samples = samples[samples.ne("")].head(5).tolist()
        lines.extend(
            [
                "## `date_added` samples",
                "",
                "The brief expected dates like `September 25, 2021`. Observed non-null samples:",
                "",
            ]
        )
        for sample in samples:
            lines.append(f"- `{sample}`")
        lines.append("")

    lines.extend(
        [
            "## Profiling seed",
            "",
            f"No random sampling is used in this report. `RANDOM_SEED` is `{RANDOM_SEED}`",
            "and is reserved for training and evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def extra_profile(df: pd.DataFrame) -> str:
    """Sentinels, date formats, and extract-time repairs. Counts are computed here."""
    n_rows = len(df.index)
    sentinels = {"not given", "n/a", "na", "none", "null", "nan"}
    sentinel_rows: list[list[object]] = []
    for col in df.columns:
        if not (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])):
            continue
        text = df[col].fillna("").astype(str).str.strip().str.casefold()
        count = int(text.isin(sentinels).sum())
        if count:
            sentinel_rows.append([f"`{col}`", count, f"{100.0 * count / n_rows:.2f}%"])

    lines = [
        "## Missingness sentinels",
        "",
        "Pandas null counts above are not the whole story. This extract writes missing",
        "director and country as the literal `Not Given` (and similar tokens).",
        "",
    ]
    if sentinel_rows:
        lines.append(_md_table(["column", "sentinel rows", "percent"], sentinel_rows))
    else:
        lines.append("No sentinel tokens found.")
    lines.append("")

    if "date_added" in df.columns:
        text = df["date_added"].fillna("").astype(str).str.strip()
        buckets = {
            "M/D/YYYY": int(text.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$").sum()),
            "ISO date": int(text.str.match(r"^\d{4}-\d{2}-\d{2}$").sum()),
            "ISO datetime": int(text.str.match(r"^\d{4}-\d{2}-\d{2} ").sum()),
            "Month name": int(text.str.match(r"^[A-Za-z]+ \d{1,2}, \d{4}$").sum()),
            "blank": int(text.eq("").sum()),
        }
        other = n_rows - sum(buckets.values())
        lines.extend(["## `date_added` format mix", ""])
        lines.append(
            _md_table(
                ["format", "rows"],
                [[name, count] for name, count in buckets.items()] + [["other", other]],
            )
        )
        lines.append("")

    if {"title", "country", "listed_in"}.issubset(df.columns):
        korean = df["listed_in"].fillna("").astype(str).str.contains("Korean", regex=False)
        pakistan = df["country"].fillna("").astype(str).str.strip().eq("Pakistan")
        lines.extend(
            [
                "## Country field",
                "",
                f"- rows whose `country` contains a comma (multi-country): "
                f"**{int(df['country'].fillna('').astype(str).str.contains(',').sum())}**",
                f"- rows labelled `Pakistan`: **{int(pakistan.sum())}**",
                f"- of which `listed_in` contains `Korean`: **{int((korean & pakistan).sum())}**",
                "",
                "Multi-country values are absent in this extract. `Pakistan` is also attached to",
                "titles whose genres are Korean, Spanish-language, or anime, so the country token",
                "is not a reliable similarity feature. The training config sets its weight to 0;",
                "see `docs/DECISIONS.md`.",
                "",
            ]
        )

    repairs = Path(__file__).resolve().parents[1] / "data" / "interim" / "extract_repairs.json"
    if repairs.exists():
        payload = json.loads(repairs.read_text(encoding="utf-8"))
        lines.extend(
            [
                "## Extract repairs",
                "",
                "Computed by `scripts/prepare_raw.py` while reading `Dataset.csv.xlsx`.",
                "The CSV is what training reads; this section records what Excel had done to it.",
                "",
                _md_table(
                    ["repair", "count"],
                    [[key, payload[key]] for key in payload if isinstance(payload[key], int)],
                ),
                "",
            ]
        )
        title_fixes = payload.get("titles", [])
        coerced = [row for row in title_fixes if row and row[0] in EXCEL_TITLE_BY_SHOW_ID]
        mojibake = len(title_fixes) - len(coerced)
        lines.append(
            f"Title cells rewritten: **{len(title_fixes)}** "
            f"({len(coerced)} Excel type coercions, {mojibake} mojibake repairs). "
            "The table below lists only the coerced titles. Mojibake repairs are UTF-8 "
            "bytes that Excel stored as Latin-1; `scripts/prepare_raw.py` reverses them."
        )
        lines.append("")
        if coerced:
            lines.append(_md_table(["show_id", "excel value", "restored title"], coerced))
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    configure_logging()
    cfg = load_config()
    csv_path = cfg.raw_csv
    if not csv_path.exists():
        print(
            f"Raw CSV not found at {csv_path}. Place the catalogue there and re-run.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    df = pd.read_csv(csv_path)
    report = build_profile(df) + "\n" + extra_profile(df)
    out = cfg.reports_dir / "data_profile.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
