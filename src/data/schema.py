"""Hand-rolled catalogue contract.

Pandera is not used. The validator's only job is to fail loudly, with the
column list it actually saw, when a swapped CSV is missing a required field.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "show_id",
    "type",
    "title",
    "director",
    "country",
    "date_added",
    "release_year",
    "rating",
    "duration",
    "listed_in",
)
OPTIONAL_COLUMNS: tuple[str, ...] = ("cast", "description")
ALLOWED_TYPES = frozenset({"Movie", "TV Show"})


class SchemaError(Exception):
    """Raised when the raw catalogue does not satisfy the contract."""


def validate_raw_frame(df: pd.DataFrame) -> None:
    """Raise ``SchemaError`` if required columns or types are unusable."""
    columns = [str(column) for column in df.columns]
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        raise SchemaError(f"column '{missing[0]}' missing — got {columns}")
    if len(df.index) == 0:
        raise SchemaError(f"catalogue has no rows — got {columns}")

    years = pd.to_numeric(df["release_year"], errors="coerce")
    if years.isna().any():
        bad = int(years.isna().sum())
        raise SchemaError(f"column 'release_year' has {bad} non-numeric values — got {columns}")

    observed_types = {str(value).strip() for value in df["type"].dropna().unique()}
    unexpected = sorted(observed_types - ALLOWED_TYPES)
    if unexpected:
        raise SchemaError(
            f"column 'type' has unexpected values {unexpected} — allowed {sorted(ALLOWED_TYPES)}"
        )
