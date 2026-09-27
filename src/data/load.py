"""Read the raw catalogue and enforce the schema."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.schema import OPTIONAL_COLUMNS, SchemaError, validate_raw_frame

logger = logging.getLogger(__name__)


def load_raw_csv(path: Path) -> pd.DataFrame:
    """Load a catalogue CSV. Missing optional columns are added as empty.

    ``cast`` and ``description`` are optional because the spreadsheet extract
    shipped with this repo does not contain them. A file that does contain
    them is used as-is.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw catalogue not found at {csv_path}. "
            "Place netflix_titles.csv there or run `python scripts/prepare_raw.py`."
        )
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=True)
    frame.columns = [str(column).strip() for column in frame.columns]
    try:
        validate_raw_frame(frame)
    except SchemaError:
        logger.error("Schema validation failed for %s", csv_path)
        raise
    for column in OPTIONAL_COLUMNS:
        if column not in frame.columns:
            logger.warning(
                "Optional column '%s' is absent from %s; text features will leave it empty",
                column,
                csv_path.name,
            )
            frame[column] = pd.NA
    return frame
