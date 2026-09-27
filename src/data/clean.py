"""Turn the raw catalogue into one row per title with explicit missingness.

Every defect called out in the data contract is handled below, and the comment
on each block names that defect.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

import pandas as pd

from src.data.repairs import MISSING_SENTINELS, repair_mojibake, repair_title

logger = logging.getLogger(__name__)

# Defect: ratings column contamination. A duration string was shifted into rating.
_RATING_DURATION_RE = re.compile(r"^\d+\s*min$", re.IGNORECASE)
_MINUTES_RE = re.compile(r"^(\d+)\s*min$", re.IGNORECASE)
_SEASONS_RE = re.compile(r"^(\d+)\s*seasons?$", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")

_DATE_FORMATS = (
    "%B %d, %Y",
    "%b %d, %Y",
    "%m/%d/%Y",
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        return False
    return False


def _as_text(value: object) -> str:
    if _is_missing(value):
        return ""
    return repair_mojibake(str(value).strip())


def _blank_sentinel(value: object) -> str:
    """Defect: missing director/country/cast are the sentinel ``Not Given``, not null.

    Empty string is returned on purpose. The literal ``Unknown`` would become a
    shared token and glue unrelated titles together.
    """
    text = _as_text(value)
    if not text or text.casefold() in MISSING_SENTINELS:
        return ""
    return text


def _split_multi(value: object) -> list[str]:
    """Defect: director, cast, country and listed_in are comma-separated lists."""
    text = _blank_sentinel(value)
    if not text:
        return []
    parts: list[str] = []
    for part in text.split(","):
        # Defect: genre strings have inconsistent spacing. Collapse it before the vocabulary.
        cleaned = _SPACE_RE.sub(" ", part).strip()
        if cleaned and cleaned.casefold() not in MISSING_SENTINELS:
            parts.append(repair_mojibake(cleaned))
    return parts


def parse_date_added(value: object) -> tuple[str, int | None]:
    """Accept both ``September 25, 2021`` and the extract's ``9/25/2021`` / ISO datetimes."""
    text = _as_text(value)
    if not text:
        return "", None
    parsed: datetime | None = None
    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
            break
        except ValueError:
            continue
    if parsed is None:
        logger.warning("Unparsed date_added value %r", text)
        return "", None
    return parsed.date().isoformat(), parsed.year


def _parse_duration(duration: str, content_type: str) -> tuple[float | None, float | None]:
    """Defect: duration is two units. Never store both in one numeric column.

    Movies fill ``duration_minutes`` only. TV shows fill ``duration_seasons`` only.
    """
    minutes_match = _MINUTES_RE.match(duration)
    seasons_match = _SEASONS_RE.match(duration)
    minutes = float(minutes_match.group(1)) if minutes_match else None
    seasons = float(seasons_match.group(1)) if seasons_match else None
    if content_type == "Movie":
        return minutes, None
    if content_type == "TV Show":
        return None, seasons
    return None, None


def clean_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy frame sorted by numeric ``show_id`` with duplicates removed."""
    frame = raw.copy()
    frame["show_id"] = frame["show_id"].map(lambda value: _as_text(value))
    # Defect: duplicate show_id. Keep the first occurrence after a stable id order.
    frame["_ord"] = frame["show_id"].str.extract(r"(\d+)", expand=False)
    frame["_ord"] = pd.to_numeric(frame["_ord"], errors="coerce").fillna(0).astype(int)
    frame = frame.sort_values(["_ord", "show_id"], kind="mergesort")
    before = len(frame.index)
    frame = frame.drop_duplicates(subset=["show_id"], keep="first")
    dropped = before - len(frame.index)
    if dropped:
        logger.info("Dropped %s duplicate show_id rows", dropped)

    records: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        show_id = str(row.show_id)
        # Defect: Excel coerced date-like titles into datetimes and numbers.
        title = repair_title(show_id, row.title)
        content_type = _as_text(row.type)
        rating_text = _blank_sentinel(row.rating)
        duration_text = _as_text(row.duration)

        # Defect: ratings column contamination — ``74 min`` landed in rating.
        if _RATING_DURATION_RE.match(rating_text):
            if not duration_text:
                duration_text = _SPACE_RE.sub(" ", rating_text)
            rating_text = ""

        minutes, seasons = _parse_duration(duration_text, content_type)
        directors = _split_multi(row.director)
        cast_list = _split_multi(row.cast)
        countries = _split_multi(row.country)
        genres = _split_multi(row.listed_in)
        date_added, date_added_year = parse_date_added(row.date_added)
        description = _blank_sentinel(row.description)
        release_year = int(pd.to_numeric(row.release_year, errors="raise"))

        records.append(
            {
                "show_id": show_id,
                "type": content_type,
                "title": title,
                "director": ", ".join(directors),
                "directors": directors,
                "has_director": bool(directors),
                "cast": ", ".join(cast_list),
                "cast_list": cast_list,
                "country": ", ".join(countries),
                "countries": countries,
                "date_added": date_added,
                "date_added_year": date_added_year,
                "release_year": release_year,
                "rating": rating_text or None,
                "duration": duration_text,
                "duration_minutes": minutes,
                "duration_seasons": seasons,
                "listed_in": ", ".join(genres),
                "genres": genres,
                "description": description,
            }
        )

    cleaned = pd.DataFrame.from_records(records)
    cleaned = cleaned.drop_duplicates(subset=["show_id"], keep="first").reset_index(drop=True)
    logger.info("Cleaned catalogue has %s titles", len(cleaned.index))
    return cleaned
