"""Build the weighted text document (the "soup") for one title."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_SPACE_RE = re.compile(r"\s+")


def fold_text(value: str) -> str:
    """Lowercase, strip accents, and turn punctuation into spaces."""
    if not value:
        return ""
    text = unicodedata.normalize("NFKD", value)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def collapse_entity(value: str) -> str:
    """Collapse a multi-word name into one token: ``Martin Scorsese`` -> ``martinscorsese``."""
    return fold_text(value).replace(" ", "")


def normalise_title(value: str) -> str:
    """Case-insensitive, accent-insensitive key for title lookup."""
    return fold_text(value)


def _repeat(blob: str, times: int) -> str:
    blob = blob.strip()
    if not blob or times <= 0:
        return ""
    return " ".join([blob] * int(times))


def build_soup(row: pd.Series, weights: dict[str, int], cast_cap: int) -> str:
    """Build one weighted document. Empty fields contribute nothing (no ``unknown`` token)."""
    genres = [str(genre) for genre in row["genres"] if str(genre).strip()]
    genre_tokens = " ".join(collapse_entity(genre) for genre in genres if collapse_entity(genre))
    genre_words = " ".join(fold_text(genre) for genre in genres if fold_text(genre))

    cast = [str(name) for name in row["cast_list"][:cast_cap]]
    cast_tokens = " ".join(token for name in cast if (token := collapse_entity(name)))
    director_tokens = " ".join(
        token for name in row["directors"] if (token := collapse_entity(str(name)))
    )
    country_tokens = " ".join(
        token for name in row["countries"] if (token := collapse_entity(str(name)))
    )
    rating = row["rating"]
    rating_token = (
        collapse_entity(str(rating)) if isinstance(rating, str) and rating.strip() else ""
    )
    description = fold_text(str(row["description"])) if isinstance(row["description"], str) else ""

    parts = [
        _repeat(genre_tokens, weights.get("genre", 0)),
        _repeat(cast_tokens, weights.get("cast", 0)),
        _repeat(director_tokens, weights.get("director", 0)),
        _repeat(country_tokens, weights.get("country", 0)),
        _repeat(collapse_entity(str(row["type"])), weights.get("type", 0)),
        _repeat(rating_token, weights.get("rating", 0)),
        _repeat(genre_words, weights.get("genre_words", 0)),
        description if weights.get("description", 0) > 0 else "",
    ]
    return " ".join(part for part in parts if part).strip()


def build_soups(frame: pd.DataFrame, weights: dict[str, int], cast_cap: int) -> list[str]:
    """Vectorised-enough row loop. The catalogue is under 10k rows."""
    return [build_soup(row, weights, cast_cap) for _, row in frame.iterrows()]
