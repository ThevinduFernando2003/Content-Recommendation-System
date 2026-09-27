"""Repairs for defects that originate in the spreadsheet extract.

The catalogue was edited in Excel. Excel coerced date-like titles (``22-Jul``,
``3%``) into datetimes and numbers, and a previous encoding pass stored UTF-8
bytes as Latin-1 characters. Both repairs are idempotent.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time

# show_id -> original title, used only when the cell still looks coerced.
EXCEL_TITLE_BY_SHOW_ID: dict[str, str] = {
    "s3997": "Feb-09",
    "s5965": "Feb-09",
    "s3963": "15-Aug",
    "s5967": "15-Aug",
    "s4523": "22-Jul",
    "s5966": "22-Jul",
    "s5964": "Oct-01",
    "s4278": "23:59",
    "s5657": "7:19",
    "s4052": "2,215",
    "s2128": "3%",
}

_COERCED_TEXT = re.compile(r"^(\d{4}-\d{2}-\d{2}(?: 00:00:00)?|\d{2}:\d{2}:\d{2}|0\.03|2215)$")

# Tokens that mean "missing" in this extract. Never keep them as text features.
# "unknown" is included for person/place fields; the film titled Unknown is a title
# and is not passed through this helper.
MISSING_SENTINELS = frozenset({"not given", "unknown", "n/a", "na", "none", "null", "nan"})


def repair_mojibake(value: str) -> str:
    """Defect: UTF-8 text was decoded as Latin-1 (``Almström`` stored as ``Ã¶``)."""
    if not value:
        return value
    try:
        repaired = value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value
    return repaired


def _looks_coerced(original: object, text: str) -> bool:
    if isinstance(original, datetime | date | time | int | float) and not isinstance(
        original, bool
    ):
        return True
    return _COERCED_TEXT.match(text) is not None


def stringify_cell(value: object) -> str:
    """Render an Excel cell as text without inventing a title."""
    if value is None:
        return ""
    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, time):
        return value.strftime("%H:%M:%S")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def repair_title(show_id: str, original: object) -> str:
    """Defect: Excel coerced date-like titles into datetimes, times, and numbers."""
    text = stringify_cell(original)
    if show_id in EXCEL_TITLE_BY_SHOW_ID and _looks_coerced(original, text):
        return EXCEL_TITLE_BY_SHOW_ID[show_id]
    return repair_mojibake(text)
