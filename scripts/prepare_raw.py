"""Build ``data/raw/netflix_titles.csv`` from the Excel extract.

Repairs Excel type-coercion on titles and Latin-1 mojibake, and writes
``date_added`` as ISO dates. Sentinel tokens such as ``Not Given`` are left
for ``src/data/clean.py``. Re-running on an unchanged spreadsheet is stable.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import load_config  # noqa: E402
from src.data.clean import parse_date_added  # noqa: E402
from src.data.repairs import repair_mojibake, repair_title, stringify_cell  # noqa: E402
from src.logging_conf import configure_logging  # noqa: E402


def _repair_text(value: object, counter: list[int]) -> str:
    original = stringify_cell(value)
    repaired = repair_mojibake(original)
    if repaired != original:
        counter[0] += 1
    return repaired


def main() -> None:
    configure_logging()
    config = load_config()
    spreadsheet = config.root / "Dataset.csv.xlsx"
    if not spreadsheet.exists():
        if config.raw_csv.exists():
            print(f"No spreadsheet at {spreadsheet}. Using existing {config.raw_csv}.")
            return
        print(f"Neither {spreadsheet} nor {config.raw_csv} exists.", file=sys.stderr)
        raise SystemExit(2)

    frame = pd.read_excel(spreadsheet, sheet_name="Dataset")
    frame.columns = [str(column).strip() for column in frame.columns]
    text_columns = [
        column
        for column in frame.columns
        if column not in {"title", "date_added", "release_year", "show_id"}
    ]
    mojibake_count = [0]
    title_rows: list[list[str]] = []
    date_datetimes = 0
    date_strings = 0
    records: list[dict[str, object]] = []

    for row in frame.itertuples(index=False):
        show_id = stringify_cell(row.show_id)
        original_title = row.title
        title = repair_title(show_id, original_title)
        original_text = stringify_cell(original_title)
        if title != original_text:
            title_rows.append([show_id, original_text, title])
        raw_date = row.date_added
        if isinstance(raw_date, datetime):
            date_datetimes += 1
        elif stringify_cell(raw_date):
            date_strings += 1
        iso_date, _year = parse_date_added(
            raw_date if isinstance(raw_date, datetime) else stringify_cell(raw_date)
        )
        record: dict[str, object] = {
            "show_id": show_id,
            "title": title,
            "date_added": iso_date or stringify_cell(raw_date),
            "release_year": stringify_cell(row.release_year),
        }
        for column in text_columns:
            record[column] = _repair_text(getattr(row, column), mojibake_count)
        records.append(record)

    ordered = list(frame.columns)
    output = pd.DataFrame(records)[ordered]
    config.raw_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(config.raw_csv, index=False, encoding="utf-8")

    notes = {
        "rows": int(len(output.index)),
        "titles_restored": len(title_rows),
        "mojibake_cells_repaired": mojibake_count[0],
        "date_added_datetimes": date_datetimes,
        "date_added_strings": date_strings,
        "titles": title_rows,
    }
    interim = config.root / "data" / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    (interim / "extract_repairs.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    print(
        f"Wrote {config.raw_csv} ({notes['rows']} rows, {notes['titles_restored']} titles restored)"
    )


if __name__ == "__main__":
    main()
