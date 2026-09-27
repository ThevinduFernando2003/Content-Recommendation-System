"""One assertion set per data defect the cleaner is required to handle."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.clean import clean_frame
from src.data.load import load_raw_csv
from src.data.schema import SchemaError


def _row(frame: pd.DataFrame, show_id: str) -> pd.Series:
    return frame.loc[frame["show_id"] == show_id].iloc[0]


def test_missing_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("show_id,type,title\ns1,Movie,Nope\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="column 'director' missing"):
        load_raw_csv(path)


def test_optional_columns_may_be_absent(tmp_path: Path) -> None:
    path = tmp_path / "thin.csv"
    path.write_text(
        "show_id,type,title,director,country,date_added,release_year,rating,duration,listed_in\n"
        "s1,Movie,Only,Jane Doe,Canada,2020-01-02,2020,PG,90 min,Dramas\n",
        encoding="utf-8",
    )
    frame = clean_frame(load_raw_csv(path))
    assert frame.loc[0, "cast"] == ""
    assert frame.loc[0, "description"] == ""


def test_rating_contamination_moves_duration(cleaned_df: pd.DataFrame) -> None:
    row = _row(cleaned_df, "s6")
    assert row["rating"] is None or pd.isna(row["rating"])
    assert row["duration_minutes"] == 74
    assert pd.isna(row["duration_seasons"])


def test_duration_units_are_separate(cleaned_df: pd.DataFrame) -> None:
    movie = _row(cleaned_df, "s1")
    show = _row(cleaned_df, "s7")
    assert movie["duration_minutes"] == 90
    assert pd.isna(movie["duration_seasons"])
    assert pd.isna(show["duration_minutes"])
    assert show["duration_seasons"] == 2


def test_multi_valued_fields_are_split(cleaned_df: pd.DataFrame) -> None:
    row = _row(cleaned_df, "s8")
    assert row["countries"] == ["United States", "India"]
    assert row["genres"] == ["Dramas", "Independent Movies"]


def test_five_genres(cleaned_df: pd.DataFrame) -> None:
    assert len(_row(cleaned_df, "s9")["genres"]) == 5


def test_missing_text_is_empty_not_unknown(cleaned_df: pd.DataFrame) -> None:
    empty_director = _row(cleaned_df, "s2")
    sentinel = _row(cleaned_df, "s18")
    unknown = _row(cleaned_df, "s20")
    assert empty_director["director"] == ""
    assert bool(empty_director["has_director"]) is False
    assert sentinel["director"] == ""
    assert sentinel["country"] == ""
    assert bool(sentinel["has_director"]) is False
    assert unknown["director"] == ""
    assert "unknown" not in sentinel["director"].lower()


def test_duplicate_show_id_kept_once(cleaned_df: pd.DataFrame) -> None:
    rows = cleaned_df.loc[cleaned_df["show_id"] == "s30"]
    assert len(rows) == 1
    assert rows.iloc[0]["title"] == "First Copy"


def test_accented_title_and_comma_title_stay_intact(cleaned_df: pd.DataFrame) -> None:
    assert _row(cleaned_df, "s12")["title"] == "Amélie"
    assert _row(cleaned_df, "s13")["title"] == "Hello, Goodbye"


def test_excel_coerced_title_is_restored(cleaned_df: pd.DataFrame) -> None:
    assert _row(cleaned_df, "s4523")["title"] == "22-Jul"


def test_mojibake_director_is_repaired(cleaned_df: pd.DataFrame) -> None:
    assert _row(cleaned_df, "s19")["directors"] == ["Alexis Almström"]


def test_dates_parse_both_formats(cleaned_df: pd.DataFrame) -> None:
    assert _row(cleaned_df, "s1")["date_added"] == "2021-09-25"
    assert _row(cleaned_df, "s2")["date_added"] == "2020-09-01"
    assert _row(cleaned_df, "s4523")["date_added"] == "2018-10-10"


def test_null_rating_stays_missing(cleaned_df: pd.DataFrame) -> None:
    assert _row(cleaned_df, "s5")["rating"] is None or pd.isna(_row(cleaned_df, "s5")["rating"])


def test_null_cast_and_country(cleaned_df: pd.DataFrame) -> None:
    assert _row(cleaned_df, "s3")["cast_list"] == []
    assert _row(cleaned_df, "s4")["countries"] == []
