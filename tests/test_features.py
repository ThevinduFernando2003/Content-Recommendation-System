"""Soup construction: weights, collapsed names, and no Unknown token."""

import pandas as pd

from src.config import load_config
from src.features.text import build_soup, collapse_entity, normalise_title


def test_multi_word_names_collapse() -> None:
    assert collapse_entity("Martin Scorsese") == "martinscorsese"
    assert collapse_entity("TV Dramas") == "tvdramas"
    assert normalise_title("Amélie") == "amelie"


def test_genre_token_is_repeated_by_its_weight(cleaned_df: pd.DataFrame) -> None:
    config = load_config()
    row = cleaned_df.loc[cleaned_df["show_id"] == "s14"].iloc[0]
    soup = build_soup(row, config.feature_weights, config.recommender.cast_cap)
    assert soup.split().count("crimetvshows") == config.feature_weights["genre"]
    assert "lenaortiz" in soup.split()
    assert "extraperson" not in soup.split()


def test_director_name_is_one_token(cleaned_df: pd.DataFrame) -> None:
    config = load_config()
    row = cleaned_df.loc[cleaned_df["show_id"] == "s1"].iloc[0]
    soup = build_soup(row, config.feature_weights, config.recommender.cast_cap)
    assert "martinscorsese" in soup.split()
    assert "martin" not in soup.split()


def test_empty_fields_do_not_emit_unknown(cleaned_df: pd.DataFrame) -> None:
    config = load_config()
    for show_id in ("s2", "s18", "s20"):
        row = cleaned_df.loc[cleaned_df["show_id"] == show_id].iloc[0]
        tokens = set(build_soup(row, config.feature_weights, config.recommender.cast_cap).split())
        assert "unknown" not in tokens
        assert "notgiven" not in tokens
