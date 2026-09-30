"""Behaviour of lookup, filters, franchise suppression, and determinism."""

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from rapidfuzz.utils import default_process

from src.config import load_config
from src.data.clean import clean_frame
from src.features.text import normalise_title
from src.models.recommender import (
    EmptyQueryError,
    Recommender,
    TitleNotFoundError,
    _near_title_score,
)


def test_does_not_recommend_itself(recommender: Recommender) -> None:
    resolved, results = recommender.recommend("Glass Alibi", n=20, exclude_same_franchise=False)
    assert resolved.title == "Glass Alibi"
    assert all(item.show_id != "s15" for item in results)
    assert all(normalise_title(item.title) != normalise_title(resolved.title) for item in results)


def test_n_is_respected_and_scores_are_ordered(recommender: Recommender) -> None:
    _resolved, results = recommender.recommend("Glass Alibi", n=3, exclude_same_franchise=False)
    assert len(results) == 3
    scores = [item.similarity for item in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_known_pair_ranks_in_the_top_three(recommender: Recommender) -> None:
    _resolved, results = recommender.recommend("City of Shadows", n=5, exclude_same_franchise=False)
    assert "Glass Alibi" in [item.title for item in results[:3]]
    assert results[0].reason.startswith("Because")
    assert "Ava Morales" in results[0].shared_people or "Crime TV Shows" in results[0].shared_genres


def test_filters_remove_types_and_years(recommender: Recommender) -> None:
    _resolved, movies = recommender.recommend("City of Shadows", n=10, content_type="Movie")
    assert movies
    assert all(item.type == "Movie" for item in movies)
    _resolved, recent = recommender.recommend(
        "Glass Alibi", n=10, min_release_year=2019, exclude_same_franchise=False
    )
    assert all(item.release_year >= 2019 for item in recent)


def test_franchise_filter_drops_sequel(recommender: Recommender) -> None:
    _resolved, hidden = recommender.recommend("Panda Legend", n=10, exclude_same_franchise=True)
    assert all(item.title != "Panda Legend 2" for item in hidden)
    _resolved, shown = recommender.recommend("Panda Legend", n=5, exclude_same_franchise=False)
    assert shown[0].title == "Panda Legend 2"


def test_exact_title_duplicate_is_not_returned(recommender: Recommender) -> None:
    resolved, results = recommender.recommend("Shadow City", n=10, content_type="Movie")
    assert resolved.title == "Shadow City"
    assert all(item.title != "Shadow City" for item in results)


def test_ambiguous_title_raises_with_suggestions(recommender: Recommender) -> None:
    with pytest.raises(TitleNotFoundError) as caught:
        recommender.recommend("Shadow City")
    assert caught.value.suggestions
    assert any("Movie" in suggestion for suggestion in caught.value.suggestions)


def test_rating_filter_and_catalogue_description(recommender: Recommender) -> None:
    _resolved, results = recommender.recommend(
        "Panda Legend", n=10, rating="PG", exclude_same_franchise=False
    )
    assert results
    assert all(item.rating == "PG" for item in results)
    sequel = next(item for item in results if item.title == "Panda Legend 2")
    assert "panda" in sequel.description.casefold()


def test_fuzzy_typo_resolves(recommender: Recommender) -> None:
    resolved, results = recommender.recommend("city of shadws", n=3, exclude_same_franchise=False)
    assert resolved.title == "City of Shadows"
    assert resolved.resolved_from == "city of shadws"
    assert results


def test_romance_mood_uses_the_catalogue_word() -> None:
    rows = []
    for show_id, title, listed_in in (
        ("s1", "Romantic Night", "Romantic Movies"),
        ("s2", "Second Romance", "Romantic Movies"),
        ("s3", "Plain Drama", "Dramas"),
        ("s4", "Plain Comedy", "Comedies"),
    ):
        rows.append(
            {
                "show_id": show_id,
                "type": "Movie",
                "title": title,
                "director": "Jane Doe",
                "cast": "Amy Adams",
                "country": "United States",
                "date_added": "September 25, 2021",
                "release_year": 2019,
                "rating": "PG-13",
                "duration": "100 min",
                "listed_in": listed_in,
                "description": "A catalogue row for a mood query.",
            }
        )
    model = Recommender(load_config()).fit(clean_frame(pd.DataFrame(rows)))
    assert model._expand_query("romance") == "romantic"
    results = model.recommend_from_text("romance", n=2)
    assert results
    assert {item.title for item in results} <= {"Romantic Night", "Second Romance"}


def test_shortened_title_prefers_the_full_name() -> None:
    query = default_process("break bad")
    full = _near_title_score(query, default_process("Breaking Bad"))
    short = _near_title_score(query, default_process("Break"))
    longer = _near_title_score(query, default_process("El Camino: A Breaking Bad Movie"))
    assert full >= 90
    assert full > short
    assert full > longer
    assert short < 90


def test_shortened_title_resolves_and_search_lists_it() -> None:
    rows = []
    for show_id, content_type, title, listed_in, duration in (
        ("s1", "TV Show", "Breaking Bad", "Crime TV Shows, TV Dramas", "5 Seasons"),
        ("s2", "Movie", "Break", "Dramas", "90 min"),
        ("s3", "Movie", "El Camino: A Breaking Bad Movie", "Dramas", "122 min"),
        ("s4", "Movie", "Glass Alibi", "Dramas", "100 min"),
    ):
        rows.append(
            {
                "show_id": show_id,
                "type": content_type,
                "title": title,
                "director": "Jane Doe",
                "cast": "Amy Adams",
                "country": "United States",
                "date_added": "September 25, 2021",
                "release_year": 2013,
                "rating": "TV-MA",
                "duration": duration,
                "listed_in": listed_in,
                "description": "A catalogue row for near-title lookup.",
            }
        )
    model = Recommender(load_config()).fit(clean_frame(pd.DataFrame(rows)))
    resolved, results = model.recommend("break bad", n=2)
    assert resolved.title == "Breaking Bad"
    assert resolved.resolved_from == "break bad"
    assert results
    assert all(item.title != "Breaking Bad" for item in results)
    hits = model.search_titles("break bad", limit=5)
    assert hits
    assert hits[0]["title"] == "Breaking Bad"


def test_lowercase_exact_match_has_no_fuzzy_note(recommender: Recommender) -> None:
    resolved, _results = recommender.recommend("glass alibi", n=3)
    assert resolved.title == "Glass Alibi"
    assert resolved.resolved_from is None


def test_unknown_title_carries_suggestions(recommender: Recommender) -> None:
    with pytest.raises(TitleNotFoundError) as caught:
        recommender.recommend("zzzznotitle")
    assert caught.value.suggestions


def test_cold_start_and_empty_vocabulary(recommender: Recommender) -> None:
    results = recommender.recommend_from_text("crime drama detectives", n=5)
    assert results
    assert any("Crime" in genre for item in results for genre in item.shared_genres)
    assert all(item.reason for item in results)
    with pytest.raises(EmptyQueryError):
        recommender.recommend_from_text("qqqq zzzz yyyy")
    with pytest.raises(EmptyQueryError):
        recommender.recommend_from_text("the and of")
    expanded = recommender.recommend_from_text("drama", n=3)
    assert expanded
    via_text = recommender.recommend_from_text("glass alibi", n=3)
    _resolved, via_title = recommender.recommend("Glass Alibi", n=3)
    assert [item.show_id for item in via_text] == [item.show_id for item in via_title]


def test_genre_only_reason_does_not_claim_taste(recommender: Recommender) -> None:
    labelled = False
    for title in recommender.titles:
        _resolved, results = recommender.recommend(str(title), n=5, exclude_same_franchise=False)
        for item in results:
            if item.shared_people:
                assert "not evidence a viewer would like it" not in item.reason
                continue
            if item.shared_genres:
                assert "Genre overlap only" in item.reason
                assert "viewing history" in item.reason
                labelled = True
                break
        if labelled:
            break
    assert labelled


def test_saturated_score_is_named(recommender: Recommender) -> None:
    _resolved, results = recommender.recommend("Glass Alibi", n=8, exclude_same_franchise=False)
    saturated = [item for item in results if item.similarity >= 0.99 and not item.shared_people]
    for item in saturated:
        assert "saturated" in item.reason


def test_explain_names_shared_features(recommender: Recommender) -> None:
    payload = recommender.explain("City of Shadows", "Glass Alibi")
    assert "Crime TV Shows" in payload["shared_genres"]
    assert "Ava Morales" in payload["shared_people"]
    assert payload["top_terms"]
    assert payload["reason"].startswith("Because")


def test_fit_is_deterministic(cleaned_df: pd.DataFrame) -> None:
    config = load_config()
    first = Recommender(config).fit(cleaned_df)
    second = Recommender(config).fit(cleaned_df)
    left = [item.title for item in first.recommend("Glass Alibi", n=10)[1]]
    right = [item.title for item in second.recommend("Glass Alibi", n=10)[1]]
    assert left == right


@settings(max_examples=25, deadline=None)
@given(
    title=st.sampled_from(
        [
            "Glass Alibi",
            "Panda Legend",
            "Amélie",
            "Hello, Goodbye",
            "Filler Title 22",
            "22-Jul",
        ]
    ),
    n=st.integers(min_value=1, max_value=12),
)
def test_recommend_bounds(recommender: Recommender, title: str, n: int) -> None:
    resolved, results = recommender.recommend(title, n=n, exclude_same_franchise=True)
    assert len(results) <= n
    assert all(item.show_id != recommender.show_ids[resolved.index] for item in results)
    assert all(normalise_title(item.title) != normalise_title(resolved.title) for item in results)
