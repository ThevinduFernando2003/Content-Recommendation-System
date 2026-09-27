"""Behaviour of lookup, filters, franchise suppression, and determinism."""

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config import load_config
from src.features.text import normalise_title
from src.models.recommender import EmptyQueryError, Recommender, TitleNotFoundError


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


def test_fuzzy_typo_resolves(recommender: Recommender) -> None:
    resolved, results = recommender.recommend("city of shadws", n=3, exclude_same_franchise=False)
    assert resolved.title == "City of Shadows"
    assert resolved.resolved_from == "city of shadws"
    assert results


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
