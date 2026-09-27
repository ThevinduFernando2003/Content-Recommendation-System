"""Content recommender over an L2-normalised sparse TF-IDF matrix.

Query time computes one sparse matrix-vector product. Because each row is
L2-normalised, that product is cosine similarity. The full dense N×N matrix
is never built.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.utils import default_process
from scipy import sparse

from src.config import AppConfig
from src.features.text import build_soups, fold_text, normalise_title
from src.features.vectorize import fit_matrix, transform_query

logger = logging.getLogger(__name__)

ContentType = Literal["Movie", "TV Show"]
_CHOICE_RE = re.compile(
    r"^(?P<title>.*) \[(?P<type>Movie|TV Show), (?P<year>\d{4})(?: \| (?P<show_id>s\d+))?\]$"
)


class TitleNotFoundError(LookupError):
    """No catalogue row matched the query. ``suggestions`` are safe to show a user."""

    def __init__(self, message: str, suggestions: list[str]) -> None:
        super().__init__(message)
        self.suggestions = suggestions


class EmptyQueryError(ValueError):
    """The free-text query survived as an all-zero vector."""


@dataclass(frozen=True)
class Recommendation:
    show_id: str
    title: str
    type: str
    release_year: int
    rating: str | None
    genres: list[str]
    similarity: float
    reason: str
    shared_genres: list[str]
    shared_people: list[str]


@dataclass(frozen=True)
class ResolvedTitle:
    index: int
    title: str
    resolved_from: str | None


def _english_list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _opt_str(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value) is True:
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _as_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value if str(item).strip()]
    try:
        if pd.isna(value) is True:
            return []
    except (TypeError, ValueError):
        pass
    return [str(item) for item in cast(Iterable[object], value) if str(item).strip()]


class Recommender:
    """Fit on a cleaned catalogue frame and return explained neighbours."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.fitted = False
        self.model_version = "untrained"
        self.vectorizer: Any = None
        self.matrix: sparse.csr_matrix | None = None
        self.feature_names: np.ndarray = np.array([])
        self.titles: np.ndarray = np.array([])
        self.title_norms: np.ndarray = np.array([])
        self.types: np.ndarray = np.array([])
        self.years: np.ndarray = np.array([])
        self.show_ids: np.ndarray = np.array([])
        self.ratings: list[str | None] = []
        self.genres: list[list[str]] = []
        self.directors: list[list[str]] = []
        self.cast_list: list[list[str]] = []
        self.countries: list[list[str]] = []
        self.descriptions: list[str] = []
        self.date_added: list[str] = []
        self.date_added_year: list[int | None] = []
        self.duration_minutes: list[float | None] = []
        self.same_title_indices: list[np.ndarray] = []
        self.title_token_sets: list[set[str]] = []
        self._show_id_to_index: dict[str, int] = {}
        self._norm_to_indices: dict[str, list[int]] = {}

    def fit(self, frame: pd.DataFrame) -> Recommender:
        """Fit the vectoriser on ``frame`` (already cleaned) and store the index."""
        cleaned = frame.reset_index(drop=True)
        weights = self.config.feature_weights
        soups = build_soups(cleaned, weights, self.config.recommender.cast_cap)
        vectorizer, matrix = fit_matrix(soups, self.config.vectorizer)
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.feature_names = np.asarray(vectorizer.get_feature_names_out())
        self._bind_frame(cleaned)
        self.fitted = True
        logger.info(
            "Fitted %s index: %s titles, %s features, %s stored nonzeros",
            self.config.vectorizer.kind,
            matrix.shape[0],
            matrix.shape[1],
            matrix.nnz,
        )
        return self

    def _bind_frame(self, cleaned: pd.DataFrame) -> None:
        self.titles = cleaned["title"].astype(str).to_numpy()
        self.title_norms = np.asarray([normalise_title(title) for title in self.titles])
        self.types = cleaned["type"].astype(str).to_numpy()
        self.years = cleaned["release_year"].astype(int).to_numpy()
        self.show_ids = cleaned["show_id"].astype(str).to_numpy()
        self.ratings = [_opt_str(value) for value in cleaned["rating"].tolist()]
        self.genres = [_as_str_list(value) for value in cleaned["genres"].tolist()]
        self.directors = [_as_str_list(value) for value in cleaned["directors"].tolist()]
        self.cast_list = [_as_str_list(value) for value in cleaned["cast_list"].tolist()]
        self.countries = [_as_str_list(value) for value in cleaned["countries"].tolist()]
        self.descriptions = [
            "" if _opt_str(value) is None else str(value)
            for value in cleaned["description"].tolist()
        ]
        self.date_added = [
            "" if _opt_str(value) is None else str(value)
            for value in cleaned["date_added"].tolist()
        ]
        years: list[int | None] = []
        for value in cleaned["date_added_year"].tolist():
            if value is None or pd.isna(value):
                years.append(None)
            else:
                years.append(int(value))
        self.date_added_year = years
        minutes: list[float | None] = []
        for value in cleaned["duration_minutes"].tolist():
            if value is None or pd.isna(value):
                minutes.append(None)
            else:
                minutes.append(float(value))
        self.duration_minutes = minutes
        self._show_id_to_index = {show_id: idx for idx, show_id in enumerate(self.show_ids)}
        grouped: dict[str, list[int]] = {}
        for idx, norm in enumerate(self.title_norms):
            grouped.setdefault(str(norm), []).append(idx)
        self._norm_to_indices = grouped
        self.same_title_indices = [
            np.asarray(grouped[str(norm)], dtype=int) for norm in self.title_norms
        ]
        self.title_token_sets = [set(str(norm).split()) for norm in self.title_norms]

    def recommend(
        self,
        title: str,
        n: int | None = None,
        content_type: ContentType | None = None,
        min_release_year: int | None = None,
        exclude_same_franchise: bool = True,
    ) -> tuple[ResolvedTitle, list[Recommendation]]:
        """Return the top ``n`` neighbours for a catalogue title."""
        self._require_fitted()
        resolved = self._resolve(title, content_type)
        limit = self.config.recommender.default_n if n is None else n
        matrix = self.matrix
        if matrix is None:
            raise RuntimeError("Recommender is not fitted. Run `make train` first.")
        scores = self._scores_from_vector(matrix[resolved.index])
        self._apply_filters(scores, resolved.index, content_type, min_release_year)
        chosen = self._take(scores, resolved.index, limit, exclude_same_franchise)
        results = [self._to_recommendation(resolved.index, index, score) for index, score in chosen]
        return resolved, results

    def recommend_from_text(self, query: str, n: int | None = None) -> list[Recommendation]:
        """Cold start: transform free text with the fitted vectoriser and query the index."""
        self._require_fitted()
        folded = fold_text(query)
        if not folded:
            raise EmptyQueryError(
                "That description has no usable words. Name a genre, country, or person."
            )
        title_hits = self._norm_to_indices.get(folded, [])
        if len(title_hits) == 1:
            _resolved, titled = self.recommend(str(self.titles[title_hits[0]]), n=n)
            return titled
        if len(title_hits) > 1:
            raise EmptyQueryError(
                f"'{query}' matches more than one catalogue title. Pick one from search."
            )
        folded = self._expand_query(folded)
        vector = transform_query(self.vectorizer, folded)
        if vector.nnz == 0:
            raise EmptyQueryError(
                "That description has no words in common with the catalogue. "
                "Try genres or places, for example 'crime drama' or 'kids comedy'."
            )
        limit = self.config.recommender.default_n if n is None else n
        scores = self._scores_from_vector(vector)
        order = np.argsort(-scores, kind="mergesort")
        results: list[Recommendation] = []
        for index in order:
            score = float(scores[int(index)])
            if score <= 0:
                break
            results.append(self._to_recommendation(None, int(index), score, query_text=folded))
            if len(results) >= limit:
                break
        return results

    def explain(self, title: str, neighbour: str) -> dict[str, Any]:
        """Shared genres, shared people, and the top TF-IDF terms in the dot product."""
        self._require_fitted()
        left = self._resolve(title, None)
        right = self._resolve(neighbour, None)
        shared_genres, shared_people, top_terms, reason = self._explanation(
            left.index, right.index, None
        )
        return {
            "query_title": left.title,
            "neighbour_title": right.title,
            "shared_genres": shared_genres,
            "shared_people": shared_people,
            "top_terms": top_terms,
            "reason": reason,
        }

    def search_titles(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Case-insensitive substring search for the UI typeahead."""
        self._require_fitted()
        needle = query.strip().casefold()
        if not needle:
            order = np.argsort(self.titles)
            return [self.brief(int(index)) for index in order[:limit]]
        starts: list[int] = []
        contains: list[int] = []
        for index, title in enumerate(self.titles):
            folded = str(title).casefold()
            if folded.startswith(needle):
                starts.append(index)
            elif needle in folded:
                contains.append(index)
        chosen = (starts + contains)[:limit]
        return [self.brief(index) for index in chosen]

    def get_title(self, show_id: str) -> dict[str, Any] | None:
        self._require_fitted()
        index = self._show_id_to_index.get(show_id)
        if index is None:
            return None
        return self.detail(index)

    def brief(self, index: int) -> dict[str, Any]:
        return {
            "show_id": str(self.show_ids[index]),
            "title": str(self.titles[index]),
            "type": str(self.types[index]),
            "release_year": int(self.years[index]),
            "rating": self.ratings[index],
        }

    def detail(self, index: int) -> dict[str, Any]:
        payload = self.brief(index)
        payload.update(
            {
                "genres": list(self.genres[index]),
                "directors": list(self.directors[index]),
                "cast": list(self.cast_list[index]),
                "countries": list(self.countries[index]),
                "description": self.descriptions[index],
                "date_added": self.date_added[index],
            }
        )
        return payload

    def catalogue_stats(self) -> dict[str, Any]:
        """Aggregates for the Explore tab. Counts come from the fitted catalogue."""
        self._require_fitted()
        n_titles = int(len(self.titles))
        type_counts = {
            "Movie": int(np.sum(self.types == "Movie")),
            "TV Show": int(np.sum(self.types == "TV Show")),
        }
        year_counts: dict[int, int] = {}
        for year in self.date_added_year:
            if year is None:
                continue
            year_counts[year] = year_counts.get(year, 0) + 1
        genre_counts: dict[str, int] = {}
        for genres in self.genres:
            for genre in genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        country_counts: dict[str, int] = {}
        for countries in self.countries:
            for country in countries:
                country_counts[country] = country_counts.get(country, 0) + 1
        rating_counts: dict[str, int] = {}
        for rating in self.ratings:
            if rating:
                rating_counts[rating] = rating_counts.get(rating, 0) + 1
        movie_minutes = [
            value
            for value, content_type in zip(self.duration_minutes, self.types, strict=True)
            if content_type == "Movie" and value is not None
        ]
        edges = [0, 60, 90, 120, 150, 180, 240, 400]
        histogram: list[dict[str, int | float]] = []
        for left, right in zip(edges[:-1], edges[1:], strict=True):
            count = sum(1 for value in movie_minutes if left <= value < right)
            histogram.append({"bin_start": left, "bin_end": right, "count": count})
        return {
            "n_titles": n_titles,
            "n_movies": type_counts["Movie"],
            "n_tv_shows": type_counts["TV Show"],
            "added_per_year": [
                {"year": year, "count": year_counts[year]} for year in sorted(year_counts)
            ],
            "top_genres": _top_counts(genre_counts, 15),
            "top_countries": _top_counts(country_counts, 15),
            "ratings": _top_counts(rating_counts, 20),
            "movie_runtime_bins": histogram,
        }

    def neighbor_table(
        self,
        n: int,
        exclude_same_franchise: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-n index and score for every catalogue row. Used by evaluation only."""
        self._require_fitted()
        assert self.matrix is not None
        n_items = int(self.matrix.shape[0])
        width = min(n, max(n_items - 1, 0))
        indices = np.full((n_items, width), -1, dtype=np.int32)
        score_table = np.zeros((n_items, width), dtype=np.float64)
        if width == 0:
            return indices, score_table
        block_size = 256
        for start in range(0, n_items, block_size):
            stop = min(start + block_size, n_items)
            block = np.asarray(
                self.matrix[start:stop].dot(self.matrix.T).toarray(), dtype=np.float64
            )
            block = np.clip(np.nan_to_num(block, nan=0.0), 0.0, 1.0)
            for local, query_idx in enumerate(range(start, stop)):
                row = block[local]
                row[self.same_title_indices[query_idx]] = -1.0
                chosen = self._take(row, query_idx, width, exclude_same_franchise)
                for rank, (index, score) in enumerate(chosen):
                    indices[query_idx, rank] = index
                    score_table[query_idx, rank] = score
        return indices, score_table

    def _scores_from_vector(self, vector: sparse.spmatrix) -> np.ndarray:
        assert self.matrix is not None
        scores = self.matrix.dot(vector.T)
        array = np.asarray(scores.toarray()).ravel().astype(np.float64)
        return np.clip(np.nan_to_num(array, nan=0.0), 0.0, 1.0)

    def _apply_filters(
        self,
        scores: np.ndarray,
        query_index: int,
        content_type: ContentType | None,
        min_release_year: int | None,
    ) -> None:
        scores[self.same_title_indices[query_index]] = -1.0
        if content_type is not None:
            scores[self.types != content_type] = -1.0
        if min_release_year is not None:
            scores[self.years < min_release_year] = -1.0

    def _take(
        self,
        scores: np.ndarray,
        query_index: int,
        n: int,
        exclude_same_franchise: bool,
    ) -> list[tuple[int, float]]:
        if n <= 0:
            return []
        order = self._rank_valid(scores)
        chosen: list[tuple[int, float]] = []
        for index in order:
            candidate = int(index)
            if exclude_same_franchise and self._same_franchise_idx(query_index, candidate):
                continue
            chosen.append((candidate, round(float(scores[candidate]), 6)))
            if len(chosen) >= n:
                break
        return chosen

    def _rank_valid(self, scores: np.ndarray) -> np.ndarray:
        """Highest-scoring valid rows, capped by ``candidate_pool``.

        A full argsort of every catalogue row is what made evaluation slow.
        The pool is large relative to N, and the franchise filter runs inside it.
        """
        valid = np.flatnonzero(scores >= 0.0)
        if valid.size == 0:
            return valid
        pool = min(int(valid.size), max(int(self.config.recommender.candidate_pool), 1))
        if int(valid.size) > pool:
            picked = np.argpartition(scores[valid], -pool)[-pool:]
            valid = valid[picked]
        return valid[np.argsort(-scores[valid], kind="mergesort")]

    def _same_franchise_idx(self, left: int, right: int) -> bool:
        return self._same_franchise_text(
            str(self.title_norms[left]), str(self.title_norms[right]), left, right
        )

    def _same_franchise_text(
        self,
        left_norm: str,
        right_norm: str,
        left_index: int | None = None,
        right_index: int | None = None,
    ) -> bool:
        if not left_norm or not right_norm:
            return False
        if left_norm == right_norm:
            return True
        limit = self.config.recommender.franchise_prefix_chars
        if _common_prefix_length(left_norm, right_norm) >= limit:
            return True
        if left_index is not None and right_index is not None:
            if self.title_token_sets[left_index].isdisjoint(self.title_token_sets[right_index]):
                return False
        elif set(left_norm.split()).isdisjoint(right_norm.split()):
            return False
        return (
            fuzz.token_set_ratio(left_norm, right_norm)
            >= self.config.recommender.franchise_token_set_ratio
        )

    def _resolve(self, title: str, content_type: ContentType | None) -> ResolvedTitle:
        query = title.strip()
        if not query:
            raise TitleNotFoundError("Title is empty.", [])
        choice = _CHOICE_RE.match(query)
        if choice:
            pinned = choice.group("show_id")
            if pinned:
                show_index = self._show_id_to_index.get(pinned)
                if show_index is not None:
                    return ResolvedTitle(show_index, str(self.titles[show_index]), query)
            indices = self._indices_for_choice(
                choice.group("title"),
                choice.group("type"),
                int(choice.group("year")),
            )
            if len(indices) == 1:
                return ResolvedTitle(indices[0], str(self.titles[indices[0]]), query)
            if len(indices) > 1:
                return self._disambiguate(query, indices, content_type, resolved_from=query)

        show_index = self._show_id_to_index.get(query)
        if show_index is not None:
            return ResolvedTitle(show_index, str(self.titles[show_index]), None)

        norm = normalise_title(query)
        exact = self._norm_to_indices.get(norm, [])
        if exact:
            return self._disambiguate(query, exact, content_type, resolved_from=None)

        matches = process.extract(
            query,
            list(dict.fromkeys(str(item) for item in self.titles)),
            scorer=fuzz.WRatio,
            processor=default_process,
            limit=self.config.recommender.fuzzy_suggestion_limit,
        )
        suggestions = [self._suggestion_for_title(match) for match, _score, _idx in matches]
        if matches:
            top_title, top_score, _ = matches[0]
            second_score = matches[1][1] if len(matches) > 1 else -1.0
            threshold = self.config.recommender.fuzzy_score_threshold
            gap = self.config.recommender.fuzzy_ambiguity_gap
            unambiguous = second_score < threshold or (top_score - second_score) >= gap
            if top_score >= threshold and unambiguous:
                indices = self._norm_to_indices.get(normalise_title(str(top_title)), [])
                resolved = self._disambiguate(query, indices, content_type, resolved_from=query)
                logger.info(
                    "Fuzzy resolved %r to %r (score %.1f)", query, resolved.title, top_score
                )
                return resolved
        raise TitleNotFoundError(
            f"No catalogue title matches '{query}'.",
            suggestions,
        )

    def _indices_for_choice(self, title: str, content_type: str, year: int) -> list[int]:
        norm = normalise_title(title.strip())
        return [
            index
            for index in self._norm_to_indices.get(norm, [])
            if self.types[index] == content_type and int(self.years[index]) == year
        ]

    def _disambiguate(
        self,
        query: str,
        indices: list[int],
        content_type: ContentType | None,
        resolved_from: str | None,
    ) -> ResolvedTitle:
        chosen = indices
        if content_type is not None:
            typed = [index for index in indices if self.types[index] == content_type]
            if typed:
                chosen = typed
        if len(chosen) == 1:
            index = chosen[0]
            catalogue_title = str(self.titles[index])
            changed = resolved_from
            if changed is None and query != catalogue_title:
                changed = None
            return ResolvedTitle(index, catalogue_title, changed)
        suggestions = [
            self._suggestion(index)
            for index in chosen[: self.config.recommender.fuzzy_suggestion_limit]
        ]
        raise TitleNotFoundError(
            f"Title '{query}' matches {len(chosen)} catalogue rows. "
            "Pass content_type or pick one of the suggestions.",
            suggestions,
        )

    def _suggestion(self, index: int) -> str:
        label = f"{self.titles[index]} [{self.types[index]}, {int(self.years[index])}]"
        peers = self._indices_for_choice(
            str(self.titles[index]), str(self.types[index]), int(self.years[index])
        )
        if len(peers) > 1:
            return f"{label[:-1]} | {self.show_ids[index]}]"
        return label

    def _suggestion_for_title(self, title: str) -> str:
        indices = self._norm_to_indices.get(normalise_title(title), [])
        if len(indices) == 1:
            return self._suggestion(indices[0])
        return title

    def _to_recommendation(
        self,
        query_index: int | None,
        index: int,
        score: float,
        query_text: str | None = None,
    ) -> Recommendation:
        similarity = float(np.clip(score, 0.0, 1.0))
        if query_index is None:
            shared_genres = self._genres_mentioned(query_text or "", index)
            shared_people: list[str] = []
            top_terms = self._query_overlap_terms(query_text or "", index)
            reason = self._reason(
                shared_genres, shared_people, top_terms, similarity, from_text=True
            )
        else:
            shared_genres, shared_people, top_terms, reason = self._explanation(
                query_index, index, similarity
            )
        return Recommendation(
            show_id=str(self.show_ids[index]),
            title=str(self.titles[index]),
            type=str(self.types[index]),
            release_year=int(self.years[index]),
            rating=self.ratings[index],
            genres=list(self.genres[index]),
            similarity=similarity,
            reason=reason,
            shared_genres=shared_genres,
            shared_people=shared_people,
        )

    def _explanation(
        self, left: int, right: int, similarity: float | None
    ) -> tuple[list[str], list[str], list[str], str]:
        shared_genres = _intersect(self.genres[left], self.genres[right])
        cap = self.config.recommender.cast_cap
        shared_people = _intersect(
            self.directors[left] + self.cast_list[left][:cap],
            self.directors[right] + self.cast_list[right][:cap],
        )
        top_terms = self._top_terms(left, right, 5)
        return (
            shared_genres,
            shared_people,
            top_terms,
            self._reason(shared_genres, shared_people, top_terms, similarity, from_text=False),
        )

    def _reason(
        self,
        shared_genres: list[str],
        shared_people: list[str],
        top_terms: list[str],
        similarity: float | None,
        from_text: bool,
    ) -> str:
        """State only what the features support.

        A cosine near 1 with no shared person is a genre-token tie. Calling that
        a confident recommendation would overclaim. There is no viewing history
        anywhere in this catalogue, so the sentence never says a viewer would
        like the title.
        """
        if from_text and shared_genres:
            reason = f"Because the description mentions {_english_list(shared_genres)}."
            if top_terms:
                reason += " Shared words: " + ", ".join(top_terms) + "."
        else:
            clauses: list[str] = []
            if shared_genres:
                clauses.append(f"both are {_english_list(shared_genres)}")
            if shared_people:
                clauses.append(f"share {_english_list(shared_people)}")
            if clauses:
                reason = "Because " + " and ".join(clauses) + "."
            elif top_terms:
                reason = "Because they share wording around " + ", ".join(top_terms) + "."
            else:
                reason = "Because their catalogue profiles overlap."
        if shared_genres and not shared_people and not from_text:
            reason += (
                " Genre overlap only: no shared cast or director, and there is no"
                " viewing history, so this is not evidence a viewer would like it."
            )
            if (
                similarity is not None
                and similarity >= self.config.recommender.saturated_similarity
            ):
                reason += " The score is saturated: other titles with the same genre string tie."
        if (
            from_text
            and similarity is not None
            and similarity < self.config.recommender.weak_text_similarity
        ):
            reason += " Weak match: little of that description is in the catalogue vocabulary."
        return reason

    def _expand_query(self, folded: str) -> str:
        """Add genre plurals and collapsed names the vectoriser actually stored.

        ``drama`` is not the token ``dramas``, and ``Martin Scorsese`` is stored
        as ``martinscorsese``. Without this, those queries are all zeros.
        """
        original = folded.split()
        extra: list[str] = []
        genre_words = self._genre_word_set()
        rewritten: list[str] = []
        for token in original:
            plurals = {token + "s", token + "es"}
            if token.endswith("y") and len(token) > 4:
                plurals.add(token[:-1] + "ies")
            hits = sorted(word for word in plurals if word in genre_words)
            if len(token) >= 4 and hits:
                # ``comedy`` is the stand-up token and drowns ``comedies``.
                rewritten.extend(hits)
            else:
                rewritten.append(token)
        names = self._feature_name_set()
        for width in (2, 3):
            if len(original) < width:
                continue
            for start in range(len(original) - width + 1):
                collapsed = "".join(original[start : start + width])
                if collapsed in names:
                    extra.append(collapsed)
        text = " ".join(rewritten)
        if extra:
            text = text + " " + " ".join(dict.fromkeys(extra))
        return text

    def _genre_word_set(self) -> set[str]:
        cached = getattr(self, "_genre_words_cache", None)
        if isinstance(cached, set):
            return cached
        words: set[str] = set()
        for genres in self.genres:
            for genre in genres:
                words.update(word for word in fold_text(genre).split() if len(word) >= 4)
        self._genre_words_cache = words
        return words

    def _feature_name_set(self) -> set[str]:
        cached = getattr(self, "_feature_name_cache", None)
        if isinstance(cached, set):
            return cached
        names = {str(name) for name in self.feature_names}
        self._feature_name_cache = names
        return names

    def _genres_mentioned(self, folded_query: str, index: int) -> list[str]:
        """Genres on this title whose words appear in a free-text query."""
        tokens = [token for token in folded_query.split() if len(token) >= 4]
        if not tokens:
            return []
        matched: list[str] = []
        for genre in self.genres[index]:
            words = [word for word in fold_text(genre).split() if len(word) >= 4]
            if any(
                word.startswith(token) or token.startswith(word)
                for token in tokens
                for word in words
            ):
                matched.append(genre)
        return matched

    def _top_terms(self, left: int, right: int, k: int) -> list[str]:
        assert self.matrix is not None
        product = self.matrix[left].multiply(self.matrix[right]).tocsr()
        if product.nnz == 0:
            return []
        data = product.data
        columns = product.indices
        take = min(k, len(data))
        if take < len(data):
            picked = np.argpartition(-data, kth=take - 1)[:take]
        else:
            picked = np.arange(len(data))
        picked = picked[np.argsort(-data[picked])]
        return [str(self.feature_names[columns[item]]) for item in picked if data[item] > 0]

    def _query_overlap_terms(self, folded_query: str, index: int) -> list[str]:
        assert self.matrix is not None
        vector = transform_query(self.vectorizer, folded_query)
        product = vector.multiply(self.matrix[index]).tocsr()
        if product.nnz == 0:
            return []
        data = product.data
        columns = product.indices
        take = min(5, len(data))
        picked = np.argsort(-data)[:take]
        return [str(self.feature_names[columns[item]]) for item in picked if data[item] > 0]

    def _require_fitted(self) -> None:
        if not self.fitted or self.matrix is None:
            raise RuntimeError("Recommender is not fitted. Run `make train` first.")


def _intersect(left: list[str], right: list[str]) -> list[str]:
    right_keys = {item.casefold() for item in right}
    found: list[str] = []
    seen: set[str] = set()
    for item in left:
        key = item.casefold()
        if key in right_keys and key not in seen:
            found.append(item)
            seen.add(key)
    return found


def _top_counts(counts: dict[str, int], limit: int) -> list[dict[str, int | str]]:
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [{"name": name, "count": count} for name, count in ordered[:limit]]


def _common_prefix_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index
