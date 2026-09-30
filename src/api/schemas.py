"""Pydantic models for every API boundary."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RecommendRequest(BaseModel):
    """Ask for neighbours of a catalogue title."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "title": "Breaking Bad",
                    "n": 5,
                    "content_type": "TV Show",
                    "min_release_year": 1990,
                    "exclude_same_franchise": True,
                }
            ]
        }
    )

    title: str = Field(min_length=1, max_length=300)
    n: int = Field(default=10, ge=1, le=50)
    content_type: Literal["Movie", "TV Show"] | None = None
    min_release_year: int | None = Field(default=None, ge=1900, le=2100)
    rating: str | None = Field(default=None, max_length=20)
    exclude_same_franchise: bool = True


class TextRecommendRequest(BaseModel):
    """Cold-start query that is not a catalogue title."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"query": "crime drama thriller", "n": 5}]}
    )

    query: str = Field(min_length=1, max_length=1000)
    n: int = Field(default=10, ge=1, le=50)
    rating: str | None = Field(default=None, max_length=20)


class RecommendationModel(BaseModel):
    show_id: str
    title: str
    type: str
    release_year: int
    rating: str | None
    description: str = ""
    genres: list[str]
    similarity: float = Field(ge=0.0, le=1.0)
    reason: str
    shared_genres: list[str]
    shared_people: list[str]


class RecommendResponse(BaseModel):
    query_title: str
    resolved_from: str | None = None
    results: list[RecommendationModel]
    took_ms: float
    model_version: str


class TextRecommendResponse(BaseModel):
    query: str
    results: list[RecommendationModel]
    took_ms: float
    model_version: str


class TitleBrief(BaseModel):
    show_id: str
    title: str
    type: str
    release_year: int
    rating: str | None = None


class TitleDetail(TitleBrief):
    genres: list[str]
    directors: list[str]
    cast: list[str]
    countries: list[str]
    description: str
    date_added: str


class NamedCount(BaseModel):
    name: str
    count: int


class YearCount(BaseModel):
    year: int
    count: int


class RuntimeBin(BaseModel):
    bin_start: float
    bin_end: float
    count: int


class StatsResponse(BaseModel):
    n_titles: int
    n_movies: int
    n_tv_shows: int
    added_per_year: list[YearCount]
    top_genres: list[NamedCount]
    top_countries: list[NamedCount]
    ratings: list[NamedCount]
    movie_runtime_bins: list[RuntimeBin]


class MetaResponse(BaseModel):
    """Training metadata. Extra keys from ``metadata.json`` are preserved."""

    model_config = ConfigDict(extra="allow")

    trained_at: str | None = None
    git_commit: str | None = None
    data_sha256: str | None = None
    row_count: int | None = None
    library_versions: dict[str, str] | None = None
    metrics: dict[str, Any] | None = None
    random_seed: int | None = None
    model_version: str | None = None
