"""Artwork URLs for catalogue titles.

Netflix does not publish a poster API. This module asks, in order:

1. TMDB ``search/multi`` when ``TMDB_API_KEY`` is set (optional, sharper stills).
2. TVMaze for TV shows (no key, portrait images).
3. The English Wikipedia REST summary (no key; films usually have a poster).

Misses return ``None``. Callers draw a title card. Image bytes are not stored.
"""

from __future__ import annotations

import os
import re
import unicodedata
from functools import lru_cache
from urllib.parse import quote

import httpx

USER_AGENT = "NetflixRecommender/1.0 (educational catalogue UI)"
_TIMEOUT = 6.0


def _norm(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", folded.lower())


def _close(query: str, candidate: str) -> bool:
    left, right = _norm(query), _norm(candidate)
    if not left or not right:
        return False
    return left == right or left in right or right in left


def _is_logo(url: str) -> bool:
    path = url.lower().split("?", 1)[0]
    return "logo" in path or path.endswith(".svg")


def _get(
    client: httpx.Client, url: str, params: dict[str, str] | None = None
) -> httpx.Response | None:
    try:
        response = client.get(url, params=params)
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    return response


def _tmdb(client: httpx.Client, title: str, content_type: str | None) -> str | None:
    key = os.environ.get("TMDB_API_KEY", "").strip()
    if not key:
        return None
    response = _get(
        client,
        "https://api.themoviedb.org/3/search/multi",
        {"api_key": key, "query": title, "include_adult": "false"},
    )
    if response is None:
        return None
    wanted = {"Movie": "movie", "TV Show": "tv"}.get(content_type or "")
    for item in response.json().get("results") or []:
        if not isinstance(item, dict):
            continue
        if wanted and item.get("media_type") != wanted:
            continue
        name = str(item.get("title") or item.get("name") or "")
        path = item.get("poster_path")
        if path and _close(title, name):
            return f"https://image.tmdb.org/t/p/w342{path}"
    return None


def _tvmaze(client: httpx.Client, title: str) -> str | None:
    response = _get(client, "https://api.tvmaze.com/search/shows", {"q": title})
    if response is None:
        return None
    rows = response.json()
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        show = row.get("show") or {}
        if not isinstance(show, dict) or not _close(title, str(show.get("name") or "")):
            continue
        image = show.get("image") or {}
        if not isinstance(image, dict):
            continue
        url = image.get("original") or image.get("medium")
        if isinstance(url, str) and url:
            return url
    return None


def _wiki_summary(client: httpx.Client, title: str) -> tuple[str, str] | None:
    slug = quote(title.strip().replace(" ", "_"), safe=":_()'%!-")
    response = _get(client, f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}")
    if response is None:
        return None
    body = response.json()
    if not isinstance(body, dict) or body.get("type") == "disambiguation":
        return None
    page_title = str(body.get("title") or "")
    image = body.get("originalimage") or body.get("thumbnail") or {}
    source = image.get("source") if isinstance(image, dict) else None
    if not isinstance(source, str) or not source or not _close(title, page_title):
        return None
    return page_title, source


def _wiki(client: httpx.Client, title: str) -> str | None:
    direct = _wiki_summary(client, title)
    if direct is not None:
        return direct[1]
    response = _get(
        client,
        "https://en.wikipedia.org/w/api.php",
        {
            "action": "opensearch",
            "search": title,
            "limit": "1",
            "namespace": "0",
            "format": "json",
        },
    )
    if response is None:
        return None
    payload = response.json()
    hits = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
    if not hits or not isinstance(hits[0], str):
        return None
    found = _wiki_summary(client, hits[0])
    if found is None or not _close(title, found[0]):
        return None
    return found[1]


def _client() -> httpx.Client:
    return httpx.Client(
        timeout=_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )


@lru_cache(maxsize=512)
def lookup_poster(title: str, content_type: str | None = None) -> str | None:
    """Return an image URL for ``title``, or ``None`` when every source misses."""
    cleaned = title.strip()
    if not cleaned:
        return None
    try:
        with _client() as client:
            tmdb = _tmdb(client, cleaned, content_type)
            if tmdb:
                return tmdb
            show = _tvmaze(client, cleaned) if content_type != "Movie" else None
            wiki = _wiki(client, cleaned)
            if content_type == "Movie":
                ordered = [wiki, show]
            else:
                ordered = [show, wiki]
            plain = [url for url in ordered if url and not _is_logo(url)]
            if plain:
                return plain[0]
            logos = [url for url in ordered if url]
            return logos[0] if logos else None
    except (httpx.HTTPError, ValueError, TypeError, KeyError):
        return None
