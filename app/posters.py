"""Artwork URLs for catalogue titles.

Netflix does not publish a poster API. This module asks, in order:

1. TMDB ``search/multi`` when ``TMDB_API_KEY`` is set (optional, sharper stills).
2. TVMaze for TV shows (no key, portrait images).
3. The English Wikipedia REST summary (no key; films usually have a poster).

Misses return ``None``. Callers draw a title card. Image bytes are not stored.
"""

from __future__ import annotations

import html
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


def _plain(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _wiki_summary(client: httpx.Client, title: str) -> tuple[str, str | None, str | None] | None:
    slug = quote(title.strip().replace(" ", "_"), safe=":_()'%!-")
    response = _get(client, f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}")
    if response is None:
        return None
    body = response.json()
    if not isinstance(body, dict) or body.get("type") == "disambiguation":
        return None
    page_title = str(body.get("title") or "")
    if not _close(title, page_title):
        return None
    image = body.get("originalimage") or body.get("thumbnail") or {}
    source = image.get("source") if isinstance(image, dict) else None
    picture = source if isinstance(source, str) and source else None
    extract = body.get("extract")
    synopsis = _plain(extract) if isinstance(extract, str) and extract.strip() else None
    if picture is None and synopsis is None:
        return None
    return page_title, picture, synopsis


def _wiki_page(client: httpx.Client, title: str) -> tuple[str | None, str | None]:
    direct = _wiki_summary(client, title)
    if direct is not None:
        return direct[1], direct[2]
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
        return None, None
    payload = response.json()
    hits = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
    if not hits or not isinstance(hits[0], str):
        return None, None
    found = _wiki_summary(client, hits[0])
    if found is None or not _close(title, found[0]):
        return None, None
    return found[1], found[2]


def _tvmaze_show(client: httpx.Client, title: str) -> tuple[str | None, str | None]:
    response = _get(client, "https://api.tvmaze.com/search/shows", {"q": title})
    if response is None:
        return None, None
    rows = response.json()
    if not isinstance(rows, list):
        return None, None
    for row in rows:
        if not isinstance(row, dict):
            continue
        show = row.get("show") or {}
        if not isinstance(show, dict) or not _close(title, str(show.get("name") or "")):
            continue
        image = show.get("image") or {}
        picture = None
        if isinstance(image, dict):
            url = image.get("original") or image.get("medium")
            picture = url if isinstance(url, str) and url else None
        summary = show.get("summary")
        synopsis = _plain(summary) if isinstance(summary, str) and summary.strip() else None
        return picture, synopsis
    return None, None


def _client() -> httpx.Client:
    return httpx.Client(
        timeout=_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )


@lru_cache(maxsize=512)
def lookup_art(title: str, content_type: str | None = None) -> tuple[str | None, str | None]:
    """Return ``(poster URL, short synopsis)``. Either side may be missing."""
    cleaned = title.strip()
    if not cleaned:
        return None, None
    try:
        with _client() as client:
            poster = _tmdb(client, cleaned, content_type)
            if content_type == "Movie":
                wiki_poster, wiki_text = _wiki_page(client, cleaned)
                show_poster, show_text = None, None
            else:
                show_poster, show_text = _tvmaze_show(client, cleaned)
                wiki_poster, wiki_text = _wiki_page(client, cleaned)
            pictures = [poster, show_poster, wiki_poster]
            if content_type == "Movie":
                pictures = [poster, wiki_poster, show_poster]
            plain = [url for url in pictures if url and not _is_logo(url)]
            logos = [url for url in pictures if url]
            chosen = plain[0] if plain else (logos[0] if logos else None)
            synopsis = wiki_text or show_text
            if content_type != "Movie":
                synopsis = show_text or wiki_text
            return chosen, synopsis
    except (httpx.HTTPError, ValueError, TypeError, KeyError):
        return None, None


def lookup_poster(title: str, content_type: str | None = None) -> str | None:
    """Return an image URL for ``title``, or ``None`` when every source misses."""
    return lookup_art(title, content_type)[0]


def lookup_synopsis(title: str, content_type: str | None = None) -> str | None:
    """Return a short plot line, or ``None`` when no source has one."""
    return lookup_art(title, content_type)[1]
