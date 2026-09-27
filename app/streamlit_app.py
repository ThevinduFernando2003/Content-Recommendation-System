"""Streamlit client. Talks to the API over HTTP and does not import the model."""

from __future__ import annotations

import os
from urllib.parse import urlencode

import httpx
import plotly.graph_objects as go
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
ACCENT = "#b20710"
SERIES = ["#b20710", "#1f4e79", "#0f766e", "#b45309", "#6b7280"]

EXAMPLES = [
    "crime drama thriller",
    "kids family comedy",
    "stand-up comedy",
]

st.set_page_config(page_title="Netflix recommendations", layout="wide")


def _error(message: str) -> None:
    st.error(message)


def _request(method: str, path: str, payload: dict | None = None) -> httpx.Response | None:
    try:
        response = httpx.request(method, f"{API_BASE_URL}{path}", json=payload, timeout=30.0)
    except httpx.HTTPError:
        _error(
            "The recommendation API is unreachable. Start it with `make api` "
            f"(expected at {API_BASE_URL}) and refresh this page."
        )
        return None
    return response


@st.cache_data(ttl=300)
def _cached_get(path: str) -> dict:
    response = httpx.get(f"{API_BASE_URL}{path}", timeout=30.0)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError("Expected a JSON object")
    return body


def _bars(names: list[str], values: list[int], title: str, x_title: str) -> go.Figure:
    figure = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=ACCENT))
    figure.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="",
        yaxis={"categoryorder": "total ascending"},
        margin={"l": 20, "r": 20, "t": 48, "b": 40},
        height=420,
    )
    return figure


def _render_results(payload: dict, heading: str) -> None:
    st.subheader(heading)
    if payload.get("resolved_from"):
        st.info(
            f"Showing results for **{payload['query_title']}**. You typed `{payload['resolved_from']}`."
        )
    results = payload.get("results") or []
    if not results:
        st.warning("No titles matched those filters.")
        return
    for item in results:
        with st.container(border=True):
            left, right = st.columns([4, 1])
            with left:
                st.markdown(f"**{item['title']}**")
                rating = item.get("rating") or "Unrated"
                st.caption(f"{item['type']} · {item['release_year']} · {rating}")
            with right:
                st.metric("Similarity", f"{item['similarity']:.2f}")
            st.progress(min(max(float(item["similarity"]), 0.0), 1.0))
            genres = item.get("genres") or []
            if genres:
                st.write(" ".join(f"`{genre}`" for genre in genres))
            st.write(item.get("reason") or "")


def _recommend_tab() -> None:
    query = st.text_input(
        "Search the catalogue", placeholder="Try breaking bad or a close spelling"
    )
    matches: list[dict] = []
    if query.strip():
        with st.spinner("Searching titles..."):
            response = _request("GET", "/titles?" + urlencode({"q": query, "limit": 20}))
        if response is None:
            return
        if response.status_code != 200:
            _error(f"Search failed ({response.status_code}).")
            return
        matches = response.json()
    labels = [f"{item['title']} ({item['type']}, {item['release_year']})" for item in matches]
    chosen = st.selectbox(
        "Matching titles", labels, index=0 if labels else None, placeholder="Type a title above"
    )
    selected = matches[labels.index(chosen)]["title"] if chosen and labels else query

    col_n, col_type, col_year, col_flag = st.columns(4)
    with col_n:
        n = st.slider("How many", min_value=1, max_value=20, value=8)
    with col_type:
        content_type = st.selectbox("Type", ["Any", "Movie", "TV Show"])
    with col_year:
        min_year = st.slider("Released in or after", min_value=1925, max_value=2021, value=1925)
    with col_flag:
        exclude = st.toggle("Hide sequels and spin-offs", value=True)

    if st.button("Recommend", type="primary", disabled=not str(selected).strip()):
        body = {
            "title": selected,
            "n": n,
            "content_type": None if content_type == "Any" else content_type,
            "min_release_year": None if min_year <= 1925 else min_year,
            "exclude_same_franchise": exclude,
        }
        with st.spinner("Finding similar titles..."):
            response = _request("POST", "/recommend", body)
        if response is None:
            return
        if response.status_code == 404:
            detail = response.json()
            _error(detail.get("detail", "Title not found."))
            suggestions = detail.get("suggestions") or []
            if suggestions:
                st.write("Did you mean:")
                for suggestion in suggestions:
                    st.write(f"- {suggestion}")
            return
        if response.status_code != 200:
            _error(response.json().get("detail", f"Request failed ({response.status_code})."))
            return
        payload = response.json()
        _render_results(payload, f"Because you asked about {payload['query_title']}")
        st.caption(f"Model {payload['model_version']} · {payload['took_ms']} ms")


def _taste_tab() -> None:
    st.write("Describe a mood. The model only knows words that appear in the catalogue.")
    if "taste" not in st.session_state:
        st.session_state["taste"] = EXAMPLES[0]
    cols = st.columns(len(EXAMPLES))
    for column, example in zip(cols, EXAMPLES, strict=True):
        with column:
            if st.button(example, key=f"ex-{example}"):
                st.session_state["taste"] = example
    text = st.text_area("Your description", key="taste", height=100)
    n = st.slider("How many", min_value=1, max_value=20, value=8, key="taste-n")
    if st.button("Find titles", type="primary"):
        with st.spinner("Matching that description..."):
            response = _request("POST", "/recommend/text", {"query": text, "n": n})
        if response is None:
            return
        if response.status_code != 200:
            detail = response.json().get("detail", "The description could not be matched.")
            _error(detail if isinstance(detail, str) else str(detail))
            return
        payload = response.json()
        _render_results(payload, "Closest titles")
        st.caption(f"Model {payload['model_version']} · {payload['took_ms']} ms")


def _explore_tab() -> None:
    try:
        with st.spinner("Loading catalogue stats..."):
            stats = _cached_get("/stats")
    except httpx.HTTPError:
        _error(f"The recommendation API is unreachable at {API_BASE_URL}.")
        return
    except ValueError as exc:
        _error(str(exc))
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Titles", stats["n_titles"])
    c2.metric("Movies", stats["n_movies"])
    c3.metric("TV shows", stats["n_tv_shows"])

    years = stats["added_per_year"]
    year_fig = go.Figure(
        go.Scatter(
            x=[row["year"] for row in years],
            y=[row["count"] for row in years],
            mode="lines+markers",
            line={"color": ACCENT},
        )
    )
    year_fig.update_layout(
        title="Titles added per year",
        xaxis_title="Year added",
        yaxis_title="Titles",
        height=360,
    )
    st.plotly_chart(year_fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        split = go.Figure(
            go.Pie(
                labels=["Movie", "TV Show"],
                values=[stats["n_movies"], stats["n_tv_shows"]],
                marker={"colors": SERIES[:2]},
                hole=0.45,
            )
        )
        split.update_layout(title="Movies and TV shows", height=380)
        st.plotly_chart(split, use_container_width=True)
    with right:
        ratings = stats["ratings"]
        st.plotly_chart(
            _bars(
                [row["name"] for row in ratings],
                [row["count"] for row in ratings],
                "Rating distribution",
                "Titles",
            ),
            use_container_width=True,
        )

    g_left, g_right = st.columns(2)
    with g_left:
        genres = stats["top_genres"]
        st.plotly_chart(
            _bars(
                [row["name"] for row in genres],
                [row["count"] for row in genres],
                "Top 15 genres",
                "Titles",
            ),
            use_container_width=True,
        )
    with g_right:
        countries = stats["top_countries"]
        st.plotly_chart(
            _bars(
                [row["name"] for row in countries],
                [row["count"] for row in countries],
                "Top 15 countries",
                "Titles",
            ),
            use_container_width=True,
        )

    bins = stats["movie_runtime_bins"]
    runtime = go.Figure(
        go.Bar(
            x=[f"{int(row['bin_start'])}–{int(row['bin_end'])} min" for row in bins],
            y=[row["count"] for row in bins],
            marker_color="#1f4e79",
        )
    )
    runtime.update_layout(
        title="Movie runtime", xaxis_title="Runtime bin", yaxis_title="Movies", height=360
    )
    st.plotly_chart(runtime, use_container_width=True)


def _about_tab() -> None:
    try:
        with st.spinner("Loading model metadata..."):
            meta = _cached_get("/meta")
    except httpx.HTTPError:
        _error(f"The recommendation API is unreachable at {API_BASE_URL}.")
        return
    metrics = meta.get("metrics") or {}
    st.subheader("How a recommendation is chosen")
    st.write(
        "Each title becomes a short text document: genres repeated so they dominate, "
        "plus director, type, and rating. Those documents are turned into TF-IDF vectors, "
        "each scaled to length 1. The similarity score is the dot product of two vectors, "
        "which is cosine similarity. The service stores the sparse matrix and computes one "
        "row of scores per request. It does not store a giant table of every pair."
    )
    if metrics:
        cols = st.columns(3)
        cols[0].metric("Genre precision@10", f"{metrics.get('genre_precision_at_10', 0):.3f}")
        cols[1].metric(
            "Random precision@10", f"{metrics.get('random_genre_precision_at_10', 0):.3f}"
        )
        cols[2].metric("Coverage@10", f"{metrics.get('coverage_at_10', 0):.3f}")
        st.caption("Genre precision is a proxy. It is not accuracy. See the evaluation report.")
    else:
        st.info("Headline metrics appear here after `make evaluate`.")
    st.write(
        f"Trained at {meta.get('trained_at') or 'unknown'} · "
        f"commit {meta.get('git_commit') or 'unknown'} · "
        f"{meta.get('row_count') or 'unknown'} titles · "
        f"seed {meta.get('random_seed') or 'unknown'}."
    )
    st.subheader("Limitations")
    st.markdown(
        """
- There is no viewing history, so the model cannot learn that people who watched one title watched another.
- This extract has no cast and no synopsis. A lot of 'similarity' is shared genre labels.
- Genre precision uses those same labels, so a high score is a sanity check, not proof the list is good.
- Some country values are wrong (titles tagged Pakistan that are not Pakistani). Country weight is 0 for that reason.
- Stop-word removal is the English list. Non-English descriptions, if you add them later, will be handled badly.
- Popular genres show up in many lists. Coverage and the Gini coefficient in the evaluation report measure that.
        """
    )


st.title("Netflix content recommendations")
st.caption(
    "Similar titles from the catalogue, with a reason for each one. Content-based — not personalised."
)

health = _request("GET", "/health")
if health is None:
    st.stop()
if health.status_code != 200:
    _error("The API process is up but not healthy.")
    st.stop()

recommend_tab, taste_tab, explore_tab, about_tab = st.tabs(
    ["Recommend", "Describe your taste", "Explore the catalogue", "About the model"]
)
with recommend_tab:
    _recommend_tab()
with taste_tab:
    _taste_tab()
with explore_tab:
    _explore_tab()
with about_tab:
    _about_tab()
