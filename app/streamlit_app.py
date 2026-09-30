"""Streamlit client. Talks to the API over HTTP and does not import the model."""

from __future__ import annotations

import html
import os
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode

import httpx
import plotly.graph_objects as go
import streamlit as st

from app.posters import lookup_poster, lookup_synopsis

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
ACCENT = "#E50914"
MATCH = "#46d369"
SERIES = ["#E50914", "#b3b3b3", "#46d369", "#564d4d", "#831010"]

MOODS = [
    "crime",
    "drama",
    "thriller",
    "kids",
    "family",
    "comedy",
    "romance",
    "adventure",
    "action",
    "horror",
    "documentary",
    "mystery",
    "anime",
    "fantasy",
    "sports",
    "music",
    "reality",
    "stand-up",
]
QUICK_TITLES = ["Breaking Bad", "Stranger Things", "The Irishman"]
RATING_GUIDE = {
    "G": "General audiences. Suitable for all ages.",
    "PG": "Parental guidance suggested. Some material may not suit young children.",
    "PG-13": "Parents strongly cautioned. Some material may be inappropriate under 13.",
    "R": "Restricted. Under 17 needs a parent or adult guardian.",
    "NC-17": "Adults only. No one 17 and under.",
    "NR": "Not rated by the film board.",
    "UR": "Unrated. No official rating was assigned.",
    "TV-Y": "Appropriate for all children.",
    "TV-Y7": "Directed to children age 7 and above.",
    "TV-Y7-FV": "Directed to children age 7 and above, with fantasy violence.",
    "TV-G": "General audience. Most parents would find this suitable for all ages.",
    "TV-PG": "Parental guidance suggested for younger children.",
    "TV-14": "Parents strongly cautioned. May be unsuitable under 14.",
    "TV-MA": "Mature audience only. May be unsuitable for ages 17 and under.",
}

st.set_page_config(page_title="Watch next", layout="wide", initial_sidebar_state="collapsed")

_CSS = """
<style>
  .stApp { background: #141414; }
  header[data-testid="stHeader"] { background: transparent; }
  .block-container { padding-top: 1.2rem; max-width: 1180px; }
  .hero {
    padding: 8px 0 18px 0;
    border-bottom: 1px solid #2a2a2a;
    margin-bottom: 18px;
  }
  .eyebrow {
    color: #E50914;
    font-weight: 700;
    letter-spacing: 0.14em;
    font-size: 0.75rem;
    text-transform: uppercase;
    margin: 0;
  }
  .hero h1 {
    color: #ffffff;
    font-size: 2.4rem;
    line-height: 1.1;
    margin: 6px 0 8px 0;
    font-weight: 700;
  }
  .hero p { color: #b3b3b3; margin: 0; font-size: 1.02rem; }
  div[data-testid="stImage"] img {
    width: 100%;
    aspect-ratio: 2 / 3;
    object-fit: cover;
    border-radius: 6px;
    background: #1f1f1f;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
  }
  div[data-testid="stImage"] img:hover {
    transform: scale(1.035);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.55);
  }
  .ph {
    aspect-ratio: 2 / 3;
    border-radius: 6px;
    display: flex;
    align-items: flex-end;
    padding: 14px;
    color: #ffffff;
    font-weight: 700;
    font-size: 1.05rem;
    line-height: 1.25;
  }
  .match { color: #46d369; font-weight: 700; margin: 8px 0 0 0; font-size: 0.95rem; }
  .card-title { color: #ffffff; font-weight: 700; margin: 2px 0 0 0; font-size: 1rem; }
  .card-meta { color: #b3b3b3; margin: 2px 0 0 0; font-size: 0.78rem; }
  .pill {
    display: inline-block;
    border: 1px solid #3a3a3a;
    border-radius: 999px;
    padding: 1px 8px;
    margin: 6px 4px 0 0;
    color: #e5e5e5;
    font-size: 0.72rem;
  }
  .reason { color: #d2d2d2; font-size: 0.78rem; line-height: 1.4; margin: 8px 0 4px 0; }
  .fine { color: #808080; font-size: 0.78rem; }
</style>
"""


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


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def _poster(title: str, content_type: str) -> str:
    return lookup_poster(title, content_type or None) or ""


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def _synopsis(title: str, content_type: str) -> str:
    return lookup_synopsis(title, content_type or None) or ""


def _rating_choice(key: str) -> str | None:
    choice = st.selectbox("Rating", ["Any", *RATING_GUIDE], key=key)
    st.markdown("**What the ratings mean**")
    for code, meaning in RATING_GUIDE.items():
        st.caption(f"{code}: {meaning}")
    if choice == "Any":
        return None
    return str(choice)


def _warm_posters(results: list[dict]) -> None:
    pairs = [(str(item.get("title") or ""), str(item.get("type") or "")) for item in results]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda pair: lookup_poster(pair[0], pair[1] or None), pairs))


def _hue(title: str) -> int:
    return sum(ord(char) for char in title) % 360


def _dark(figure: go.Figure) -> go.Figure:
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e5e5",
        title_font_color="#ffffff",
    )
    figure.update_xaxes(gridcolor="#333333", zerolinecolor="#333333")
    figure.update_yaxes(gridcolor="#333333", zerolinecolor="#333333")
    return figure


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
    return _dark(figure)


def _card(item: dict) -> None:
    title = str(item.get("title") or "Untitled")
    content_type = str(item.get("type") or "")
    url = _poster(title, content_type)
    if url:
        st.image(url, use_container_width=True)
    else:
        safe = html.escape(title)
        hue = _hue(title)
        st.markdown(
            (
                f'<div class="ph" style="background:linear-gradient(165deg,'
                f'hsl({hue},42%,28%),#141414);">{safe}</div>'
            ),
            unsafe_allow_html=True,
        )
    score = float(item.get("similarity") or 0.0)
    percent = round(score * 100)
    rating = item.get("rating") or "Unrated"
    year = item.get("release_year") or ""
    st.markdown(f'<p class="match">{percent}% Match</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="card-title">{html.escape(title)}</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="card-meta">{html.escape(content_type)} · {year} · {html.escape(str(rating))}</p>',
        unsafe_allow_html=True,
    )
    meaning = RATING_GUIDE.get(str(rating))
    if meaning:
        st.caption(meaning)
    genres = item.get("genres") or []
    if genres:
        pills = "".join(
            f'<span class="pill">{html.escape(str(genre))}</span>' for genre in genres[:3]
        )
        st.markdown(pills, unsafe_allow_html=True)
    blurb = str(item.get("description") or "").strip() or _synopsis(title, content_type)
    if blurb:
        st.markdown(f'<p class="reason">{html.escape(blurb)}</p>', unsafe_allow_html=True)
    reason = str(item.get("reason") or "")
    if reason:
        with st.expander("See more"):
            st.write(reason)


def _render_results(payload: dict, heading: str) -> None:
    st.markdown(f"### {heading}")
    if payload.get("resolved_from"):
        st.info(
            f"Showing results for **{payload['query_title']}**. You typed `{payload['resolved_from']}`."
        )
    results = payload.get("results") or []
    if not results:
        st.warning("No titles matched those filters.")
        return
    with st.spinner("Loading artwork..."):
        _warm_posters(results)
    per_row = 5
    for start in range(0, len(results), per_row):
        row = results[start : start + per_row]
        columns = st.columns(per_row)
        for column, item in zip(columns, row, strict=False):
            with column:
                _card(item)
    st.markdown(
        '<p class="fine">A coloured card means no picture was found. '
        "Match % is cosine similarity, not a personal taste score.</p>",
        unsafe_allow_html=True,
    )


def _recommend_tab() -> None:
    st.caption("Start from a title already in the catalogue.")
    if "catalogue_query" not in st.session_state:
        st.session_state["catalogue_query"] = ""
    st.write("Quick picks")
    picks = st.columns(len(QUICK_TITLES))
    for column, title in zip(picks, QUICK_TITLES, strict=True):
        with column:
            if st.button(title, key=f"quick-{title}", use_container_width=True):
                st.session_state["catalogue_query"] = title
                st.session_state["run_recommend"] = True

    query = st.text_input(
        "Search the catalogue",
        key="catalogue_query",
        placeholder="Type a show or movie",
    )
    matches: list[dict] = []
    if query.strip():
        with st.spinner("Searching titles..."):
            response = _request("GET", "/titles?" + urlencode({"q": query, "limit": 20}))
        if response is None:
            st.session_state.pop("run_recommend", None)
            return
        if response.status_code != 200:
            st.session_state.pop("run_recommend", None)
            _error(f"Search failed ({response.status_code}).")
            return
        matches = response.json()
    labels = [f"{item['title']} ({item['type']}, {item['release_year']})" for item in matches]
    if matches and query.strip().casefold() not in str(matches[0]["title"]).casefold():
        st.caption(
            f"Closest catalogue title is {matches[0]['title']}. "
            "Recommend uses the title selected below."
        )
    chosen = st.selectbox(
        "Matching titles",
        labels,
        index=0 if labels else None,
        placeholder="Type a show or movie",
    )
    selected = matches[labels.index(chosen)]["title"] if chosen and labels else query

    with st.expander("Filters", expanded=False):
        col_n, col_type, col_year, col_flag = st.columns(4)
        with col_n:
            n = st.slider("How many", min_value=1, max_value=20, value=10)
        with col_type:
            content_type = st.selectbox("Type", ["Any", "Movie", "TV Show"])
        with col_year:
            min_year = st.slider("Released in or after", min_value=1925, max_value=2021, value=1925)
        with col_flag:
            exclude = st.toggle("Hide sequels and spin-offs", value=True)
        rating = _rating_choice("recommend-rating")

    run = st.button("Recommend", type="primary", disabled=not str(selected).strip())
    auto = st.session_state.pop("run_recommend", False)
    if run or auto:
        body = {
            "title": selected,
            "n": n,
            "content_type": None if content_type == "Any" else content_type,
            "min_release_year": None if min_year <= 1925 else min_year,
            "rating": rating,
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
        st.session_state["last_recommend"] = payload
        st.session_state["last_recommend_ms"] = (
            f"Model {payload['model_version']} · {payload['took_ms']} ms"
        )
    payload = st.session_state.get("last_recommend")
    if payload:
        _render_results(payload, f"More like {payload['query_title']}")
        st.caption(st.session_state.get("last_recommend_ms", ""))


def _toggle_mood(mood: str) -> None:
    words = str(st.session_state.get("taste", "")).split()
    key = mood.casefold()
    if key in {word.casefold() for word in words}:
        words = [word for word in words if word.casefold() != key]
    else:
        words.append(mood)
    st.session_state["taste"] = " ".join(words)


def _taste_tab() -> None:
    st.write("Pick one or more moods. They combine in the box, and you can still type your own.")
    if "taste" not in st.session_state:
        st.session_state["taste"] = ""
    chosen = {word.casefold() for word in str(st.session_state.get("taste", "")).split()}
    per_row = 6
    for start in range(0, len(MOODS), per_row):
        columns = st.columns(per_row)
        for column, mood in zip(columns, MOODS[start : start + per_row], strict=False):
            with column:
                if st.button(
                    mood,
                    key=f"mood-{mood}",
                    type="primary" if mood.casefold() in chosen else "secondary",
                    use_container_width=True,
                ):
                    _toggle_mood(mood)
    text = st.text_area(
        "Your description",
        key="taste",
        height=80,
        placeholder="Type a mood",
    )
    n = st.slider("How many", min_value=1, max_value=20, value=10, key="taste-n")
    with st.expander("Filters", expanded=False):
        taste_rating = _rating_choice("taste-rating")
    if st.button("Find titles", type="primary", disabled=not str(text).strip()):
        with st.spinner("Matching that description..."):
            response = _request(
                "POST", "/recommend/text", {"query": text, "n": n, "rating": taste_rating}
            )
        if response is None:
            return
        if response.status_code != 200:
            detail = response.json().get("detail", "The description could not be matched.")
            _error(detail if isinstance(detail, str) else str(detail))
            return
        payload = response.json()
        st.session_state["last_taste"] = payload
        st.session_state["last_taste_ms"] = (
            f"Model {payload['model_version']} · {payload['took_ms']} ms"
        )
    payload = st.session_state.get("last_taste")
    if payload:
        _render_results(payload, "Closest titles")
        st.caption(st.session_state.get("last_taste_ms", ""))


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
            line={"color": ACCENT, "width": 3},
            marker={"color": "#ffffff", "size": 6},
        )
    )
    year_fig.update_layout(
        title="Titles added per year",
        xaxis_title="Year added",
        yaxis_title="Titles",
        height=360,
    )
    st.plotly_chart(_dark(year_fig), use_container_width=True)

    left, right = st.columns(2)
    with left:
        split = go.Figure(
            go.Pie(
                labels=["Movie", "TV Show"],
                values=[stats["n_movies"], stats["n_tv_shows"]],
                marker={"colors": SERIES[:2]},
                hole=0.55,
            )
        )
        split.update_layout(title="Movies and TV shows", height=380)
        st.plotly_chart(_dark(split), use_container_width=True)
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
            marker_color=ACCENT,
        )
    )
    runtime.update_layout(
        title="Movie runtime", xaxis_title="Runtime bin", yaxis_title="Movies", height=360
    )
    st.plotly_chart(_dark(runtime), use_container_width=True)


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
- This extract has no cast and no synopsis. A lot of similarity is shared genre labels.
- Genre precision uses those same labels, so a high score is a sanity check, not proof the list is good.
- Some country values are wrong (titles tagged Pakistan that are not Pakistani). Country weight is 0 for that reason.
- Stop-word removal is the English list. Non-English descriptions, if you add them later, will be handled badly.
- Popular genres show up in many lists. Coverage and the Gini coefficient in the evaluation report measure that.
- Poster art is looked up live. A missing picture is a failed lookup, not a claim that the title has no art.
        """
    )


st.markdown(_CSS, unsafe_allow_html=True)
st.markdown(
    """
<div class="hero">
  <p class="eyebrow">Catalogue</p>
  <h1>Find something to watch</h1>
  <p>Similar titles from this catalogue, each with a reason. Content-based — not your watch history.</p>
</div>
""",
    unsafe_allow_html=True,
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
