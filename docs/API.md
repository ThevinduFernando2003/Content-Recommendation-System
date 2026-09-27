# API

Base URL for a local process: `http://127.0.0.1:8000`.

The JSON below was returned by that process on 27 September 2026 after `scripts/train.py` and `scripts/evaluate.py`. Re-run the curls to refresh it. Every response includes an `X-Request-ID` header.

Interactive docs: `http://127.0.0.1:8000/docs`.

## GET /health

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status": "ok"}
```

## GET /ready

`200` after artifacts load. `503` with `{"detail":"Model artifacts are not loaded"}` before that.

```bash
curl http://127.0.0.1:8000/ready
```

```json
{"status": "ready"}
```

## GET /meta

```bash
curl http://127.0.0.1:8000/meta
```

```json
{
  "trained_at": "2026-09-27T07:16:06Z",
  "git_commit": "3a7b256e6fa2af08019da9a9ed81539eaa01e5f4",
  "data_sha256": "c7cf60a4429efddb009439488b6929e62d3d2f69f5da16e068bc9bfb852390d3",
  "source_csv": "data/raw/netflix_titles.csv",
  "row_count": 8790,
  "n_features": 4256,
  "nnz": 136936,
  "random_seed": 42,
  "vectorizer_kind": "tfidf",
  "model_version": "3a7b256@2026-09-27T07:16:06Z",
  "similarity": "sparse_dot_l2_normalised",
  "metrics": {
    "genre_precision_at_10": 0.9973265073947668,
    "random_genre_precision_at_10": 0.24219567690557453,
    "recommend_p95_ms": 14.916595071554184
  }
}
```

The live body also includes `feature_weights`, `library_versions`, and the rest of `metrics`. The numbers above are the fields from that response, not a shortened invention.

## GET /titles?q=&limit=

```bash
curl "http://127.0.0.1:8000/titles?q=breaking%20bad&limit=3"
```

```json
[
  {"show_id": "s5941", "title": "Breaking Bad", "type": "TV Show", "release_year": 2013, "rating": "TV-MA"},
  {"show_id": "s3365", "title": "The Road to El Camino: Behind the Scenes of El Camino: A Breaking Bad Movie", "type": "Movie", "release_year": 2019, "rating": "TV-14"},
  {"show_id": "s3429", "title": "El Camino: A Breaking Bad Movie", "type": "Movie", "release_year": 2019, "rating": "TV-MA"}
]
```

## GET /titles/{show_id}

```bash
curl http://127.0.0.1:8000/titles/s5941
```

```json
{
  "show_id": "s5941",
  "title": "Breaking Bad",
  "type": "TV Show",
  "release_year": 2013,
  "rating": "TV-MA",
  "genres": ["Crime TV Shows", "TV Dramas", "TV Thrillers"],
  "directors": [],
  "cast": [],
  "countries": ["United States"],
  "description": "",
  "date_added": "2013-02-08"
}
```

Unknown `show_id` is `404` with `{"detail": "No title with show_id '...'."}`.

## POST /recommend

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d "{\"title\":\"breakng bad\",\"n\":1}"
```

```json
{
  "query_title": "Breaking Bad",
  "resolved_from": "breakng bad",
  "results": [
    {
      "show_id": "s2768",
      "title": "Ozark",
      "type": "TV Show",
      "release_year": 2020,
      "rating": "TV-MA",
      "genres": ["Crime TV Shows", "TV Dramas", "TV Thrillers"],
      "similarity": 1.0,
      "reason": "Because both are Crime TV Shows, TV Dramas and TV Thrillers.",
      "shared_genres": ["Crime TV Shows", "TV Dramas", "TV Thrillers"],
      "shared_people": []
    }
  ],
  "took_ms": 90.084,
  "model_version": "3a7b256@2026-09-27T07:16:06Z"
}
```

`took_ms` moves between calls. That value is the one returned for this typo on the running server. An exact title leaves `resolved_from` null.

Unknown title, from `{"title":"zzzznotitle","n":3}`:

```json
{
  "detail": "No catalogue title matches 'zzzznotitle'.",
  "suggestions": ["Zozo [Movie, 2005]", "Battle [Movie, 2018]", "Karzzzz [Movie, 2008]", "Glee [TV Show, 2015]", "Elite [TV Show, 2021]"]
}
```

`n: 0` is `422`:

```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": ["body", "n"],
      "msg": "Input should be greater than or equal to 1",
      "input": 0,
      "ctx": {"ge": 1}
    }
  ]
}
```

The recommend routes are limited to `RATE_LIMIT` (default `60/minute` per IP). Over the limit the body is `429` with `detail` and `request_id`.

## POST /recommend/text

```bash
curl -X POST http://127.0.0.1:8000/recommend/text \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"crime drama thriller\",\"n\":2}"
```

```json
{
  "query": "crime drama thriller",
  "results": [
    {
      "show_id": "s669",
      "title": "Hap and Leonard",
      "type": "TV Show",
      "release_year": 2018,
      "rating": "TV-MA",
      "genres": ["Crime TV Shows", "TV Dramas"],
      "similarity": 0.17551030218601227,
      "reason": "Because they share wording around crime.",
      "shared_genres": [],
      "shared_people": []
    },
    {
      "show_id": "s922",
      "title": "StartUp",
      "type": "TV Show",
      "release_year": 2018,
      "rating": "TV-MA",
      "genres": ["Crime TV Shows", "TV Dramas"],
      "similarity": 0.17551030218601227,
      "reason": "Because they share wording around crime.",
      "shared_genres": [],
      "shared_people": []
    }
  ],
  "took_ms": 10.715,
  "model_version": "3a7b256@2026-09-27T07:16:06Z"
}
```

A query with no in-vocabulary tokens, `{"query":"zzzzqqqq","n":2}`, is `400`:

```json
{
  "detail": "That description has no words in common with the catalogue. Try genres or places, for example 'crime drama' or 'kids comedy'."
}
```

## GET /stats

```bash
curl http://127.0.0.1:8000/stats
```

Returned body, in full:

- `n_titles`: 8790
- `n_movies`: 6126
- `n_tv_shows`: 2664
- `added_per_year`: 2008=2, 2009=2, 2010=1, 2011=13, 2012=3, 2013=11, 2014=24, 2015=82, 2016=426, 2017=1185, 2018=1648, 2019=2016, 2020=1879, 2021=1498
- `top_genres` starts International Movies 2752, Dramas 2426, Comedies 1674 (15 rows)
- `top_countries` starts United States 3240, India 1057, United Kingdom 638, Pakistan 421 (15 rows)
- `ratings` starts TV-MA 3205, TV-14 2157, TV-PG 861 (14 rows)
- `movie_runtime_bins`: 0–60 min 457, 60–90 1380, 90–120 3091, 120–150 936, 150–180 213, 180–240 46, 240–400 3

Each list entry is `{"name","count"}` or `{"year","count"}` or `{"bin_start","bin_end","count"}`.

## GET /metrics

Prometheus text. After the calls above, the scraped counters included:

```
http_requests_total{method="GET",path="/health",status="200"} 1.0
http_requests_total{method="POST",path="/recommend",status="200"} 6.0
http_requests_total{method="POST",path="/recommend",status="404"} 2.0
http_requests_total{method="POST",path="/recommend",status="422"} 1.0
http_requests_total{method="POST",path="/recommend/text",status="400"} 1.0
```

Those counts are from one local process and will differ after more traffic.
