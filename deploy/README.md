# Deploy

Nothing in this folder has been applied to a host. There is no public URL yet.
The steps below are what to run once you have an account. Do not commit tokens.

The image is built from the repository `Dockerfile`. The builder stage runs
`scripts/train.py`, so the image contains the model and does not need the CSV
at runtime. Set `PORT` to whatever the platform injects. The UI reads
`API_BASE_URL`.

## Environment

| Variable | Purpose | Example |
| --- | --- | --- |
| `PORT` | Port the API or the UI listens on | `8000` |
| `API_BASE_URL` | Where Streamlit sends HTTP calls | `http://api:8000` |
| `CORS_ORIGINS` | Allowed browser origins, comma-separated | `https://your-ui.example` |
| `RATE_LIMIT` | slowapi limit on the recommend routes | `60/minute` |
| `LOG_LEVEL` | Python log level | `INFO` |
| `MODELS_DIR` | Optional override for the artifact directory | `models` |

## Hugging Face Spaces (Docker)

1. Create a Space with the Docker SDK.
2. Push this repository, including `data/raw/netflix_titles.csv`. The file is required at image build time.
3. Put `deploy/spaces.yaml` contents into the Space README YAML front matter, or create two Spaces if you want the API and the UI separated.
4. For a single Space running the API, the Dockerfile `CMD` is already uvicorn. Open `/docs`.
5. For the UI, override the start command with:

```
streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true
```

and set `API_BASE_URL` to the API Space URL (https, no trailing slash).
Add that UI origin to the API Space `CORS_ORIGINS`.

6. After the Space is healthy, from your machine:

```
curl https://YOUR-API/health
curl https://YOUR-API/meta
curl -X POST https://YOUR-API/recommend -H "Content-Type: application/json" -d "{\"title\":\"Breaking Bad\",\"n\":5}"
```

Paste those responses into the README. Do not invent them.

Free tier is enough for the sparse index (about 2 MB of artifacts). The first build is the slow part, because it trains.

## Render

`deploy/render.yaml` defines an API web service and a Streamlit web service from the same Dockerfile.
Render's free web services sleep when idle. The first request after sleep can take a minute while the container starts. Say that next to the URL so an evaluator does not treat the delay as a bug.

Create the services from the blueprint, set `CORS_ORIGINS` to the UI URL Render assigns, and set the UI `API_BASE_URL` to the API URL.

## Fly.io

`deploy/fly.toml` is a starting point for the API process. Run `fly launch` in a clone and replace the generated app name. Scale memory to 1 GB. The sparse matrix does not need more.

## What was not done

No Hugging Face, Render, or Fly credentials were available in this environment, so the services were not deployed and no live URL is claimed.
