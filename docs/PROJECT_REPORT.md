# Project report

What was built, which file was actually used as the dataset, and what is still left.

## Did we use Dataset.csv?

There is no file named `dataset.csv` or `Dataset.csv` in this repository.

The file that was here at the start is **`Dataset.csv.xlsx`**, an Excel workbook whose name ends in `.csv.xlsx`. That workbook is the source. It was not fed straight into the model.

`scripts/prepare_raw.py` reads `Dataset.csv.xlsx` (sheet of catalogue rows) and writes **`data/raw/netflix_titles.csv`**. Every later step uses that CSV:

- `scripts/profile_data.py` profiles it and writes `reports/data_profile.md`
- `scripts/train.py` loads it, cleans it, and fits the model
- `scripts/evaluate.py` loads the same CSV again for the ablation fits
- the Docker build copies `data/raw/netflix_titles.csv` and trains inside the image

So yes: the workbook you provided is the dataset. The system runs on the CSV extracted from it, after the repairs below. The public Kaggle/TidyTuesday Netflix file (the one that has `cast` and `description`) was **not** substituted in. Where that public file was used at all, it was only to recover a handful of titles Excel had turned into dates and numbers.

## What the file actually contains

`reports/data_profile.md` is the measured profile, not the brief’s expected table.

| Fact | Measured value |
| --- | --- |
| Rows | 8,790 (the brief expected about 8,791) |
| Columns in the workbook / CSV | 10 |
| Columns the brief expected | 12, including `cast` and `description` |
| Columns present | `show_id`, `type`, `title`, `director`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in` |
| `cast` | absent |
| `description` | absent |
| Missing director | 2,588 rows (29.44%), stored as the text `Not Given`, not as a blank cell |
| Missing country | 287 rows (3.27%), same sentinel |
| Rows labelled Pakistan | 421, including titles that are not Pakistani |
| Multi-country cells | 0 |
| Rating values that are really a duration (`74 min` and similar) | 0 in this file |
| Duration strings that fail to parse | 0 |
| Duplicate `(title, type, release_year)` | 3 pairs, after Excel date titles were restored |
| `release_year` | 1925 to 2021 |

The pipeline treats `cast` and `description` as optional. When they are missing they become an empty string. They are never filled with the word `Unknown`, because that word would make unrelated titles look similar.

## What was done

### 1. Turn the workbook into a CSV

`scripts/prepare_raw.py` does three repairs and leaves `Not Given` for the cleaner:

- Excel had coerced some titles into dates, times, or numbers (`Feb-09`, `15-Aug`, `22-Jul`, `Oct-01`, `23:59`, `7:19`, `2,215`, `3%`). Those are restored only when the cell still looks coerced. The film titled `Unknown` is left alone.
- Some text was UTF-8 bytes stored as Latin-1 (for example Pokémon). Those cells are reversed. The prepare run recorded 157 restored titles and 311 mojibake cells.
- `date_added` was a mix of Excel datetimes and strings. The CSV stores ISO dates (`2021-09-25`).

The repair log is `data/interim/extract_repairs.json`.

### 2. Profile the real file

`scripts/profile_data.py` writes `reports/data_profile.md`. The real file wins over the brief. The differences are written in that report, not papered over.

### 3. Clean the catalogue

`src/data/load.py` checks the column contract and raises a clear error if a required column is missing. `src/data/clean.py` then:

- moves a duration that landed in `rating` into `duration` and clears `rating` (0 rows in this extract; the code and the 60-row test fixture still cover it)
- parses movie length into `duration_minutes` and TV length into `duration_seasons`, never one shared number
- splits comma-separated `director`, `cast`, `country`, and `listed_in`
- drops sentinel tokens from person and place fields
- deduplicates on `show_id` only

`scripts/train.py` writes the cleaned table to `data/processed/titles.parquet` (8,790 titles).

### 4. Build the text features and the model

Each title becomes one weighted document. Multi-word names are collapsed so `Martin Scorsese` is the token `martinscorsese`, not the shared word `martin`.

Shipped weights in `config.yaml`:

| Field | Times repeated | Why |
| --- | --- | --- |
| Collapsed genre | 3 | Strongest similarity signal |
| Spaced genre words | 1 | So a sentence like “crime drama” can match |
| Cast, first 3 names | 2 | Empty in this extract |
| Director | 2 | This is what makes The Irishman recommend other Scorsese films |
| Country | 0 | The Pakistan labels are wrong often enough that the token was turned off |
| Type, rating | 1 | |
| Description | 1 | Empty in this extract |

That text is TF-IDF (English stop words, unigrams and bigrams, `min_df=2`, sublinear tf, float32), then L2-normalised. Cosine similarity is one sparse dot product at request time. The full 8790×8790 dense matrix is not built and not stored.

A count vectoriser exists behind a config flag and is compared in the evaluation. It is not what the API serves.

Artifacts in `models/` (2,062,075 bytes):

- `tfidf_vectorizer.joblib`
- `item_matrix.npz` (4,256 features, 136,936 stored values)
- `title_index.parquet`
- `nn_index.joblib` (saved, not used on the request path)
- `metadata.json` (training time, git SHA `3a7b256`, CSV SHA-256, row count, library versions, metrics)

Training on this machine took about 16 seconds.

### 5. Evaluate, without calling it accuracy

`scripts/evaluate.py` wrote `reports/evaluation.md` and the charts in `reports/figures/`. There are no watch labels, so these are proxies.

| Metric | Value |
| --- | --- |
| Genre precision@10 | 0.9973 (95% CI 0.9967–0.9979) |
| Random baseline precision@10 | 0.2422 (95% CI 0.2383–0.2461) |
| Jaccard@10 | 0.9580 |
| Catalogue coverage of top-10 lists | 0.8729 |
| Gini of how often titles are recommended | 0.5764 |
| Intra-list diversity (higher means a more varied list) | 0.0934 |
| `recommend()` p50 / p95 | 10.24 ms / 14.92 ms |
| Cold load of the artifacts | 899.3 ms |

Ablation (genre precision@10):

| Configuration | Precision@10 | Note |
| --- | --- | --- |
| Description only | n/a | No description column, so the vectoriser had an empty vocabulary |
| Genres only | 1.0000 | |
| Full soup (what is shipped) | 0.9973 | |
| Genre weight 1 | 0.9875 | |
| Genre weight 6 | 0.9997 | |
| Country weight 1 | 0.9973 | |
| Count vectoriser, full soup | 1.0000 | Ties whole genre buckets at similarity 1.0 |

Genre precision is high because genre is both the feature and the label. The intra-list number is the more honest one: a top 10 is often ten near-copies.

Eight titles were read by hand. The useful case is The Irishman (other Scorsese films, and the reason line names Martin Scorsese). The weak cases are Breaking Bad (any other Crime / TV Drama / TV Thriller ties at 1.0; Better Call Saul does not reach the top 5) and Kota Factory (the file tags it as an international romantic comedy, so the neighbours are romantic comedies). Verdicts are in `reports/qualitative_verdicts.json`.

### 6. API and UI

FastAPI in `src/api/` loads the artifacts once at startup.

| Method | Path |
| --- | --- |
| GET | `/health`, `/ready`, `/meta`, `/titles`, `/titles/{show_id}`, `/stats`, `/metrics` |
| POST | `/recommend`, `/recommend/text` |

Checked against the running server: `breakng bad` resolved to Breaking Bad; `zzzznotitle` returned 404 with suggestions; `n: 0` returned 422; a nonsense sentence returned 400. Example bodies are in `docs/API.md`.

Streamlit (`app/streamlit_app.py`) has four tabs and calls the API over HTTP. It does not import the model. The tabs were not clicked through in a browser. `docs/screenshots/` is empty for that reason.

### 7. Tests, lint, Docker

- `pytest`: 44 passed, 88.77% line coverage on `src/` (the gate is 80%). Tests use `tests/fixtures/mini_netflix.csv`, not the full catalogue.
- `ruff check`, `black --check`, and `mypy src/` pass.
- `.github/workflows/ci.yml` is written. It has not been run on GitHub, because this work has not been pushed.
- `docker compose up --build` was run locally. The image is 944 MB. `/health` returned ok. Recommending The Irishman returned Gangs of New York. Streamlit answered HTTP 200. The stack was then stopped. The image’s own `metadata.json` says `git_commit: unknown` because `.git` is not copied into the build.

### 8. Docs already in the repo

| File | Role |
| --- | --- |
| `README.md` | How to run it, the metric table, limitations |
| `docs/DECISIONS.md` | Why each non-obvious choice was made |
| `docs/API.md` | Curl examples from the local server |
| `docs/DEMO_SCRIPT.md` | A short spoken walkthrough |
| `reports/data_profile.md` | Measured shape of the CSV |
| `reports/evaluation.md` | Metrics, ablation, qualitative lists |
| `notebooks/01_exploration.ipynb` | EDA that imports `src/`, executed with output saved |
| `deploy/` | Hugging Face, Render, and Fly instructions. Not applied. |

## What is not done

1. **No public URL.** Hugging Face, Render, and Fly were not used. `deploy/README.md` is the procedure. After a Space or a Render service is up, hit `/health`, `/meta`, and a real `/recommend`, and paste those responses into the README. Do not invent them.
2. **No commit and no push.** The working tree is ahead of the single commit `3a7b256` (“Initial commit”) only as uncommitted files. CI cannot go green until this is committed and pushed to `main`.
3. **No screenshots.** The brief wants numbered shots of each UI tab, `/docs`, a green CI run, the coverage report, and the deployed site with the URL bar visible.
4. **The UI was not exercised in a browser.** HTTP checks and unit tests cover the API. They do not prove the Streamlit layout, the franchise toggle, or the charts.
5. **A clean-machine rehearsal of `git clone` → `make setup` → `make train` was not done.** Train, test, and Docker were run in this checkout.

## Overfitting, underfitting, accuracy, correctness

Checked again on 27 September 2026 with `scripts/reliability.py`. The write-up is `reports/reliability.md`.

Accuracy was not computed and still cannot be. There are no watches or ratings. Reporting one would be false.

There is no train/test loss, so there is no classical overfitting or underfitting curve. The stand-in for overfitting is saturation: on a 400-title sample (seed 42), **51.55%** of top-10 neighbours have cosine at least 0.99, and only **4.50%** share a director or actor. The model has collapsed onto genre tokens. That is also why intra-list diversity is 0.0934. It is not fixed by more training epochs. TF-IDF here is one fit, not a loop that can be stopped early.

Underfitting shows up where the catalogue has no text. A mood sentence such as "slow burn melancholy" is refused. Cast is empty, so people cannot separate two titles that share a genre.

Correctness changes made after that measurement:

- A genre-only neighbour now says "Genre overlap only" and, when the score is at least 0.99, that the score is saturated. It does not say a viewer would like the title.
- A free-text query that is actually a catalogue title (`breaking bad`) uses title search instead of the empty description path.
- `drama` is expanded to the catalogue token `dramas`. `comedy` is expanded to `comedies`, because the stand-up token `comedy` was outranking children-and-family titles. `Martin Scorsese` is joined into `martinscorsese`, which is how the director is stored.
- Empty text, stop words, and strings with no catalogue tokens are still refused.

No-history positions after those changes:

| Query | Result |
| --- | --- |
| empty, stop words, `qqqq zzzz yyyy`, `slow burn melancholy` | refused |
| `crime drama thriller` | The Lincoln Lawyer, 0.768, Dramas and Thrillers |
| `kids family comedy` | Shaun the Sheep Movie: Farmageddon, 0.236, Children & Family and Comedies. Still a weak score. |
| `stand-up comedy` | Lokillo: Nothing's the Same, 0.589 |
| `Martin Scorsese` | The Irishman, 0.590 |
| `drama` | The Life of David Gale, 0.435 |
| `breaking bad` typed as taste | neighbours of Breaking Bad, with the saturated-genre warning |
| `breakng bad` as a title | resolves to Breaking Bad |
| `zzzznotitle` | 404 with suggestions |

Pytest after these changes: 88.85% coverage on `src/`, still above 80%.

## What to do next, in order

### A. Finish the submission the brief actually grades

1. Commit the project (source, tests, `data/raw/netflix_titles.csv`, reports, docs). Do not commit `.venv`, `__pycache__`, or `.env`. `models/` is gitignored; the Docker build retrains, and `make train` rebuilds it locally.
2. Push to `main` and wait until `.github/workflows/ci.yml` is green. Only then add the status badge to the README. A red or missing badge is worse than none.
3. Deploy with the steps in `deploy/README.md`. Hugging Face Spaces (Docker) is the path that does not need a card. Set `PORT` to the port the host injects, and set the UI’s `API_BASE_URL` to the API’s public URL. Put that URL in `CORS_ORIGINS`.
4. From outside your laptop, call `/health`, `/meta`, and `POST /recommend` with `{"title":"Breaking Bad","n":5}`. Paste the real JSON into the README.
5. Run `python scripts/serve.py`, open the four tabs, and save screenshots into `docs/screenshots/` as `01` onward. Include `/docs`, the green Actions run, and the pytest coverage line.
6. Click the UI: search `breakng bad`, confirm the “resolved to Breaking Bad” notice and a “Because…” line on every card, toggle sequels on Naruto Shippuden: The Movie, and run the “crime drama thriller” example.

### B. Make the recommendations less of a genre tie

The model is doing what it was asked. The data cannot support a better content match until the missing fields exist.

1. Add `cast` and `description` from a catalogue extract that actually has them, then re-run `make train` and `make evaluate`. The director effect on The Irishman is the preview of what cast would do. Descriptions would give the cold-start box something real to match.
2. Keep country weight at 0 until the Pakistan labels are corrected. Do not turn the weight up to chase genre precision; the ablation already shows country weight 1 does not help that number, and it would glue unrelated titles together.
3. If a synopsis is added, try a sentence-transformer embedding as a second index and compare it in `scripts/evaluate.py` the same way TF-IDF and the count vectoriser are compared. Do not replace the current model until that table exists.
4. Collaborative filtering, a hybrid re-ranker, and an A/B test need watch or rating events. This file does not have them. Do not report an accuracy until that data exists.

### C. Small engineering follow-ups

- Re-run `scripts/train.py` after the next commit so `metadata.json` records the new git SHA instead of `3a7b256`.
- The latency number (p95 14.92 ms) was measured with `show_id`, because a few display titles exist twice (`Esperando la carroza`, and the restored `Feb-09` / `15-Aug` / `22-Jul` pairs). A typo such as `breakng bad` took about 90 ms on one live call, still under the 200 ms budget. If you quote latency, say which kind of query.
- `docs/screenshots/README.md` explains the empty folder. Replace that note with the real images once they exist. Delete the note rather than leaving both.
