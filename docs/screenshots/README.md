# Screenshots

This folder is empty on purpose.

The UI, the OpenAPI page, a browser pointed at a public URL, a green GitHub Actions run, and a coverage report were not captured. This session had no browser tool, the services were not deployed, and the workflow has not been pushed to GitHub. Inventing those images would violate the rule that every figure in the docs comes from something that was actually run.

What was run instead:

- `pytest` on 27 September 2026: 44 passed, 88.77% line coverage on `src/`.
- The API on `http://127.0.0.1:8000`. The bodies are in `docs/API.md`.
- `scripts/evaluate.py`, which wrote `reports/evaluation.md` and `reports/figures/*.png`.

To add the missing shots later: `python scripts/serve.py`, then save the four Streamlit tabs, `http://127.0.0.1:8000/docs`, the Actions run, and the pytest coverage summary into this folder as `01` through `07`.
