"""Artifacts must load in a fresh interpreter, not only in the process that saved them."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import joblib

from src.config import load_config
from src.models.persistence import save_recommender
from src.models.recommender import Recommender

ROOT = Path(__file__).resolve().parents[1]


def test_round_trip_in_a_subprocess(recommender: Recommender, tmp_path: Path) -> None:
    source = ROOT / "tests" / "fixtures" / "mini_netflix.csv"
    save_recommender(
        recommender, tmp_path, source_csv=source, metrics={"genre_precision_at_10": 1.0}
    )
    expected = [
        item.title
        for item in recommender.recommend("City of Shadows", n=5, exclude_same_franchise=False)[1]
    ]
    code = """
import json
import os
from pathlib import Path
from src.config import load_config
from src.models.persistence import load_recommender
directory = Path(os.environ["MODEL_DIR"])
model = load_recommender(directory, load_config())
titles = [item.title for item in model.recommend("City of Shadows", n=5, exclude_same_franchise=False)[1]]
print(json.dumps(titles))
"""
    env = os.environ.copy()
    env["MODEL_DIR"] = str(tmp_path)
    env["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == expected
    neighbours = joblib.load(tmp_path / "nn_index.joblib")
    assert neighbours.n_neighbors >= 1
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["row_count"] == len(recommender.titles)
    assert metadata["data_sha256"]
    assert metadata["random_seed"] == load_config().random_seed
