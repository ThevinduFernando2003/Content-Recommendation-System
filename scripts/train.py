"""Fit the recommender from the raw CSV and write ``models/``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import load_config  # noqa: E402
from src.data.clean import clean_frame  # noqa: E402
from src.data.load import load_raw_csv  # noqa: E402
from src.logging_conf import configure_logging  # noqa: E402
from src.models.persistence import save_recommender  # noqa: E402
from src.models.recommender import Recommender  # noqa: E402


def main() -> None:
    configure_logging()
    config = load_config()
    np.random.seed(config.random_seed)
    raw = load_raw_csv(config.raw_csv)
    cleaned = clean_frame(raw)
    config.paths.processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(config.paths.processed_path, index=False)
    model = Recommender(config).fit(cleaned)
    save_recommender(model, config.models_dir, source_csv=config.raw_csv, metrics=None)
    print(f"Trained on {len(cleaned.index)} titles.")
    print(f"Artifacts: {config.models_dir}")
    print(f"Processed: {config.paths.processed_path}")


if __name__ == "__main__":
    main()
