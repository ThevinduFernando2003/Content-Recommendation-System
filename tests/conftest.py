"""Shared fixtures. The suite never reads the full catalogue."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("RATE_LIMIT", "10000/minute")

from src.config import load_config
from src.data.clean import clean_frame
from src.data.load import load_raw_csv
from src.models.recommender import Recommender

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "mini_netflix.csv"


@pytest.fixture(scope="session")
def cleaned_df():
    return clean_frame(load_raw_csv(FIXTURE))


@pytest.fixture(scope="session")
def recommender(cleaned_df):
    model = Recommender(load_config()).fit(cleaned_df)
    model.model_version = "test"
    return model
