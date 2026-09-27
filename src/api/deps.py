"""Request-scoped access to the process-wide recommender."""

from __future__ import annotations

from fastapi import HTTPException, Request

from src.models.recommender import Recommender


def get_recommender(request: Request) -> Recommender:
    """Return the model loaded at startup. 503 until that has happened."""
    model = getattr(request.app.state, "recommender", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded")
    return model
