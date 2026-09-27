"""FastAPI application. Artifacts are loaded once in the lifespan, not per request."""

from __future__ import annotations

import inspect
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, get_type_hints

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.deps import get_recommender
from src.api.schemas import (
    MetaResponse,
    RecommendationModel,
    RecommendRequest,
    RecommendResponse,
    StatsResponse,
    TextRecommendRequest,
    TextRecommendResponse,
    TitleBrief,
    TitleDetail,
)
from src.config import load_config
from src.logging_conf import configure_logging, request_id_var
from src.models.persistence import load_metadata, load_recommender
from src.models.recommender import (
    EmptyQueryError,
    Recommendation,
    Recommender,
    TitleNotFoundError,
)

logger = logging.getLogger(__name__)

REQUESTS = Counter(
    "http_requests_total",
    "HTTP requests",
    labelnames=("method", "path", "status"),
)
LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    labelnames=("path",),
)
ERRORS = Counter(
    "http_errors_total",
    "HTTP responses with status >= 500",
    labelnames=("status",),
)

RATE_LIMIT = os.environ.get("RATE_LIMIT", "60/minute")
limiter = Limiter(key_func=get_remote_address)


def _cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _limited(func):  # type: ignore[no-untyped-def]
    """Apply slowapi without hiding the endpoint signature from FastAPI.

    slowapi's wrapper is defined in another module. With postponed annotations,
    FastAPI evaluates those strings in the wrapper's globals and then treats the
    body model as a missing query parameter. Resolve the annotations here.
    """
    wrapped = limiter.limit(RATE_LIMIT)(func)
    hints = get_type_hints(func)
    signature = inspect.signature(func)
    parameters = [
        param.replace(annotation=hints.get(param.name, param.annotation))
        for param in signature.parameters.values()
    ]
    wrapped.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters,
        return_annotation=hints.get("return", signature.return_annotation),
    )
    return wrapped


def _to_model(item: Recommendation) -> RecommendationModel:
    return RecommendationModel(
        show_id=item.show_id,
        title=item.title,
        type=item.type,
        release_year=item.release_year,
        rating=item.rating,
        genres=item.genres,
        similarity=item.similarity,
        reason=item.reason,
        shared_genres=item.shared_genres,
        shared_people=item.shared_people,
    )


def create_app(recommender: Recommender | None = None) -> FastAPI:
    """Build the app. Tests pass a fitted recommender; production loads artifacts."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        if recommender is not None:
            app.state.recommender = recommender
            app.state.metadata = {
                "trained_at": None,
                "git_commit": None,
                "data_sha256": None,
                "row_count": int(len(recommender.show_ids)),
                "library_versions": {},
                "metrics": None,
                "random_seed": recommender.config.random_seed,
                "model_version": recommender.model_version,
            }
            logger.info("Using injected recommender %s", recommender.model_version)
        else:
            config = load_config()
            try:
                app.state.recommender = load_recommender(config.models_dir, config)
                app.state.metadata = load_metadata(config.models_dir)
                app.state.metadata["model_version"] = app.state.recommender.model_version
                logger.info("Startup metadata: %s", app.state.metadata)
            except FileNotFoundError:
                logger.exception("Model artifacts are missing; /ready will stay 503")
                app.state.recommender = None
                app.state.metadata = {}
        yield

    app = FastAPI(
        title="Netflix content recommender",
        version="1.0.0",
        summary="Content-based recommendations over a Netflix catalogue.",
        description=(
            "Similarity is cosine over an L2-normalised TF-IDF matrix of genres, "
            "people, and catalogue metadata. There is no collaborative signal and "
            "no accuracy claim."
        ),
        lifespan=lifespan,
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled error before a response was formed")
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal error", "request_id": request_id},
            )
        duration_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        status = str(response.status_code)
        REQUESTS.labels(request.method, path, status).inc()
        LATENCY.labels(path).observe(duration_ms / 1000)
        if response.status_code >= 500:
            ERRORS.labels(status).inc()
        logger.info("%s %s %s %.2f", request.method, request.url.path, status, duration_ms)
        request_id_var.reset(token)
        return response

    @app.exception_handler(TitleNotFoundError)
    async def title_not_found(request: Request, exc: TitleNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"detail": str(exc), "suggestions": exc.suggestions},
        )

    @app.exception_handler(EmptyQueryError)
    async def empty_query(request: Request, exc: EmptyQueryError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> Response:
        if isinstance(exc, HTTPException):
            return await http_exception_handler(request, exc)
        if isinstance(exc, RequestValidationError):
            return await request_validation_exception_handler(request, exc)
        if isinstance(exc, TitleNotFoundError):
            return JSONResponse(
                status_code=404,
                content={"detail": str(exc), "suggestions": exc.suggestions},
            )
        request_id = request_id_var.get("-")
        logger.exception("Unhandled error")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal error", "request_id": request_id},
        )

    @app.get("/health", summary="Liveness", description="Process is up. Does not check the model.")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(
        "/ready",
        summary="Readiness",
        description="200 once artifacts are loaded, otherwise 503.",
    )
    def ready(request: Request) -> dict[str, str]:
        if getattr(request.app.state, "recommender", None) is None:
            raise HTTPException(status_code=503, detail="Model artifacts are not loaded")
        return {"status": "ready"}

    @app.get("/meta", response_model=MetaResponse, summary="Model metadata")
    def meta(request: Request) -> dict[str, Any]:
        payload = dict(getattr(request.app.state, "metadata", {}) or {})
        model = getattr(request.app.state, "recommender", None)
        if model is not None:
            payload["model_version"] = model.model_version
        return payload

    @app.get("/titles", response_model=list[TitleBrief], summary="Typeahead search")
    def titles(
        q: str = Query(default="", max_length=300),
        limit: int = Query(default=20, ge=1, le=50),
        model: Recommender = Depends(get_recommender),
    ) -> list[dict[str, Any]]:
        return model.search_titles(q, limit)

    @app.get("/titles/{show_id}", response_model=TitleDetail, summary="One catalogue row")
    def title_detail(show_id: str, model: Recommender = Depends(get_recommender)) -> dict[str, Any]:
        found = model.get_title(show_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"No title with show_id '{show_id}'.")
        return found

    @app.post(
        "/recommend",
        response_model=RecommendResponse,
        summary="Recommend similar titles",
        description=(
            "Fuzzy, case-insensitive title lookup. Unknown titles return 404 with "
            "suggestions. Filters run after similarity is computed."
        ),
    )
    @_limited
    def recommend(
        request: Request,
        body: RecommendRequest,
        model: Recommender = Depends(get_recommender),
    ) -> RecommendResponse:
        started = time.perf_counter()
        resolved, results = model.recommend(
            body.title,
            n=body.n,
            content_type=body.content_type,
            min_release_year=body.min_release_year,
            exclude_same_franchise=body.exclude_same_franchise,
        )
        return RecommendResponse(
            query_title=resolved.title,
            resolved_from=resolved.resolved_from,
            results=[_to_model(item) for item in results],
            took_ms=round((time.perf_counter() - started) * 1000, 3),
            model_version=model.model_version,
        )

    @app.post(
        "/recommend/text",
        response_model=TextRecommendResponse,
        summary="Recommend from a free-text description",
        description="Cold start. The fitted vectoriser transforms the text. An out-of-vocabulary query returns 400.",
    )
    @_limited
    def recommend_text(
        request: Request,
        body: TextRecommendRequest,
        model: Recommender = Depends(get_recommender),
    ) -> TextRecommendResponse:
        started = time.perf_counter()
        results = model.recommend_from_text(body.query, n=body.n)
        return TextRecommendResponse(
            query=body.query,
            results=[_to_model(item) for item in results],
            took_ms=round((time.perf_counter() - started) * 1000, 3),
            model_version=model.model_version,
        )

    @app.get("/stats", response_model=StatsResponse, summary="Catalogue summary")
    def stats(model: Recommender = Depends(get_recommender)) -> dict[str, Any]:
        return model.catalogue_stats()

    @app.get(
        "/metrics",
        summary="Prometheus metrics",
        description="Request counters and a latency histogram.",
    )
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def _rate_limit_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = request_id_var.get("-")
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Try again in a minute.", "request_id": request_id},
    )


app = create_app()
