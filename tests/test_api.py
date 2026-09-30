"""HTTP contract: shapes, 404 suggestions, validation, and adversarial input."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.schemas import RecommendResponse, TitleDetail
from src.models.recommender import Recommender


def _client(recommender: Recommender) -> TestClient:
    return TestClient(create_app(recommender))


def test_health_ready_and_meta(recommender: Recommender) -> None:
    with _client(recommender) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").status_code == 200
        meta = client.get("/meta")
        assert meta.status_code == 200
        assert meta.json()["row_count"] == len(recommender.titles)
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "http_requests_total" in metrics.text


def test_ready_is_503_without_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "absent"))
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
        ready = client.get("/ready")
        assert ready.status_code == 503


def test_recommend_shape_and_fuzzy(recommender: Recommender) -> None:
    with _client(recommender) as client:
        response = client.post(
            "/recommend",
            json={"title": "city of shadws", "n": 3, "exclude_same_franchise": False},
        )
        assert response.status_code == 200
        assert response.headers.get("X-Request-ID")
        body = RecommendResponse.model_validate(response.json())
        assert body.query_title == "City of Shadows"
        assert body.resolved_from == "city of shadws"
        assert 1 <= len(body.results) <= 3
        assert body.results[0].reason


def test_rating_filter_is_exact(recommender: Recommender) -> None:
    with _client(recommender) as client:
        response = client.post(
            "/recommend",
            json={
                "title": "Panda Legend",
                "n": 10,
                "rating": "PG",
                "exclude_same_franchise": False,
            },
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert results
        assert all(item["rating"] == "PG" for item in results)
        assert all(item["description"] for item in results)


def test_unknown_title_is_404_with_suggestions(recommender: Recommender) -> None:
    with _client(recommender) as client:
        response = client.post("/recommend", json={"title": "zzzznotitle"})
        assert response.status_code == 404
        payload = response.json()
        assert "suggestions" in payload
        assert payload["suggestions"]


def test_validation_error_names_the_field(recommender: Recommender) -> None:
    with _client(recommender) as client:
        response = client.post("/recommend", json={"title": "Glass Alibi", "n": 0})
        assert response.status_code == 422
        assert "n" in response.text


def test_adversarial_inputs_do_not_500(recommender: Recommender) -> None:
    with _client(recommender) as client:
        empty = client.post("/recommend", json={"title": ""})
        assert empty.status_code == 422
        huge = client.post("/recommend", json={"title": "a" * 10_000})
        assert huge.status_code == 422
        for title in ["🎬🎬🎬", "'; DROP TABLE titles;--", "../../etc/passwd"]:
            response = client.post("/recommend", json={"title": title, "n": 3})
            assert response.status_code in {200, 400, 404}
            assert response.status_code != 500


def test_text_cold_start_and_titles(recommender: Recommender) -> None:
    with _client(recommender) as client:
        found = client.post("/recommend/text", json={"query": "crime drama", "n": 3})
        assert found.status_code == 200
        assert found.json()["results"]
        missing = client.post("/recommend/text", json={"query": "qqqq zzzz yyyy"})
        assert missing.status_code == 400
        search = client.get("/titles", params={"q": "panda", "limit": 5})
        assert search.status_code == 200
        assert any("Panda" in item["title"] for item in search.json())
        detail = client.get("/titles/s15")
        assert detail.status_code == 200
        TitleDetail.model_validate(detail.json())
        absent = client.get("/titles/s-does-not-exist")
        assert absent.status_code == 404
        stats = client.get("/stats")
        assert stats.status_code == 200
        assert stats.json()["n_titles"] == len(recommender.titles)
