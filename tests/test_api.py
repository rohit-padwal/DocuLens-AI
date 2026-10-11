import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from doculens.agents.graph import RagPipeline
from doculens.api import create_app
from tests.test_graph import FakeAgents, FakeRetriever


def test_api_auth_validation_and_verified_response(settings, chunk, draft, verification):
    settings.api_key = SecretStr("test-secret")
    pipeline = RagPipeline(FakeRetriever([chunk]), FakeAgents(draft, [verification]), settings)
    with TestClient(create_app(settings, lambda _: pipeline)) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200
        assert client.post("/query", json={"question": "retention?"}).status_code == 401
        invalid = client.post(
            "/query", json={"question": "   "}, headers={"X-API-Key": "test-secret"}
        )
        assert invalid.status_code == 422
        response = client.post(
            "/query", json={"question": "retention?"}, headers={"X-API-Key": "test-secret"}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "verified"
        assert response.headers["X-Request-ID"]


@pytest.mark.parametrize(
    "error,status", [(TimeoutError("secret"), 504), (RuntimeError("secret"), 502)]
)
def test_api_errors_hide_provider_details(settings, error, status):
    class BrokenPipeline:
        async def run(self, question):
            raise error

    with TestClient(create_app(settings, lambda _: BrokenPipeline())) as client:
        response = client.post("/query", json={"question": "retention?"})
        assert response.status_code == status
        assert "secret" not in response.text
        assert client.app.state.active_requests == 0


def test_capacity_rejected_before_pipeline_call(settings):
    with TestClient(create_app(settings, lambda _: None)) as client:
        client.app.state.active_requests = settings.max_concurrent_requests
        response = client.post("/query", json={"question": "retention?"})
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "1"


@pytest.mark.parametrize("headers", [{}, {"Content-Length": "1"}])
def test_query_payload_limit_counts_actual_body_bytes(settings, headers):
    with TestClient(create_app(settings, lambda _: None)) as client:
        response = client.post(
            "/query", content=b"x" * (settings.max_request_bytes + 1), headers=headers
        )
        assert response.status_code == 413
        assert response.headers["X-Request-ID"]
        assert client.app.state.active_requests == 0


def test_declared_oversized_payload_rejected_even_if_body_small(settings):
    with TestClient(create_app(settings, lambda _: None)) as client:
        response = client.post(
            "/query", content=b"{}", headers={"Content-Length": str(settings.max_request_bytes + 1)}
        )
        assert response.status_code == 413


def test_query_limit_works_for_chunked_requests_and_stops_reading_at_limit():
    import asyncio

    from doculens.middleware import QueryBodyLimit

    async def exercise():
        calls = 0
        sent = []

        async def unreachable(*args):
            raise AssertionError("Oversized request reached the application")

        async def receive():
            nonlocal calls
            calls += 1
            return {"type": "http.request", "body": b"123456", "more_body": True}

        async def send(message):
            sent.append(message)

        await QueryBodyLimit(unreachable, max_bytes=10)(
            {"type": "http", "path": "/query", "method": "POST", "headers": []}, receive, send
        )
        assert calls == 2
        assert sent[0]["status"] == 413

    asyncio.run(exercise())
