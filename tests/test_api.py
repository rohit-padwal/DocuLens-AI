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
