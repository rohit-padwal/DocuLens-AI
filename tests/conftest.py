import pytest

from doculens.config import Settings
from doculens.schemas import Chunk, ClaimCheck, Draft, Verification


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, persist_directory=tmp_path / "chroma")


@pytest.fixture
def chunk():
    return Chunk(
        id="chunk_0123456789abcdef", text="Audit logs are retained for 90 days.", source="policy.md"
    )


@pytest.fixture
def draft(chunk):
    return Draft(
        answer=f"Audit logs are retained for 90 days. [{chunk.id}]",
        citations=[chunk.id],
        insufficient_context=False,
    )


@pytest.fixture
def verification(chunk):
    return Verification(
        grounded=True,
        answers_question=True,
        hallucination_risk=0.05,
        claims=[ClaimCheck(claim="90-day retention", supported=True, evidence_ids=[chunk.id])],
        feedback="Supported",
        revised_query="audit log retention",
    )
