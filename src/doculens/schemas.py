"""Typed contracts shared by the graph, providers, API, and evaluator."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Chunk(BaseModel):
    id: str
    text: str
    source: str
    page: int | None = None


class Draft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(description="Answer with inline [chunk_id] citations for factual claims")
    citations: list[str] = Field(description="Exactly the chunk IDs cited inline in the answer")
    insufficient_context: bool


class ClaimCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str
    supported: bool
    evidence_ids: list[str]


class Verification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    grounded: bool
    answers_question: bool
    hallucination_risk: float = Field(ge=0, le=1)
    claims: list[ClaimCheck]
    feedback: str
    revised_query: str = Field(description="Focused retrieval query to fill missing evidence")


class TraceEvent(BaseModel):
    node: str
    attempt: int
    duration_ms: float
    detail: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must contain non-whitespace characters")
        return value.strip()


class Answer(BaseModel):
    question: str
    answer: str
    status: Literal["verified", "abstained"]
    sources: list[Chunk]
    attempts: int
    verification: Verification | None
    trace: list[TraceEvent]
