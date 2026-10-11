"""Bounded retrieve → generate → verify loop, including deterministic evidence gates."""

import asyncio
import operator
import re
from time import perf_counter
from typing import Annotated, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from doculens.config import Settings
from doculens.pipelines.retrieval import budget_context, reciprocal_rank_fusion
from doculens.schemas import Answer, Chunk, Draft, QueryRequest, TraceEvent, Verification


class Retriever(Protocol):
    async def retrieve(self, query: str) -> list[Chunk]: ...


class Agents(Protocol):
    async def generate(self, question: str, chunks: list[Chunk], feedback: str) -> Draft: ...

    async def verify(self, question: str, chunks: list[Chunk], draft: Draft) -> Verification: ...


class RagState(TypedDict, total=False):
    question: str
    retrieval_query: str
    chunks: list[Chunk]
    draft: Draft
    verification: Verification
    attempts: int
    feedback: str
    accepted: bool
    answer: Answer
    trace: Annotated[list[TraceEvent], operator.add]


class RunResult(BaseModel):
    answer: Answer
    retrieved_chunks: list[Chunk]


def evidence_errors(draft: Draft, report: Verification, chunks: list[Chunk]) -> list[str]:
    """A judge cannot override invalid citation IDs or unsupported claim flags."""
    available = {chunk.id for chunk in chunks}
    listed = set(draft.citations)
    inline = set(re.findall(r"\[([^\[\]]+)\]", draft.answer))
    errors = []
    if not draft.answer.strip():
        errors.append("Draft answer is empty")
    if not listed or listed != inline or not listed <= available:
        errors.append("Use matching inline/listed citations from the retrieved chunks")
    if not report.claims:
        errors.append("Verifier must enumerate factual claims")
    for claim in report.claims:
        if not claim.supported or not claim.evidence_ids or not set(claim.evidence_ids) <= listed:
            errors.append(f"Unsupported or incorrectly cited claim: {claim.claim}")
    if not report.grounded or not report.answers_question:
        errors.append("Verifier rejected grounding or question coverage")
    return errors


class RagPipeline:
    def __init__(self, retriever: Retriever, agents: Agents, settings: Settings) -> None:
        self.retriever, self.agents, self.settings = retriever, agents, settings
        builder = StateGraph(RagState)
        builder.add_node("retrieve", self.retrieve)
        builder.add_node("generate", self.generate)
        builder.add_node("verify", self.verify)
        builder.add_node("finalize", self.finalize)
        builder.add_node("abstain", self.abstain)
        builder.add_edge(START, "retrieve")
        builder.add_conditional_edges("retrieve", self.after_retrieve)
        builder.add_conditional_edges("generate", self.after_generate)
        builder.add_conditional_edges("verify", self.after_verify)
        builder.add_edge("finalize", END)
        builder.add_edge("abstain", END)
        self.graph = builder.compile()

    def event(self, state: RagState, node: str, started: float, detail: str) -> list[TraceEvent]:
        return [
            TraceEvent(
                node=node,
                attempt=state["attempts"],
                duration_ms=round((perf_counter() - started) * 1000, 2),
                detail=detail,
            )
        ]

    async def retrieve(self, state: RagState) -> dict:
        started = perf_counter()
        original = await self.retriever.retrieve(state["question"])
        rankings = [original]
        if state["retrieval_query"] != state["question"]:
            rankings.append(await self.retriever.retrieve(state["retrieval_query"]))
        chunks = budget_context(
            reciprocal_rank_fusion(rankings), self.settings.top_k, self.settings.max_context_chars
        )
        return {
            "chunks": chunks,
            "trace": self.event(state, "retrieve", started, f"Retrieved {len(chunks)} chunks"),
        }

    def after_retrieve(self, state: RagState) -> Literal["generate", "abstain"]:
        return "generate" if state["chunks"] else "abstain"

    async def generate(self, state: RagState) -> dict:
        started = perf_counter()
        draft = await self.agents.generate(state["question"], state["chunks"], state["feedback"])
        updated = {**state, "attempts": state["attempts"] + 1}
        return {
            "draft": draft,
            "attempts": updated["attempts"],
            "trace": self.event(updated, "generate", started, "Draft generated"),
        }

    def after_generate(self, state: RagState) -> Literal["verify", "abstain"]:
        return "abstain" if state["draft"].insufficient_context else "verify"

    async def verify(self, state: RagState) -> dict:
        started = perf_counter()
        report = await self.agents.verify(state["question"], state["chunks"], state["draft"])
        errors = evidence_errors(state["draft"], report, state["chunks"])
        if report.hallucination_risk > self.settings.max_hallucination_risk:
            errors.append("Hallucination risk exceeds the configured threshold")
        feedback = "\n".join([report.feedback, *errors])
        query = report.revised_query.strip()[:2000] or state["question"]
        return {
            "verification": report,
            "accepted": not errors,
            "feedback": feedback,
            "retrieval_query": query,
            "trace": self.event(state, "verify", started, "Passed" if not errors else "Rejected"),
        }

    def after_verify(self, state: RagState) -> Literal["finalize", "retrieve", "abstain"]:
        if state["accepted"]:
            return "finalize"
        if state["attempts"] <= self.settings.max_revisions:
            return "retrieve"
        return "abstain"

    def finalize(self, state: RagState) -> dict:
        cited = set(state["draft"].citations)
        return {
            "answer": Answer(
                question=state["question"],
                answer=state["draft"].answer,
                status="verified",
                sources=[chunk for chunk in state["chunks"] if chunk.id in cited],
                attempts=state["attempts"],
                verification=state["verification"],
                trace=state["trace"],
            )
        }

    def abstain(self, state: RagState) -> dict:
        return {
            "answer": Answer(
                question=state["question"],
                answer="I could not verify an answer from the available documents. "
                "Try a more specific question or add relevant documents.",
                status="abstained",
                sources=[],
                attempts=state["attempts"],
                verification=state.get("verification"),
                trace=state["trace"],
            )
        }

    async def run(self, question: str) -> RunResult:
        question = QueryRequest(question=question).question
        # End-to-end deadline includes retrieval, all model retries, and graph revisions.
        async with asyncio.timeout(self.settings.request_timeout_seconds):
            state = await self.graph.ainvoke(
                {
                    "question": question,
                    "retrieval_query": question,
                    "attempts": 0,
                    "feedback": "",
                    "trace": [],
                },
                config={"recursion_limit": 3 * (self.settings.max_revisions + 1) + 5},
            )
        return RunResult(answer=state["answer"], retrieved_chunks=state["chunks"])
