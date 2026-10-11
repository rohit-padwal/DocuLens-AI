"""Run the actual graph on held-out cases, then score the final response with Ragas."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import uuid
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from doculens.agents.models import GENERATOR_PROMPT, VERIFIER_PROMPT
from doculens.config import Settings
from doculens.service import build_pipeline


class TestCase(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=2000)
    reference: str
    expected_sources: list[str] = Field(default_factory=list)
    expected_status: Literal["verified", "abstained"] = "verified"

    @model_validator(mode="after")
    def validate_reference(self) -> "TestCase":
        if not self.question.strip():
            raise ValueError("Question must not be blank")
        if self.expected_status == "verified" and not self.reference.strip():
            raise ValueError("Answerable cases need a gold reference answer for context recall")
        return self


def load_cases(path: Path) -> list[TestCase]:
    cases = [
        TestCase.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("Dataset must contain cases with unique IDs")
    if not any(case.expected_status == "verified" for case in cases):
        raise ValueError("Dataset must include at least one answerable case")
    return cases


def finite_score(value: Any) -> float:
    score = float(value)
    if not math.isfinite(score):
        raise ValueError("Evaluator returned a non-finite score; failing the evaluation")
    return score


def summarize(rows: list[dict], thresholds: dict[str, float]) -> dict:
    answerable = [row for row in rows if row["expected_status"] == "verified"]
    metrics = {metric: mean(row["metrics"][metric] for row in answerable) for metric in thresholds}
    status_accuracy = mean(row["status"] == row["expected_status"] for row in rows)
    return {
        "cases": len(rows),
        "answerable_cases": len(answerable),
        "means": metrics,
        "status_accuracy": status_accuracy,
        "abstention_rate": mean(row["status"] == "abstained" for row in rows),
        "mean_attempts": mean(row["attempts"] for row in rows),
        "mean_latency_ms": mean(row["latency_ms"] for row in rows),
        "passed": all(metrics[name] >= threshold for name, threshold in thresholds.items())
        and status_accuracy == 1.0,
    }


class RagasJudge:
    """Use the collection-based API; construct clients only for explicit evaluation runs."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
        try:
            from openai import AsyncOpenAI
            from ragas.embeddings.base import embedding_factory
            from ragas.llms import llm_factory
            from ragas.metrics.collections import AnswerRelevancy, ContextRecall, Faithfulness
        except ImportError as exc:
            raise RuntimeError(
                'Install evaluation dependencies: pip install -e ".[evaluation]"'
            ) from exc
        self.client = (
            client
            if client is not None
            else AsyncOpenAI(
                api_key=settings.require_provider_key().get_secret_value(),
                timeout=settings.model_timeout_seconds,
                max_retries=2,
            )
        )
        llm = llm_factory(settings.evaluation_model, client=self.client)
        embeddings = embedding_factory("openai", model=settings.embedding_model, client=self.client)
        self.recall = ContextRecall(llm=llm)
        self.relevance = AnswerRelevancy(llm=llm, embeddings=embeddings)
        self.faithfulness = Faithfulness(llm=llm)

    async def score(
        self, question: str, response: str, contexts: list[str], reference: str
    ) -> dict:
        # Sequential judging bounds API pressure; every result is checked for NaN/inf.
        recall = await self.recall.ascore(
            user_input=question, retrieved_contexts=contexts, reference=reference
        )
        relevance = await self.relevance.ascore(user_input=question, response=response)
        faithful = await self.faithfulness.ascore(
            user_input=question, response=response, retrieved_contexts=contexts
        )
        return {
            "context_recall": finite_score(recall.value),
            "answer_relevancy": finite_score(relevance.value),
            "faithfulness": finite_score(faithful.value),
        }

    async def close(self) -> None:
        await self.client.close()


async def evaluate_cases(cases: list[TestCase], pipeline, judge, output: Path) -> list[dict]:
    """Always store the actual final prompt context, including for abstention cases."""
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output.open("w", encoding="utf-8") as report:
        for case in cases:
            started = perf_counter()
            run = await pipeline.run(case.question)
            latency = round((perf_counter() - started) * 1000, 2)
            contexts = [chunk.text for chunk in run.retrieved_chunks]
            if case.expected_status == "verified":
                if run.answer.status == "abstained":
                    # Do not reward abstention on answerable cases or silently omit it.
                    if contexts:
                        recall = await judge.recall.ascore(
                            user_input=case.question,
                            retrieved_contexts=contexts,
                            reference=case.reference,
                        )
                        recall_score = finite_score(recall.value)
                    else:
                        recall_score = 0.0
                    metrics = {
                        "context_recall": recall_score,
                        "answer_relevancy": 0.0,
                        "faithfulness": 0.0,
                    }
                else:
                    metrics = await judge.score(
                        case.question, run.answer.answer, contexts, case.reference
                    )
            else:
                # Faithfulness/recall are undefined for intentionally unanswerable cases.
                metrics = {}
            retrieved_sources = {chunk.source for chunk in run.retrieved_chunks}
            row = {
                "id": case.id,
                "question": case.question,
                "reference": case.reference,
                "expected_status": case.expected_status,
                "status": run.answer.status,
                "response": run.answer.answer,
                "retrieved_contexts": contexts,
                "retrieved_ids": [chunk.id for chunk in run.retrieved_chunks],
                "retrieved_sources": sorted(retrieved_sources),
                "source_recall": len(set(case.expected_sources) & retrieved_sources)
                / len(set(case.expected_sources))
                if case.expected_sources
                else None,
                "attempts": run.answer.attempts,
                "latency_ms": latency,
                "metrics": metrics,
                "trace": [event.model_dump() for event in run.answer.trace],
            }
            rows.append(row)
            report.write(json.dumps(row, allow_nan=False) + "\n")
            report.flush()
    return rows


async def run_evaluation(args: argparse.Namespace) -> bool:
    # Exclusive directory creation prevents a failed rerun from reusing old passing results.
    # Reserve it before constructing clients or spending any provider tokens.
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        return await complete_evaluation(args)
    except Exception as exc:
        write_summary(
            args.output / "summary.json",
            {
                "passed": False,
                "run_status": "failed",
                "error_type": type(exc).__name__,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        raise


def write_summary(path: Path, summary: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


async def complete_evaluation(args: argparse.Namespace) -> bool:
    settings = Settings()
    cases = load_cases(args.dataset)
    pipeline = build_pipeline(settings)
    judge = RagasJudge(settings)
    try:
        rows = await evaluate_cases(cases, pipeline, judge, args.output / "results.jsonl")
    finally:
        await judge.close()
    thresholds = {
        "context_recall": args.min_context_recall,
        "answer_relevancy": args.min_answer_relevancy,
        "faithfulness": args.min_faithfulness,
    }
    summary = summarize(rows, thresholds)
    summary["run_status"] = "completed"
    summary["thresholds"] = thresholds
    summary["metadata"] = {
        "created_at": datetime.now(UTC).isoformat(),
        "generation_model": settings.generation_model,
        "verifier_model": settings.verifier_model,
        "evaluation_model": settings.evaluation_model,
        "embedding_model": settings.embedding_model,
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(
            json.dumps(
                [chunk.model_dump() for chunk in pipeline.retriever.chunks], sort_keys=True
            ).encode()
        ).hexdigest(),
        "prompts_sha256": hashlib.sha256((GENERATOR_PROMPT + VERIFIER_PROMPT).encode()).hexdigest(),
        "versions": {name: version(name) for name in ("doculens-ai", "langgraph", "ragas")},
        "retrieval": {"top_k": settings.top_k, "candidate_k": settings.candidate_k},
        "max_revisions": settings.max_revisions,
        "max_hallucination_risk": settings.max_hallucination_risk,
    }
    write_summary(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary["passed"]


def threshold(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("Threshold must be a finite number between 0 and 1")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the agentic RAG graph with Ragas")
    parser.add_argument("--dataset", type=Path, default=Path("data/evaluation/cases.jsonl"))
    default_output = Path("reports") / (
        "run-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="New report directory; existing paths are never overwritten",
    )
    parser.add_argument("--min-context-recall", type=threshold, default=0.8)
    parser.add_argument("--min-answer-relevancy", type=threshold, default=0.7)
    parser.add_argument("--min-faithfulness", type=threshold, default=0.85)
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(run_evaluation(args)) else 1)


if __name__ == "__main__":
    main()
