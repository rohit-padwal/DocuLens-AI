import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from doculens.agents.graph import RagPipeline
from doculens.evaluation.run import (
    RagasJudge,
    evaluate_cases,
    finite_score,
    load_cases,
    summarize,
)
from doculens.evaluation.run import (
    TestCase as EvalCase,
)
from tests.test_graph import FakeAgents, FakeRetriever


def test_sample_cases_have_real_gold_answers():
    cases = load_cases(Path("data/evaluation/cases.jsonl"))
    assert len(cases) == 11
    assert sum(case.expected_status == "abstained" for case in cases) == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nan_or_infinite_scores_fail_evaluation(value):
    with pytest.raises(ValueError, match="non-finite"):
        finite_score(value)


def test_actual_context_is_sent_to_judge(settings, chunk, draft, verification, tmp_path):
    class Judge:
        async def score(self, question, response, contexts, reference):
            assert contexts == [chunk.text]
            assert response == draft.answer
            assert reference == "90 days"
            return {"context_recall": 1.0, "answer_relevancy": 0.8, "faithfulness": 1.0}

    pipeline = RagPipeline(FakeRetriever([chunk]), FakeAgents(draft, [verification]), settings)
    cases = [EvalCase(id="retention", question="retention?", reference="90 days")]
    path = tmp_path / "results.jsonl"
    rows = asyncio.run(evaluate_cases(cases, pipeline, Judge(), path))
    assert json.loads(path.read_text())["retrieved_contexts"] == [chunk.text]
    assert summarize(rows, {"context_recall": 0.8})["passed"]
    assert not summarize(rows, {"answer_relevancy": 0.9})["passed"]


def test_answerable_abstentions_count_as_failures(settings, chunk, draft, tmp_path):
    class Recall:
        async def ascore(self, **kwargs):
            return SimpleNamespace(value=0.6)

    draft.insufficient_context = True
    pipeline = RagPipeline(FakeRetriever([chunk]), FakeAgents(draft, []), settings)
    cases = [EvalCase(id="retention", question="retention?", reference="90 days")]
    rows = asyncio.run(
        evaluate_cases(
            cases, pipeline, SimpleNamespace(recall=Recall()), tmp_path / "results.jsonl"
        )
    )
    assert rows[0]["metrics"]["answer_relevancy"] == 0
    assert rows[0]["metrics"]["context_recall"] == 0.6
    assert not summarize(rows, {"context_recall": 0.5})["passed"]


def test_ragas_collection_api_constructs_without_network(settings):
    from pydantic import SecretStr

    settings.openai_api_key = SecretStr("offline-test-key")
    judge = RagasJudge(settings)
    assert callable(judge.recall.ascore)
    assert callable(judge.relevance.ascore)
    assert callable(judge.faithfulness.ascore)
    asyncio.run(judge.close())


def test_real_ragas_metrics_score_through_mock_provider(settings):
    """Exercise SDK parsing, Ragas prompts, embeddings, and metric result contracts offline."""
    from collections import deque

    import httpx
    from openai import AsyncOpenAI

    responses = deque(
        [
            {
                "classifications": [
                    {"statement": "90 days", "reason": "In context", "attributed": 1}
                ]
            },
            *[{"question": "What is the retention period?", "noncommittal": 0}] * 3,
            {"statements": ["Audit logs are retained for 90 days."]},
            {
                "statements": [
                    {
                        "statement": "Audit logs are retained for 90 days.",
                        "reason": "In context",
                        "verdict": 1,
                    }
                ]
            },
        ]
    )

    def handle(request):
        payload = json.loads(request.content)
        if request.url.path.endswith("/embeddings"):
            inputs = payload["input"]
            count = 1 if isinstance(inputs, str) else len(inputs)
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": settings.embedding_model,
                    "data": [
                        {"object": "embedding", "index": i, "embedding": [1.0, 0.0, 0.0]}
                        for i in range(count)
                    ],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": settings.evaluation_model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(responses.popleft()),
                        },
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    async def score():
        client = AsyncOpenAI(
            api_key="offline-test-key",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        )
        judge = RagasJudge(settings, client=client)
        try:
            return await judge.score(
                "What is the retention period?",
                "Audit logs are retained for 90 days.",
                ["Audit logs are retained for 90 days."],
                "90 days",
            )
        finally:
            await judge.close()

    scores = asyncio.run(score())
    assert scores == {"context_recall": 1.0, "answer_relevancy": 1.0, "faithfulness": 1.0}
    assert not responses
