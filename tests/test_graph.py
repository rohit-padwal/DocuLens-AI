import asyncio

import pytest

from doculens.agents.graph import RagPipeline, evidence_errors


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.queries = []

    async def retrieve(self, query):
        self.queries.append(query)
        return self.chunks


class FakeAgents:
    def __init__(self, draft, reports):
        self.draft = draft
        self.reports = iter(reports)
        self.feedback = []

    async def generate(self, question, chunks, feedback):
        self.feedback.append(feedback)
        return self.draft

    async def verify(self, question, chunks, draft):
        return next(self.reports)


def test_verified_answer_keeps_citations_and_trace(settings, chunk, draft, verification):
    pipeline = RagPipeline(FakeRetriever([chunk]), FakeAgents(draft, [verification]), settings)
    result = asyncio.run(pipeline.run("What is the retention period?"))
    assert result.answer.status == "verified"
    assert result.answer.sources == [chunk]
    assert result.answer.attempts == 1
    assert [event.node for event in result.answer.trace] == ["retrieve", "generate", "verify"]


def test_feedback_and_revised_query_drive_second_attempt(settings, chunk, draft, verification):
    bad = verification.model_copy(
        update={
            "grounded": False,
            "hallucination_risk": 0.9,
            "feedback": "Check retention evidence",
        }
    )
    retriever = FakeRetriever([chunk])
    agents = FakeAgents(draft, [bad, verification])
    result = asyncio.run(RagPipeline(retriever, agents, settings).run("retention?"))
    assert result.answer.status == "verified"
    assert result.answer.attempts == 2
    assert "Check retention evidence" in agents.feedback[1]
    assert bad.revised_query in retriever.queries
    assert retriever.queries.count("retention?") == 2


def test_exhausted_budget_abstains_and_hides_rejected_draft(settings, chunk, draft, verification):
    bad = verification.model_copy(update={"hallucination_risk": 0.9})
    pipeline = RagPipeline(FakeRetriever([chunk]), FakeAgents(draft, [bad] * 3), settings)
    result = asyncio.run(pipeline.run("retention?"))
    assert result.answer.status == "abstained"
    assert result.answer.attempts == settings.max_revisions + 1
    assert result.answer.sources == []
    assert result.answer.answer != draft.answer
    assert result.retrieved_chunks == [chunk]


def test_zero_revision_budget_allows_one_attempt(settings, chunk, draft, verification):
    settings.max_revisions = 0
    bad = verification.model_copy(update={"grounded": False})
    result = asyncio.run(
        RagPipeline(FakeRetriever([chunk]), FakeAgents(draft, [bad]), settings).run("retention?")
    )
    assert result.answer.status == "abstained"
    assert result.answer.attempts == 1


def test_empty_retrieval_skips_models(settings):
    result = asyncio.run(RagPipeline(FakeRetriever([]), None, settings).run("retention?"))
    assert result.answer.status == "abstained"
    assert result.answer.attempts == 0


def test_insufficient_context_skips_verifier(settings, chunk, draft):
    draft.insufficient_context = True
    result = asyncio.run(
        RagPipeline(FakeRetriever([chunk]), FakeAgents(draft, []), settings).run(
            "What is the price?"
        )
    )
    assert result.answer.status == "abstained"
    assert result.answer.attempts == 1


@pytest.mark.parametrize(
    "mode",
    [
        "unknown",
        "inline_mismatch",
        "unsupported",
        "empty_claims",
        "wrong_evidence",
        "coverage",
        "empty_answer",
        "blank_claim",
    ],
)
def test_deterministic_gate_rejects_bad_evidence(mode, chunk, draft, verification):
    if mode == "unknown":
        draft.answer = "A fact [unknown]"
        draft.citations = ["unknown"]
    elif mode == "inline_mismatch":
        draft.answer = "A fact with no citation"
    elif mode == "unsupported":
        verification.claims[0].supported = False
    elif mode == "empty_claims":
        verification.claims = []
    elif mode == "wrong_evidence":
        verification.claims[0].evidence_ids = ["unknown"]
    elif mode == "coverage":
        verification.answers_question = False
    elif mode == "blank_claim":
        verification.claims[0].claim = "   "
    else:
        draft.answer = ""
    assert evidence_errors(draft, verification, [chunk])


def test_deadline_includes_model_calls(settings, chunk):
    class SlowAgents:
        async def generate(self, *args):
            await asyncio.sleep(1)

    settings.request_timeout_seconds = 0.01
    with pytest.raises(TimeoutError):
        asyncio.run(RagPipeline(FakeRetriever([chunk]), SlowAgents(), settings).run("retention?"))


def test_runs_do_not_share_graph_state(settings, chunk, draft, verification):
    pipeline = RagPipeline(FakeRetriever([chunk]), FakeAgents(draft, [verification] * 2), settings)

    async def run_two():
        return await asyncio.gather(pipeline.run("first?"), pipeline.run("second?"))

    first, second = asyncio.run(run_two())
    assert first.answer.question == "first?"
    assert second.answer.question == "second?"
    assert first.answer.attempts == second.answer.attempts == 1
    assert len(first.answer.trace) == len(second.answer.trace) == 3


def test_abstention_after_empty_retry_does_not_return_stale_verification(
    settings, chunk, draft, verification
):
    class DisappearingEvidence:
        def __init__(self):
            self.calls = 0

        async def retrieve(self, query):
            self.calls += 1
            return [chunk] if self.calls == 1 else []

    bad = verification.model_copy(update={"grounded": False})
    result = asyncio.run(
        RagPipeline(DisappearingEvidence(), FakeAgents(draft, [bad]), settings).run("retention?")
    )
    assert result.answer.status == "abstained"
    assert result.answer.verification is None
    assert result.retrieved_chunks == []


def test_insufficient_context_on_revision_does_not_return_previous_claim_checks(
    settings, chunk, draft, verification
):
    class RevisedAgents(FakeAgents):
        async def generate(self, question, chunks, feedback):
            return (
                draft if not feedback else draft.model_copy(update={"insufficient_context": True})
            )

    bad = verification.model_copy(update={"grounded": False})
    result = asyncio.run(
        RagPipeline(FakeRetriever([chunk]), RevisedAgents(draft, [bad]), settings).run("retention?")
    )
    assert result.answer.status == "abstained"
    assert result.answer.attempts == 2
    assert result.answer.verification is None
