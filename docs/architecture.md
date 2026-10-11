# Implementation blueprint

## Component boundaries

`config.py` validates environment variables and cross-field constraints. `schemas.py`
defines transport and provider contracts. `service.py` composes concrete adapters; tests
inject local doubles into the same graph. No provider call happens during module import.

| Component | Responsibility | Key invariant |
| --- | --- | --- |
| `pipelines/ingestion.py` | Load UTF-8 text, Markdown, and PDF pages; split with overlap | Source paths stay within the selected document root |
| `pipelines/index.py` | Persist Chroma chunks and embeddings | IDs are deterministic; index configuration must match |
| `pipelines/retrieval.py` | Fuse semantic MMR and BM25Plus rankings | Context contains unique full chunks within a character budget |
| `agents/models.py` | Separate generation and verification prompts | Every provider result must satisfy a Pydantic schema |
| `agents/graph.py` | Execute the bounded verification loop | A rejected draft is never returned as the final answer |
| `api.py` / `middleware.py` | Bound payloads, validate requests, authenticate, limit concurrency | Body limits apply before parsing; provider errors are sanitized |
| `evaluation/run.py` | Execute held-out cases and score actual outputs | Missing/invalid scores cannot silently pass |

## Request lifecycle

1. Validate and strip the question (maximum 2,000 characters).
2. Retrieve dense candidates with maximal marginal relevance to reduce redundancy.
   Independently score the stored corpus with BM25Plus (`delta=0` to avoid boosting
   nonmatches). Positive IDF keeps common terms useful in small corpora. Fuse rankings using
   `sum(1 / (60 + rank))`; these scores order candidates, they are not confidence estimates.
3. Keep up to `top_k` complete chunks within `max_context_chars` (a character bound,
   not an exact tokenizer count). Pass exactly these chunks to the generator.
4. Generate a typed draft with inline chunk citations. A declared evidence gap leads
   directly to abstention.
5. Verify every factual claim with a separate prompt and configurable verifier model.
   Require grounding, question coverage, supported claims, valid evidence IDs, and an
   estimated hallucination risk below the threshold. The score is not calibrated.
6. On rejection, retain critique feedback and retrieve with both the original question
   and a focused revised query. Context is replaced, not indefinitely accumulated.
7. Return a verified answer with cited excerpts, or abstain after `1 + max_revisions`
   drafts. An end-to-end asynchronous deadline covers the entire graph.

Every retrieval attempt clears the previous verifier report. If a new attempt has
insufficient evidence, its abstention cannot expose claim checks for a different draft
or different chunks. Empty claim text also fails the deterministic evidence gate.

The generator and verifier are separate roles; using the same underlying model can
produce correlated mistakes. Configure different models when testing that tradeoff.
Deterministic gates validate structure and evidence references. Semantic entailment
still depends on the verifier; valid citation IDs alone do not establish truth.

## Index lifecycle

Use one stable document root per collection. Sources are paths relative to that root;
reusing a collection for a different root with the same filenames can replace sources.
IDs hash source, page, start offset, and text. Repeated ingestion skips unchanged chunks.
For changed files, new chunks are added before stale chunks are removed. Files missing
from a subsequent ingestion directory remain indexed; build a new collection to obtain
an exact clean snapshot, or implement an explicit reviewed deletion command.

Embedding model and chunk settings are stored with each chunk. Incompatible settings
fail early; select a new collection to rebuild. Chroma is also the lexical corpus source,
so there is no separate BM25 manifest to drift out of sync.

This implementation targets a small to medium single-tenant corpus. BM25 loads the
entire collection into memory at startup. Parsing preflight loads every input before
any writes or provider setup, so a malformed later file leaves the index unchanged.
Preflight holds the prepared chunks in memory; factor that into ingestion capacity.
Ingestion is not a multi-file transaction; embedding or persistence failures can still
leave an incomplete update. Stop query workers during ingestion and
restart them afterward. For continuously updated production corpora, build versioned
collections and atomically switch an external active-index pointer after validation.

## Evaluation design

Every answerable JSONL row has a human-written reference answer. Context recall judges
how much of that reference is supported by retrieved chunks. Answer relevancy compares
the input with generated questions about the response. Faithfulness judges the support
for response claims. Expected source paths also support a deterministic source-level
recall diagnostic, which is distinct from claim-level Ragas context recall.

Only answerable cases contribute to the three metric means. An abstention on an
answerable case gets zero answer relevancy and faithfulness; retrieval recall is still
measured. Unanswerable cases are evaluated through status accuracy because recall and
faithfulness have no meaningful gold facts. The gate requires every expected status to
match in addition to metric thresholds. Exceptions and NaN/inf scores fail the run.

The CLI exclusively creates a new output directory before any provider setup. Existing
directories are rejected without modification. Each completed summary is written to a
temporary file and atomically renamed. A failed run writes an explicit failing summary
with the exception type and preserves any partial JSONL results for debugging. No raw
exception text is saved because provider errors may contain confidential data.

Store `results.jsonl` and `summary.json` together. Keep the dataset, model names,
dependency versions, prompts, and corpus fixed for comparisons. Judge scores vary
even with temperature zero; repeat runs and manually audit claim judgments before
drawing conclusions. The tiny sample set demonstrates wiring, not general quality.

## Production extension plan

The repository supplies tested service boundaries and operational safeguards. Before
deploying it for multiple customers, implement:

- Tenant-scoped retrieval filters and authorization for each returned source.
- A gateway with TLS, request-body limits, per-client rate limits, and quotas.
- A managed vector service or Chroma server for multiple replicas; local persistence
  is intended for a single process, not horizontally scaled shared filesystem writes.
- Versioned index builds, backups, rollback, and immutable evaluation corpus snapshots.
- Metrics for latency, token cost, provider failures, abstention, and judge disagreement.
- A larger reviewed benchmark, calibrated verifier thresholds, and human escalation.
- OCR for scanned PDFs, richer table parsing, and a cross-encoder reranker if justified
  by ablation results.

Cancellation stops waiting for an asynchronous operation, but cannot kill a blocking
retrieval thread already started by `asyncio.to_thread`; per-provider timeouts remain
necessary. The model's schema adherence and critique do not provide an injection-proof
security boundary. The graph has no arbitrary tool execution privileges.
