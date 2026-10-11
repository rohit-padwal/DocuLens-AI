"""Dense MMR + BM25 retrieval fused by reciprocal rank, with a context budget."""

import asyncio
import re
from collections import defaultdict
from collections.abc import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Plus

from doculens.schemas import Chunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def as_chunk(document: Document) -> Chunk:
    return Chunk(
        id=document.metadata["chunk_id"],
        text=document.page_content,
        source=document.metadata["source"],
        page=document.metadata.get("page") or None,
    )


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Chunk]], k: int = 60) -> list[Chunk]:
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, chunk in enumerate(ranking, 1):
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            chunks[chunk.id] = chunk
            scores[chunk.id] += 1 / (k + rank)
    return [chunks[key] for key in sorted(scores, key=lambda key: (-scores[key], key))]


def budget_context(chunks: Sequence[Chunk], top_k: int, max_chars: int) -> list[Chunk]:
    """Keep full chunks only: returned citations refer to precisely what the LLM sees."""
    selected: list[Chunk] = []
    seen: set[str] = set()
    used = 0
    for chunk in chunks:
        if chunk.id in seen:
            continue
        if len(selected) >= top_k:
            break
        if used + len(chunk.text) > max_chars:
            continue
        selected.append(chunk)
        seen.add(chunk.id)
        used += len(chunk.text)
    return selected


class HybridRetriever:
    def __init__(
        self, store: Chroma, chunks: list[Chunk], top_k: int, candidate_k: int, max_chars: int
    ) -> None:
        if not chunks:
            raise ValueError("Index is empty; run doculens ingest first")
        self.store = store
        self.chunks = chunks
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.max_chars = max_chars
        # Okapi IDF can be zero/negative for common terms in small collections. BM25Plus
        # keeps IDF positive; delta=0 avoids giving nonmatching chunks a baseline score.
        self.bm25 = BM25Plus([tokenize(chunk.text) or ["__empty__"] for chunk in chunks], delta=0.0)

    def _search(self, query: str) -> list[Chunk]:
        dense = self.store.max_marginal_relevance_search(
            query, k=self.candidate_k, fetch_k=self.candidate_k * 2, lambda_mult=0.7
        )
        scores = self.bm25.get_scores(tokenize(query))
        indices = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        # Do not promote arbitrary documents when the query has no lexical matches.
        lexical = [self.chunks[i] for i in indices if scores[i] > 0][: self.candidate_k]
        fused = reciprocal_rank_fusion([[as_chunk(doc) for doc in dense], lexical])
        return budget_context(fused, self.top_k, self.max_chars)

    async def retrieve(self, query: str) -> list[Chunk]:
        return await asyncio.to_thread(self._search, query)
