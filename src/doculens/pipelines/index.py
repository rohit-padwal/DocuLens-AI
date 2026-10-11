"""Chroma is the source of truth for both semantic and lexical retrieval."""

from pathlib import Path

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from doculens.config import Settings
from doculens.pipelines.ingestion import chunk_file, discover_files
from doculens.pipelines.retrieval import HybridRetriever
from doculens.schemas import Chunk


def open_store(settings: Settings, embeddings: Embeddings | None = None) -> Chroma:
    if embeddings is None:
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.require_provider_key(),
            request_timeout=settings.model_timeout_seconds,
            max_retries=2,
        )
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.persist_directory),
        collection_metadata={"hnsw:space": "cosine"},
        client_settings=ChromaSettings(anonymized_telemetry=False),
    )


def load_chunks(store: Chroma, settings: Settings) -> list[Chunk]:
    data = store.get(include=["documents", "metadatas"])
    chunks = []
    for chunk_id, text, metadata in zip(
        data["ids"], data["documents"], data["metadatas"], strict=True
    ):
        expected = {
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Index configuration changed; ingest into a new collection")
        if metadata.get("chunk_id") != chunk_id:
            raise ValueError("Index contains invalid chunk metadata")
        chunks.append(
            Chunk(id=chunk_id, text=text, source=metadata["source"], page=metadata["page"] or None)
        )
    return sorted(chunks, key=lambda chunk: chunk.id)


def ingest_directory(root: Path, settings: Settings, store: Chroma | None = None) -> dict[str, int]:
    files = discover_files(root)
    # Validate every input before opening provider clients or mutating an existing index.
    # Embedding failures can still interrupt writes; this is a parsing preflight, not a transaction.
    prepared = [chunk_file(path, root, settings) for path in files]
    store = store if store is not None else open_store(settings)
    load_chunks(store, settings)  # Reject incompatible collections before making any changes.
    inserted = removed = unchanged = 0
    for documents in prepared:
        source = documents[0].metadata["source"]
        old_ids = set(store.get(where={"source": source})["ids"])
        new_ids = {doc.metadata["chunk_id"] for doc in documents}
        fresh = [doc for doc in documents if doc.metadata["chunk_id"] not in old_ids]
        for offset in range(0, len(fresh), 128):
            batch = fresh[offset : offset + 128]
            store.add_documents(batch, ids=[doc.metadata["chunk_id"] for doc in batch])
        # Delete superseded chunks only after the new embeddings have been persisted.
        stale = sorted(old_ids - new_ids)
        if stale:
            store.delete(ids=stale)
        inserted += len(fresh)
        removed += len(stale)
        unchanged += len(new_ids & old_ids)
    return {"files": len(files), "inserted": inserted, "removed": removed, "unchanged": unchanged}


def build_retriever(settings: Settings) -> HybridRetriever:
    store = open_store(settings)
    return HybridRetriever(
        store,
        load_chunks(store, settings),
        settings.top_k,
        settings.candidate_k,
        settings.max_context_chars,
    )
