import asyncio
import hashlib

import pytest
from langchain_core.embeddings import Embeddings

from doculens.pipelines.index import ingest_directory, load_chunks, open_store
from doculens.pipelines.retrieval import HybridRetriever, budget_context, reciprocal_rank_fusion


class LocalEmbeddings(Embeddings):
    """Deterministic, local test embeddings; never used in the application."""

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [byte / 255 for byte in hashlib.sha256(text.encode()).digest()]


def test_fusion_deduplicates_each_ranking(chunk):
    other = chunk.model_copy(update={"id": "other"})
    assert reciprocal_rank_fusion([[chunk, chunk, other], [other]]) == [other, chunk]


def test_context_budget_preserves_complete_chunks(chunk):
    large = chunk.model_copy(update={"id": "large", "text": "x" * 100})
    assert budget_context([large, chunk], top_k=2, max_chars=50) == [chunk]


def test_real_chroma_ingestion_is_idempotent_and_removes_stale_chunks(settings, tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    path = root / "policy.md"
    path.write_text("Audit logs are retained for 90 days.", encoding="utf-8")
    store = open_store(settings, LocalEmbeddings())
    first = ingest_directory(root, settings, store)
    old_ids = store.get()["ids"]
    again = ingest_directory(root, settings, store)
    assert first["inserted"] == 1
    assert again["inserted"] == 0
    assert again["unchanged"] == 1
    path.write_text("Audit logs are now retained for 365 days.", encoding="utf-8")
    changed = ingest_directory(root, settings, store)
    assert changed["inserted"] == changed["removed"] == 1
    assert not set(old_ids) & set(store.get()["ids"])
    chunks = load_chunks(store, settings)
    assert chunks[0].source == "policy.md"
    retriever = HybridRetriever(store, chunks, 1, 5, 1000)
    assert asyncio.run(retriever.retrieve("audit retention")) == chunks


def test_index_configuration_change_is_rejected(settings, tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "policy.txt").write_text("Retention policy.", encoding="utf-8")
    store = open_store(settings, LocalEmbeddings())
    ingest_directory(root, settings, store)
    settings.embedding_model = "different-model"
    with pytest.raises(ValueError, match="configuration changed"):
        load_chunks(store, settings)


def test_oversized_file_is_rejected_before_embedding(settings, tmp_path):
    from doculens.pipelines.ingestion import chunk_file

    path = tmp_path / "large.txt"
    path.write_text("x" * 100, encoding="utf-8")
    settings.max_file_bytes = 10
    with pytest.raises(ValueError, match="exceeds"):
        chunk_file(path, tmp_path, settings)


def test_symlink_outside_document_root_is_rejected(tmp_path):
    from doculens.pipelines.ingestion import discover_files

    root = tmp_path / "docs"
    root.mkdir()
    external = tmp_path / "external.txt"
    external.write_text("private", encoding="utf-8")
    (root / "linked.txt").symlink_to(external)
    with pytest.raises(ValueError, match="escapes"):
        discover_files(root)


def test_pdf_chunks_preserve_one_based_page_numbers(settings, tmp_path):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    from doculens.pipelines.ingestion import chunk_file

    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    for text in (b"Retention is 90 days.", b"Backups run daily."):
        page = writer.add_blank_page(width=300, height=300)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )
        content = DecodedStreamObject()
        content.set_data(b"BT /F1 12 Tf 50 100 Td (" + text + b") Tj ET")
        page[NameObject("/Contents")] = writer._add_object(content)
    path = tmp_path / "policy.pdf"
    writer.write(path)
    chunks = chunk_file(path, tmp_path, settings)
    assert [doc.metadata["page"] for doc in chunks] == [1, 2]
    assert "90 days" in chunks[0].page_content
