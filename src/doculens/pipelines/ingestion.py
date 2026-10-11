"""Bounded document loading and deterministic chunking for PDF, Markdown, and text."""

import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from doculens.config import Settings

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}


def chunk_file(path: Path, root: Path, settings: Settings) -> list[Document]:
    resolved = path.resolve()
    source = resolved.relative_to(root.resolve()).as_posix()
    if resolved.stat().st_size > settings.max_file_bytes:
        raise ValueError(f"File exceeds max_file_bytes: {source}")
    if resolved.suffix.lower() == ".pdf":
        reader = PdfReader(resolved)
        if reader.is_encrypted:
            raise ValueError(f"Encrypted PDF is unsupported: {source}")
        pages = [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, 1)]
    else:
        pages = [(0, resolved.read_text(encoding="utf-8"))]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    result: list[Document] = []
    for page_number, text in pages:
        if not text.strip():
            continue
        parts = splitter.create_documents([text])
        for part in parts:
            fingerprint = "\0".join(
                [source, str(page_number), str(part.metadata["start_index"]), part.page_content]
            )
            chunk_id = "chunk_" + hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
            part.metadata.update(
                chunk_id=chunk_id,
                source=source,
                page=page_number,
                embedding_model=settings.embedding_model,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            result.append(part)
    if not result:
        raise ValueError(f"No extractable text: {source}; scanned PDFs require an OCR stage")
    return result


def discover_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"Document directory does not exist: {root}")
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise ValueError("No PDF, Markdown, or text files found")
    for path in files:
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Document symlink escapes the document directory: {path.name}")
    return files
