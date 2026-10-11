import pytest

from doculens.pipelines.index import ingest_directory, open_store
from tests.test_retrieval import LocalEmbeddings


def test_bad_later_file_does_not_partially_update_existing_index(settings, tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    valid = root / "a-policy.txt"
    valid.write_text("Old retention policy", encoding="utf-8")
    store = open_store(settings, LocalEmbeddings())
    ingest_directory(root, settings, store)
    before = store.get(include=["documents", "metadatas"])
    valid.write_text("Changed retention policy", encoding="utf-8")
    (root / "z-broken.txt").write_bytes(b"\xff\xfe")
    with pytest.raises(UnicodeDecodeError):
        ingest_directory(root, settings, store)
    assert store.get(include=["documents", "metadatas"]) == before


def test_invalid_input_is_reported_before_constructing_provider_clients(settings, tmp_path):
    (tmp_path / "empty.md").write_text("   ", encoding="utf-8")
    settings.openai_api_key = None
    with pytest.raises(ValueError, match="No extractable text"):
        ingest_directory(tmp_path, settings)
