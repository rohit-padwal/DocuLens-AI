"""Convenience entry point: uvicorn app:app --host 127.0.0.1 --port 8000."""

from doculens.api import create_app

app = create_app()
