"""Composition root shared by the command line and HTTP interface."""

from doculens.agents.graph import RagPipeline
from doculens.agents.models import StructuredAgents
from doculens.config import Settings
from doculens.pipelines.index import build_retriever


def build_pipeline(settings: Settings) -> RagPipeline:
    return RagPipeline(build_retriever(settings), StructuredAgents(settings), settings)
