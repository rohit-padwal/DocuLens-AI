"""Local ingestion and question-answering commands."""

import argparse
import asyncio
import json
from pathlib import Path

from doculens.config import Settings
from doculens.pipelines.index import ingest_directory
from doculens.service import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="DocuLens-AI: ingest documents or ask a question")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser(
        "ingest", help="Index a directory of PDF, Markdown, and text files"
    )
    ingest.add_argument("directory", type=Path)
    ask = commands.add_parser("ask", help="Run retrieval, generation, and verification")
    ask.add_argument("question")
    args = parser.parse_args()
    settings = Settings()
    if args.command == "ingest":
        print(json.dumps(ingest_directory(args.directory, settings), indent=2))
    else:
        result = asyncio.run(build_pipeline(settings).run(args.question))
        print(result.answer.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
