"""FastAPI service with startup validation, admission control, and sanitized errors."""

import asyncio
import logging
import secrets
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from doculens.agents.graph import RagPipeline
from doculens.config import Settings
from doculens.schemas import Answer, QueryRequest
from doculens.service import build_pipeline

logger = logging.getLogger("doculens.api")


def create_app(
    settings: Settings | None = None,
    pipeline_factory: Callable[[Settings], RagPipeline] = build_pipeline,
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = await asyncio.to_thread(pipeline_factory, settings)
        app.state.active_requests = 0
        app.state.ready = True
        yield
        app.state.ready = False

    app = FastAPI(title="DocuLens-AI", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        started = perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            # Log exception types only: provider exceptions can contain prompts or credentials.
            logger.error("request_failed id=%s type=%s", request_id, type(exc).__name__)
            response = JSONResponse(status_code=500, content={"detail": "Internal server error"})
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request id=%s method=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            response.status_code,
            (perf_counter() - started) * 1000,
        )
        return response

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready():
        if not getattr(app.state, "ready", False):
            raise HTTPException(status_code=503, detail="Service is not ready")
        return {"status": "ready"}

    @app.post("/query", response_model=Answer)
    async def query(
        payload: QueryRequest,
        api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    ):
        configured = settings.api_key.get_secret_value() if settings.api_key else ""
        if configured and not secrets.compare_digest((api_key or "").encode(), configured.encode()):
            raise HTTPException(status_code=401, detail="Invalid API key")
        # No awaits between the capacity check and increment: atomic within one event loop.
        if app.state.active_requests >= settings.max_concurrent_requests:
            raise HTTPException(status_code=429, detail="Service is at capacity; retry later")
        app.state.active_requests += 1
        try:
            result = await app.state.pipeline.run(payload.question)
            return result.answer
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Query deadline exceeded") from exc
        except Exception as exc:
            logger.warning("pipeline_failed type=%s", type(exc).__name__)
            raise HTTPException(
                status_code=502, detail="Retrieval or model provider failed"
            ) from exc
        finally:
            app.state.active_requests -= 1

    return app
