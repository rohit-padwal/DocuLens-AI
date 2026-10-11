"""Validated configuration; loading does not instantiate external clients."""

from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCULENS_", env_file=".env", extra="ignore", populate_by_name=True
    )

    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    api_key: SecretStr | None = None
    generation_model: str = "gpt-4o-mini"
    verifier_model: str = "gpt-4o-mini"
    evaluation_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    persist_directory: Path = Path("data/chroma")
    collection_name: str = "doculens"
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=150, ge=0)
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=20, ge=1, le=100)
    max_context_chars: int = Field(default=16000, ge=1000, le=100000)
    max_revisions: int = Field(default=2, ge=0, le=5)
    max_hallucination_risk: float = Field(default=0.15, ge=0, le=1)
    model_timeout_seconds: float = Field(default=30, gt=0, le=120)
    request_timeout_seconds: float = Field(default=120, gt=0, le=600)
    max_concurrent_requests: int = Field(default=4, ge=1, le=32)
    max_request_bytes: int = Field(default=16384, ge=1024, le=1_000_000)
    max_file_bytes: int = Field(default=20_000_000, ge=1)

    @model_validator(mode="after")
    def check_constraints(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be at least top_k")
        return self

    def require_provider_key(self) -> SecretStr:
        if not self.openai_api_key or not self.openai_api_key.get_secret_value().strip():
            raise ValueError("Set OPENAI_API_KEY in .env or the environment")
        return self.openai_api_key
