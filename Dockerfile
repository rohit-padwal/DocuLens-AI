FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DOCULENS_PERSIST_DIRECTORY=/app/data/chroma \
    LANGSMITH_TRACING=false \
    ANONYMIZED_TELEMETRY=false

WORKDIR /app
COPY pyproject.toml README.md constraints-py312.txt ./
COPY src ./src
RUN pip install -c constraints-py312.txt . && useradd --create-home --uid 10001 doculens \
    && mkdir -p /app/data/chroma && chown -R doculens:doculens /app/data

COPY app.py ./
COPY data/sample_docs ./data/sample_docs
USER doculens
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz')"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
