# syntax=docker/dockerfile:1

# ---------------------------------------------------------------- builder ----
FROM python:3.14-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY agent_service ./agent_service

# Extras are configurable so an image can be built for a specific store, e.g.
#   docker build --build-arg EXTRAS="chroma,faiss,openai" .
ARG EXTRAS="chroma,faiss"
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install ".[${EXTRAS}]"

# ---------------------------------------------------------------- runtime ----
FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    HOST=0.0.0.0 \
    PORT=8000 \
    CHROMA_PERSIST_DIR=/data/chroma \
    FAISS_INDEX_PATH=/data/faiss

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 appuser \
 && mkdir -p /data && chown -R appuser:appuser /data

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY agent_service ./agent_service
COPY scripts ./scripts
COPY pyproject.toml README.md ./

USER appuser
EXPOSE 8000
VOLUME ["/data"]

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

# The console script calls uvicorn with log_config=None so our structured JSON
# logger stays the only handler.
CMD ["agent-service"]
