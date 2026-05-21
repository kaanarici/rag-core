# Minimal image for ``rag-core serve`` (Journey C compose).
FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY src ./src

RUN pip install --no-cache-dir '.[runtime]'

EXPOSE 8787

# Override via compose ``command`` / environment (see docs/self-host/config.md).
CMD [
  "rag-core",
  "serve",
  "--host",
  "0.0.0.0",
  "--port",
  "8787",
  "--qdrant-url",
  "http://qdrant:6333",
  "--embedding-provider",
  "demo",
  "--embedding-model",
  "demo-dense-v1",
  "--embedding-dimensions",
  "64",
]
