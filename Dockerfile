# MedQuery: one image serving the API and the built frontend, with corpus v2 built inside.
#   docker compose up            (full image: downloads the 10 public PDFs, builds corpus v2)
#   docker build --target app .  (CI: app only, no downloads)
# No secrets are baked in: GROQ_API_KEY is read from the environment at run time.

# --- 1. frontend: the static build, calling the API on the same origin ---------------------
FROM node:22.23.2-bookworm-slim AS frontend
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN VITE_API_URL="" npm run build

# --- 2. app: Python dependencies from the lock, the medquery package, the frontend ----------
FROM python:3.11.16-slim-bookworm AS app
COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /usr/local/bin/uv
# Files are copied in already owned by the non-root user (a chown afterwards would duplicate
# every layer), and uv keeps no download cache in the image.
RUN useradd --create-home --uid 1000 app && mkdir -p /app/data && chown -R app:app /app
USER app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY --chown=app:app pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY --chown=app:app medquery/ medquery/
COPY --chown=app:app corpus/manifest.toml corpus/manifest.toml
COPY --chown=app:app docs/SOURCES.md docs/SOURCES.md
RUN uv sync --locked --no-dev
COPY --from=frontend --chown=app:app /src/frontend/dist frontend/dist

# --- 3. corpus: download the PDFs (sha256-checked), build corpus v2, cache embeddings -------
FROM app AS corpus
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*
USER app
RUN python -m medquery.ingest download \
    && python -m medquery.ingest build --corpus v2 \
    && python -m medquery.api.prepare

# --- 4. runtime (default): the app plus the prepared data; no OCR tools ---------------------
FROM app AS runtime
# Only what the live app reads: corpus v2 and the embedding/model caches (not the raw PDFs).
COPY --from=corpus --chown=app:app /app/data/processed/v2 /app/data/processed/v2
COPY --from=corpus --chown=app:app /app/data/cache /app/data/cache
ENV MEDQUERY_HOST=0.0.0.0 \
    MEDQUERY_PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import json,os,sys,urllib.request as u; r=json.load(u.urlopen('http://127.0.0.1:%s/health' % os.environ.get('MEDQUERY_PORT','8000'), timeout=4)); sys.exit(0 if r.get('ready') else 1)"
CMD ["python", "-m", "medquery.api"]
