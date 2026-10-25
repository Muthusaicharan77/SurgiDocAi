# SurgiDoc API and web UI (CPU). Code and configuration only: model weights, the
# guideline index and run data are mounted at runtime (see docker-compose.yml).

FROM node:22-slim AS ui
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY ui ./
RUN npm run build

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU-only PyTorch first, so the extras below do not pull the CUDA build.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[agent,trace,serve,download]"

# Paths the code reads relative to the working directory.
COPY configs ./configs
COPY reports ./reports
COPY scripts ./scripts
COPY --from=ui /ui/dist ./ui/dist

RUN useradd --create-home surgidoc && mkdir -p data runs weights \
    && chown -R surgidoc /app
USER surgidoc

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "surgidoc.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
