# Gateway image — compliance, streaming, and metrics. No model, no CUDA.
#
# The GPU stack lives entirely in the vLLM container. Keeping it out of here is
# not just an image-size decision: it means the UPL disclaimer text can be
# changed and redeployed in seconds, without restarting a process that holds an
# 8B model in VRAM and takes minutes to warm up.
#
# That is the fourth argument from serve/disclaimer.py cashed out in the
# deployment topology. Putting the disclaimer in the weights makes changing one
# word a retrain; putting it in the model *server* would make it a reload of the
# whole model. Putting it in a separate stateless process makes it a rolling
# restart of a ~200MB container.

FROM python:3.12-slim AS runtime

# uv is used here for the same reason as in development — one resolver, one lock
# file, no drift between what CI tested and what ships.
COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Only the `serve` extra, installed explicitly rather than via `uv sync`.
#
# `uv sync --extra serve` would also pull the project's base dependencies —
# datasets, pyarrow, scipy, the Anthropic SDK — none of which the serving path
# imports. Those belong to data synthesis and evaluation, which do not run here.
# Pinned identically to pyproject.toml's serve extra; CI runs the same versions.
RUN uv pip install --no-cache \
    "fastapi==0.115.12" \
    "uvicorn[standard]==0.34.3" \
    "httpx==0.28.1" \
    "prometheus-client==0.22.1" \
    "pydantic==2.11.5" \
    "sse-starlette==2.3.6"

COPY src/legalmind/__init__.py /app/src/legalmind/__init__.py
COPY src/legalmind/serve /app/src/legalmind/serve

# Build-time guard. If the serving path ever grows an import that is not in the
# list above, the build fails here rather than the container crash-looping in
# production with an ImportError. It also pins the claim this image makes: the
# gateway depends on nothing heavy.
RUN python -c "from legalmind.serve.gateway import app; print('gateway imports clean')"

# Non-root. The gateway reads no files and writes none; there is nothing it
# needs root for.
RUN useradd --create-home --uid 10001 legalmind
USER legalmind

EXPOSE 8080

# Checks the app, not the port: a process that is listening but cannot serve
# /healthz is exactly the state a port check reports as healthy.
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/healthz', timeout=2).status==200 else 1)"

CMD ["uvicorn", "legalmind.serve.gateway:app", "--host", "0.0.0.0", "--port", "8080"]
