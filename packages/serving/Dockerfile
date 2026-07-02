# freight-parser — serving image (CPU / laptop stand-in)
#
# This is the CPU demo image: a python:3.11-slim base with no GPU, no CUDA.
# It runs the FastAPI app + `outlines` constrained decoding on the CPU so the
# whole thing fits on a conference laptop.
#
# PRODUCTION NOTE: in production this service ran on a GPU serving engine
# (e.g. vLLM) using the *same* constrained-decoding approach — identical
# prompt, identical JSON schema, identical reference guard — just with far
# more throughput and lower latency. This CPU image is the laptop stand-in
# for teaching / live demos, not the production deployment.
#
# Build from the REPO ROOT so the whole uv workspace is in the build context:
#   docker build -f packages/serving/Dockerfile -t freight-serving .
#   docker run --rm -p 8000:8000 freight-serving
# Then open http://localhost:8000/ for the playground.

FROM python:3.11-slim

# uv: fast, reproducible installs. Pulled from the official distroless image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# uv install hygiene inside a container layer.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Copy the entire workspace (root pyproject.toml, uv.lock, and every package).
# The serving package depends on the freight-schema and freight-finetune
# workspace members, so they must all be present for `uv sync` to resolve.
COPY . /app

# Resolve and install exactly what the committed lockfile pins.
RUN uv sync --frozen --no-editable

EXPOSE 8000

# Container-native health check: the API exposes GET /ready. No curl in slim,
# so use the bundled Python to probe it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4).status==200 else 1)"

# Serve the FastAPI app (defined in serving/app.py as `app`).
CMD ["uv", "run", "uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
