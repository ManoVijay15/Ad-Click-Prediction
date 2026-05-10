FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Source must be present for setuptools to discover the src package
COPY pyproject.toml ./
COPY src/ ./src/

# Non-editable user install — bakes the package into ~/.local
RUN pip install --no-cache-dir --user .


FROM python:3.11-slim AS runtime

# OpenMP runtime for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app && useradd --system --gid app app

WORKDIR /app

COPY --from=builder --chown=app:app /root/.local /home/app/.local

ENV PATH=/home/app/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/health'); sys.exit(0)" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
