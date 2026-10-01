FROM python:3.13-slim

# Telemetry and hub access are disabled here and again in app.py, so the app
# cannot re-enable them by accident.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    HUGGINGFACE_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    ANONYMIZED_TELEMETRY=False \
    DO_NOT_TRACK=1 \
    NO_PROXY=* \
    TESSERACT_CMD=/usr/bin/tesseract \
    LLM_BASE_URL=http://host.docker.internal:11434/v1

# Tesseract powers scanned-page and image ingestion without any model round trip.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY .streamlit/ ./.streamlit/
COPY app.py README.md ./
COPY demo_data/ ./demo_data/
COPY tools/ ./tools/

# Non-root, and only the workspace is writable.
RUN useradd --create-home --uid 10001 servant \
    && mkdir -p /app/workspace/out \
    && chown -R servant:servant /app
USER servant

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"

# --server.address must be 0.0.0.0: .streamlit/config.toml pins 127.0.0.1 for
# native runs, and the flag overrides it inside the container.
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
