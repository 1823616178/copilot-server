# Playwright needs Chromium + system libs. The official Playwright Python image
# ships them preinstalled. Keep its version in sync with requirements.txt.
FROM mcr.microsoft.com/playwright/python:v1.60.0-noble

WORKDIR /app

# Override for networks where PyPI is unreachable.
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_DEFAULT_TIMEOUT=60
ARG PIP_RETRIES=3

# Install Python deps first so the layer caches across code changes.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --disable-pip-version-check \
        --index-url "${PIP_INDEX_URL}" \
        --timeout "${PIP_DEFAULT_TIMEOUT}" --retries "${PIP_RETRIES}" \
        -r requirements.txt \
    && python -m pip check

COPY app.py .
COPY copilot/ copilot/
COPY server/ server/

# Serve on all interfaces inside the container; map the port in compose.
ENV HOST=0.0.0.0 \
    PORT=8000 \
    PYTHONUNBUFFERED=1 \
    COPILOT_HEADLESS=1 \
    TZ=Asia/Shanghai

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=4)" || exit 1

CMD ["python", "app.py"]
