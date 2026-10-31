# Playwright needs Chromium + system libs. The official Playwright Python image
# ships them preinstalled and matches our playwright>=1.60 pin.
FROM mcr.microsoft.com/playwright/python:v1.60.0-noble

WORKDIR /app

# Install Python deps first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m playwright install chromium

COPY . .

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
