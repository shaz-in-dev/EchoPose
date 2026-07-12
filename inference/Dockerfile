FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ── Model checkpoint ─────────────────────────────────────────────────────────
# models/pose_net.pt is gitignored (9.3 MB). If the file is missing the server
# starts in simulation mode — useful for evaluating the UI without hardware.
# To use a real model, mount it at runtime:
#   docker run -v ./models:/app/models echopose_inference
# Or build with: DOCKER_BUILDKIT=1 docker build --secret id=model,src=models/pose_net.pt .
RUN mkdir -p models

# ── Data directory ───────────────────────────────────────────────────────────
# Session history (SQLite) and any captured CSI data are stored here.
# Mount a volume to persist across container restarts.
RUN mkdir -p data

EXPOSE 8765

# Use curl if available (faster than Python urllib) with fallback
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -sf http://localhost:${INFERENCE_WS_PORT:-8765}/health || \
      python -c "import urllib.request; urllib.request.urlopen('http://localhost:${INFERENCE_WS_PORT:-8765}/health')" || exit 1

# Use shell form so ${INFERENCE_WS_PORT} is expanded at container start time
CMD ["sh", "-c", "gunicorn server:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:${INFERENCE_WS_PORT:-8765}"]
