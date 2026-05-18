FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${INFERENCE_WS_PORT:-8765}/health')" || exit 1

# Use shell form so ${INFERENCE_WS_PORT} is expanded at container start time
CMD ["sh", "-c", "gunicorn server:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:${INFERENCE_WS_PORT:-8765}"]
