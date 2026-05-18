"""
logging/structured_logger.py — Production-Grade JSON Logging (Feature 7)

Creates searchable logs suitable for ElasticSearch / Logstash / Kibana.
"""

import json
import logging
import time
import uuid
import atexit
import psutil
from pathlib import Path

class StructuredLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(__file__).resolve().parent.parent.parent / log_dir
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "structured_inference.jsonl"

        # We write directly to a JSON Lines file
        self._file = open(self.log_file, "a", encoding="utf-8")
        # Ensure the file is flushed and closed on interpreter shutdown
        atexit.register(self.close)
        # M15: timer-based flush to avoid syscall overhead on every write
        self._last_flush = time.monotonic()
        self._flush_interval = 1.0  # seconds

    def log_inference(self, latency_ms: float, mean_confidence: float, anomalies: list, node_status: dict):
        """JSON log with searchable fields"""
        trace_id = str(uuid.uuid4())
        
        log_entry = {
            'timestamp': time.time(),
            'trace_id': trace_id,
            'service': 'rf_inference',
            'latency_ms': round(latency_ms, 2),
            'mean_confidence': round(mean_confidence, 4),
            'anomalies_detected': anomalies,
            'node_status': node_status,
            'memory_mb': round(psutil.virtual_memory().used / 1e6, 2)
        }
        
        self._file.write(json.dumps(log_entry) + "\n")
        # M15: flush at most once per second instead of on every write
        now = time.monotonic()
        if now - self._last_flush >= self._flush_interval:
            self._file.flush()
            self._last_flush = now

    def log_error(self, message: str, exception: str = ""):
        log_entry = {
            'timestamp': time.time(),
            'service': 'rf_inference',
            'level': 'ERROR',
            'message': message,
            'exception': exception
        }
        self._file.write(json.dumps(log_entry) + "\n")
        # M15: flush at most once per second instead of on every write
        now = time.monotonic()
        if now - self._last_flush >= self._flush_interval:
            self._file.flush()
            self._last_flush = now

    def close(self):
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
