"""
custom_logger.py — Production-Grade JSON Logging (Feature 7)

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
        self.log_dir = Path(__file__).resolve().parent.parent / log_dir
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "structured_inference.jsonl"
        
        # We write directly to a JSON Lines file
        self._file = open(self.log_file, "a", encoding="utf-8")
        atexit.register(self.close)

    def log_inference(self, latency_ms: float, mean_confidence: float, anomalies: list, node_status: dict):
        """JSON log with searchable fields"""
        trace_id = str(uuid.uuid4())

        # CI tests may mock psutil; ensure we always emit a primitive float.
        try:
            used_bytes = getattr(psutil.virtual_memory(), "used", 0.0)
            memory_mb = round(float(used_bytes) / 1e6, 2)
        except Exception:
            memory_mb = 0.0
        
        log_entry = {
            'timestamp': time.time(),
            'trace_id': trace_id,
            'service': 'rf_inference',
            'latency_ms': round(latency_ms, 2),
            'mean_confidence': round(mean_confidence, 4),
            'anomalies_detected': anomalies,
            'node_status': node_status,
            'memory_mb': memory_mb
        }
        
        self._file.write(json.dumps(log_entry) + "\n")
        self._file.flush()
        
    def log_error(self, message: str, exception: str = ""):
        log_entry = {
            'timestamp': time.time(),
            'service': 'rf_inference',
            'level': 'ERROR',
            'message': message,
            'exception': exception
        }
        self._file.write(json.dumps(log_entry) + "\n")
        self._file.flush()

    def close(self):
        if not self._file.closed:
            self._file.close()
