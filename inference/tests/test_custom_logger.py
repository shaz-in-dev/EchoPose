"""tests/test_custom_logger.py — StructuredLogger coverage."""

import pytest
import json
import os
import tempfile
from unittest.mock import patch, MagicMock


def test_structured_logger_writes_json():
    # Mock psutil to avoid dependency issues
    with patch.dict("sys.modules", {"psutil": MagicMock()}):
        import importlib
        import custom_logger
        importlib.reload(custom_logger)

        with tempfile.TemporaryDirectory() as td:
            logger = custom_logger.StructuredLogger.__new__(custom_logger.StructuredLogger)
            from pathlib import Path
            logger.log_dir = Path(td)
            logger.log_file = Path(td) / "test.jsonl"
            logger._file = open(logger.log_file, "a", encoding="utf-8")

            # Mock psutil.virtual_memory for log_inference
            mock_vm = MagicMock()
            mock_vm.used = 1e9
            with patch("psutil.virtual_memory", return_value=mock_vm):
                logger.log_inference(
                    latency_ms=15.3,
                    mean_confidence=0.87,
                    anomalies=["jitter"],
                    node_status={"node0": "ok"},
                )

            logger.log_error("test error", "ValueError")
            logger.close()

            lines = logger.log_file.read_text().strip().split("\n")
            assert len(lines) == 2

            entry = json.loads(lines[0])
            assert entry["latency_ms"] == 15.3
            assert entry["mean_confidence"] == 0.87
            assert "trace_id" in entry

            err = json.loads(lines[1])
            assert err["level"] == "ERROR"
            assert err["message"] == "test error"


def test_close_idempotent():
    with patch.dict("sys.modules", {"psutil": MagicMock()}):
        import importlib
        import custom_logger
        importlib.reload(custom_logger)

        with tempfile.TemporaryDirectory() as td:
            logger = custom_logger.StructuredLogger.__new__(custom_logger.StructuredLogger)
            from pathlib import Path
            logger.log_dir = Path(td)
            logger.log_file = Path(td) / "test.jsonl"
            logger._file = open(logger.log_file, "a", encoding="utf-8")
            logger.close()
            logger.close()  # should not raise
