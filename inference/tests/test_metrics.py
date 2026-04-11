"""tests/test_metrics.py — SystemMetrics coverage."""

import pytest
from unittest.mock import patch, MagicMock


def test_record_inference():
    with patch("monitoring.metrics.start_http_server"):
        with patch("monitoring.metrics.psutil") as mock_psutil:
            mock_vm = MagicMock()
            mock_vm.used = 2e9
            mock_psutil.virtual_memory.return_value = mock_vm

            from monitoring.metrics import SystemMetrics
            m = SystemMetrics(port=0)
            m.record_inference(25.0, 0.85)
            # Just verify no crash — Prometheus metrics are internal


def test_record_node_health():
    with patch("monitoring.metrics.start_http_server"):
        with patch("monitoring.metrics.psutil") as mock_psutil:
            mock_vm = MagicMock()
            mock_vm.used = 1e9
            mock_psutil.virtual_memory.return_value = mock_vm

            from monitoring.metrics import SystemMetrics
            m = SystemMetrics(port=0)
            m.record_node_health({"node0": 0.95, "node1": 0.88})


def test_record_drop():
    with patch("monitoring.metrics.start_http_server"):
        with patch("monitoring.metrics.psutil") as mock_psutil:
            mock_vm = MagicMock()
            mock_vm.used = 1e9
            mock_psutil.virtual_memory.return_value = mock_vm

            from monitoring.metrics import SystemMetrics
            m = SystemMetrics(port=0)
            m.record_drop()
