"""
monitoring/metrics.py — Real-time system health dashboard (Feature 6)

Exposes inference latency, confidence scores, memory usage, and node health 
to a Prometheus endpoint for Grafana dashboards.
"""

from prometheus_client import Histogram, Gauge, Counter, start_http_server
import psutil
import time

class SystemMetrics:
    def __init__(self, port=9090):
        # Prevent double registration issues during testing/reloads
        import prometheus_client
        prometheus_client.REGISTRY = prometheus_client.CollectorRegistry(auto_describe=True)
        
        self.latency_ms = Histogram(
            'echo_pose_inference_latency_ms', 
            'End-to-end inference latency in milliseconds'
        )
        self.confidence_score = Gauge(
            'echo_pose_inference_confidence', 
            'Average skeleton confidence score'
        )
        self.node_health = Gauge(
            'echo_pose_node_health', 
            'Node health percentage', 
            ['node_id']
        )
        self.frame_drops = Counter(
            'echo_pose_frame_drop_total', 
            'Total skipped or corrupted frames'
        )
        self.memory_mb = Gauge(
            'echo_pose_memory_usage_mb', 
            'System memory usage in MB'
        )
        
        # Start Prometheus scraping server in background
        try:
            start_http_server(port)
        except OSError:
            pass # Already running in this process/port

    def record_inference(self, latency: float, confidence: float):
        self.latency_ms.observe(latency)
        self.confidence_score.set(confidence)
        self.memory_mb.set(psutil.virtual_memory().used / 1e6)
        
    def record_node_health(self, health_dict: dict):
        for node_id, health in health_dict.items():
            self.node_health.labels(node_id=str(node_id)).set(health * 100)
            
    def record_drop(self):
        self.frame_drops.inc()
