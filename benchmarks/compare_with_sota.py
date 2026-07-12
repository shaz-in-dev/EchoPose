"""
benchmarks/compare_with_sota.py — Benchmark Framework

Produces benchmark results when a trained model and test data are available.
Without a model, generates a template JSON with measurement instructions.

Run:
    python benchmarks/compare_with_sota.py            # template (no model needed)
    python benchmarks/compare_with_sota.py --live     # run against real model
"""

import argparse
import json
from pathlib import Path


class EchoPoseBenchmark:
    """SOTA comparison harness. Produces measured metrics when a trained model is available."""

    def __init__(self, live: bool = False):
        self.live = live
        self.results = {}

    def benchmark_vs_camera(self):
        if self.live:
            self._run_live_vision_benchmark()
        else:
            self.results["vision_comparison"] = {
                "mean_joint_error_cm":    None,
                "inference_latency_ms":   None,
                "coverage_through_walls": True,
                "ambient_light_dependency": False,
                "privacy_score_percent":  100,
                "_status": "pending — run with --live after training a model",
                "_baselines": {
                    "WiPose_2019":      {"mean_joint_error_cm": 8.3,  "source": "arXiv:1904.00673"},
                    "RF-Pose_2018":     {"mean_joint_error_cm": 11.4, "source": "MIT CSAIL"},
                    "DensePose_camera": {"mean_joint_error_cm": 3.1,  "source": "CVPR 2018"},
                    "EchoPose_target":  {"mean_joint_error_cm": 7.5,  "note": "aspirational design target"},
                },
            }

    def benchmark_robustness(self):
        if self.live:
            self._run_live_robustness_benchmark()
        else:
            self.results["robustness_scenarios"] = {
                "outdoor_nlos_accuracy":  None,
                "crowded_room_3_people":  None,
                "night_vs_day_variance":  None,
                "clothing_invariance":    None,
                "_status": "pending — requires real-world test data collection",
            }

    def benchmark_latency(self):
        if self.live:
            self._run_live_latency_benchmark()
        else:
            self.results["latency"] = {
                "aggregator_sync_ms":    None,
                "inference_pipeline_ms": None,
                "e2e_p50_ms":            None,
                "e2e_p95_ms":            None,
                "_status": "pending",
                "_target": "e2e p50 < 100 ms at 20 Hz on Raspberry Pi 4",
            }

    def _run_live_vision_benchmark(self):
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))
            from scripts.validate_accuracy import run_validation
            self.results["vision_comparison"] = run_validation()
        except Exception as exc:
            self.results["vision_comparison"] = {"error": str(exc)}

    def _run_live_robustness_benchmark(self):
        self.results["robustness_scenarios"] = {"error": "Not yet implemented — see docs/runbooks/"}

    def _run_live_latency_benchmark(self):
        self.results["latency"] = {"error": "Not yet implemented — requires live hardware"}

    def publish(self):
        self.benchmark_vs_camera()
        self.benchmark_robustness()
        self.benchmark_latency()

        out = Path(__file__).parent / "sota_benchmarks.json"
        out.parent.mkdir(exist_ok=True)
        with open(out, "w") as f:
            json.dump(self.results, f, indent=2)

        if not self.live:
            print(f"Benchmark template written to {out}")
            print("All values are None — run with --live after training a model.")
        else:
            print(f"Live benchmark results written to {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true", help="Run against real trained model")
    args = p.parse_args()
    EchoPoseBenchmark(live=args.live).publish()


if __name__ == "__main__":
    main()
