import torch
import hashlib
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from pipeline.pose import PoseNet, FEATURE_SHAPE

def export_to_onnx():
    models_dir = Path(__file__).parent.parent / "models"
    pt_path = models_dir / "pose_net.pt"
    onnx_path = models_dir / "pose_net.onnx"
    models_dir.mkdir(exist_ok=True)

    model = PoseNet()
    if pt_path.exists():
        model.load_state_dict(torch.load(pt_path, map_location="cpu"))
        print(f"Loaded PyTorch checkpoint from {pt_path}")
    else:
        print(f"No checkpoint at {pt_path}; exporting with random weights (for testing only).")

    model.eval()

    nodes, subcarriers, doppler_bins = FEATURE_SHAPE
    dummy_input = torch.randn(1, nodes, subcarriers, doppler_bins)

    print(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["csi_features"],
        output_names=["keypoints"],
        dynamic_axes={"csi_features": {0: "batch_size"}, "keypoints": {0: "batch_size"}},
        dynamo=False,
    )

    sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    print(f"Successfully exported PoseNet to ONNX format!")
    print(f"SHA-256: {sha}")
    print(f"Set EXPECTED_ONNX_HASH={sha} in .env for integrity verification.")

if __name__ == "__main__":
    export_to_onnx()
