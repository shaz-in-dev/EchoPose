import torch
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from pipeline.pose import PoseNet, FEATURE_SHAPE

def export_to_onnx():
    models_dir = Path(__file__).parent.parent / "models"
    pt_path = models_dir / "pose_net.pt"
    onnx_path = models_dir / "pose_net.onnx"

    if not pt_path.exists():
        print(f"❌ Cannot find PyTorch model at {pt_path}")
        print("Please run scripts/download_weights.py first.")
        return

    print("Loading PyTorch model...")
    model = PoseNet()
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()

    # Create dummy input based on our FEATURE_SHAPE (B, N, S, D)
    # B=1 (batch size), N=3 (nodes), S=64 (subcarriers), D=16 (doppler_bins)
    nodes, subcarriers, doppler_bins = FEATURE_SHAPE
    dummy_input = torch.randn(1, nodes, subcarriers, doppler_bins)

    print(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["csi_features"],
        output_names=["keypoints"],
        dynamic_axes={"csi_features": {0: "batch_size"}, "keypoints": {0: "batch_size"}}
    )
    print("✅ Successfully exported PoseNet to ONNX format!")

if __name__ == "__main__":
    export_to_onnx()
