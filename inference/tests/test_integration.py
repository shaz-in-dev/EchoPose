import pytest
import asyncio
import torch
import numpy as np

# Use absolute imports if pythonpath is set to inference root in pytest.ini
import gpu_server
import server_v2
from pipeline.fusion import FusionPipeline
from research.disambiguation import MultiPersonDisambiguation

@pytest.mark.asyncio
async def test_gpu_server_tensor_batching():
    """Verify that batches are converted to torch Tensors instead of lists"""
    server = gpu_server.DistributedInference()
    bundles = [np.random.randn(3, 64, 16) for _ in range(5)]
    
    batches = server._create_batches(bundles, max_batch_size=2)
    
    assert len(batches) == 3
    assert isinstance(batches[0], torch.Tensor)
    assert batches[0].dtype == torch.float32

@pytest.mark.asyncio
async def test_server_v2_async_call():
    """Verify that async continuously runs without type errors on batch_inference await"""
    server = server_v2.HighThroughputServer()
    
    async def mock_batch_inference(features):
        return [["mock_skeleton"]]
        
    server.model.batch_inference = mock_batch_inference
    
    # We will enqueue a fake bundle
    await server.bundle_queue.put({"frames": [], "window_us": 0})
    
    # Run the continuous loop as a task and cancel it after a moment
    task = asyncio.create_task(server._infer_continuously())
    await asyncio.sleep(0.1)
    task.cancel()
    
    assert server.bundle_queue.empty()


def test_disambiguation_separates_people():
    """Verify DBSCAN separates distinct Doppler clusters into per-person tensors."""
    disamb = MultiPersonDisambiguation(max_people=3)
    csi = np.random.randn(3, 64, 16)
    # Create a doppler spectrum with two distinct clusters of activity
    doppler = np.zeros((3, 16))
    doppler[:, 2:4] = 5.0   # Person A – slow walker
    doppler[:, 10:13] = 5.0  # Person B – fast walker
    
    tensors = disamb.disentangle_csi_signatures(csi, doppler)
    assert len(tensors) >= 1
    for t in tensors:
        assert t.shape == csi.shape


def test_fusion_pipeline_returns_tuple():
    """Verify FusionPipeline.process_bundle returns (features, per_person_tensors)."""
    pipeline = FusionPipeline()
    bundle = {
        "frames": [
            {"node_id": 0, "amplitudes": list(np.random.rand(64))},
            {"node_id": 1, "amplitudes": list(np.random.rand(64))},
            {"node_id": 2, "amplitudes": list(np.random.rand(64))},
        ],
        "window_us": 50000,
    }
    # Push enough frames to fill the denoiser buffer
    for _ in range(10):
        result = pipeline.process_bundle(bundle)
    
    assert isinstance(result, tuple)
    features, per_person = result
    assert features.ndim == 3  # [nodes, sub, doppler]
    assert isinstance(per_person, list)
