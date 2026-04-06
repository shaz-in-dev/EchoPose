"""
inference/gpu_server.py — Multi-GPU DataParallel Inference (Feature 9)

Scales the PoseNetV2 model across an arbitrary number of CUDA GPUs
for maximum batch throughput in enterprise deployments.
"""

import torch
import torch.nn as nn
import asyncio
import numpy as np
from pipeline.pose_net_v2 import PoseNetV2

class DistributedInference:
    """Scales inference seamlessly to multiple physical GPUs"""
    
    def __init__(self):
        self.device_count = torch.cuda.device_count()
        self.has_gpus = self.device_count > 0
        
        if self.has_gpus:
            # Data parallelism across available GPUs
            model = PoseNetV2().cuda()
            self.pose_net = nn.DataParallel(model, device_ids=list(range(self.device_count)))
            self.pose_net.eval()
        else:
            self.pose_net = PoseNetV2().eval()
            
    def _create_batches(self, bundles: list, max_batch_size: int = 32):
        """Chunk incoming websocket bundles into parallelizable GPU tensors"""
        # ... logic to pad and stack variable length bundles into [B, N, S, D]
        batches = [bundles[i:i + max_batch_size] for i in range(0, len(bundles), max_batch_size)]
        out = []
        for batch in batches:
            # Build contiguous float32 arrays first to avoid slow tensor-from-list conversions.
            batch_np = np.asarray(batch, dtype=np.float32)
            out.append(torch.from_numpy(batch_np))
        return out
        
    async def _infer_batch(self, batch_tensor: torch.Tensor):
        loop = asyncio.get_event_loop()

        def _forward():
            with torch.no_grad():
                if self.has_gpus:
                    batch_tensor_dev = batch_tensor.cuda(non_blocking=True)
                else:
                    batch_tensor_dev = batch_tensor
                poses = self.pose_net(batch_tensor_dev)
                return poses.cpu().numpy()

        return await loop.run_in_executor(None, _forward)

    async def batch_inference(self, feature_bundles: list):
        """Process multiple UI clients concurrently across GPUs"""
        batches = self._create_batches(feature_bundles)
        
        # Dispatch to PyTorch async background threads to saturate GPUs
        tasks = [self._infer_batch(b) for b in batches]
        results = await asyncio.gather(*tasks)
        
        # Flatten and return
        return [pose for batch in results for pose in batch]
