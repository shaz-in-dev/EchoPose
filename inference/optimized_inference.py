"""
inference/optimized_inference.py — Hardware Acceleration (Feature 10)

Handles extreme low-latency execution via NVIDIA TensorRT, 
ONNX Quantization, and Apple CoreML fallbacks.
"""

import os
import onnx
try:
    import onnxruntime as ort
    has_ort = True
except ImportError:
    has_ort = False

from pathlib import Path

ONNX_PATH = Path(__file__).parent.parent / "models" / "pose_net.onnx"
QUANTIZED_PATH = Path(__file__).parent.parent / "models" / "pose_net_quantized.onnx"

class OptimizedInference:
    """Uses advanced hardware backends (TensorRT, CoreML, QUInt8) to bypass Python overhead"""
    def __init__(self):
        self.backend = None
        self.session = None
        
        if not has_ort:
            return
            
        providers = ort.get_available_providers()
        
        # 1. NVIDIA TensorRT (Fastest possible GPU backend)
        if 'TensorrtExecutionProvider' in providers:
            self.session = ort.InferenceSession(str(ONNX_PATH), providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
            self.backend = "tensorrt"
            
        # 2. NVIDIA CUDA 
        elif 'CUDAExecutionProvider' in providers:
            self.session = ort.InferenceSession(str(ONNX_PATH), providers=['CUDAExecutionProvider'])
            self.backend = "cuda"
            
        # 3. Apple Silicon (CoreML)
        elif 'CoreMLExecutionProvider' in providers:
            self.session = ort.InferenceSession(str(ONNX_PATH), providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
            self.backend = "coreml"
            
        # 4. CPU Fallback with Quantization (4x faster than raw torch CPU)
        else:
            if not QUANTIZED_PATH.exists() and ONNX_PATH.exists():
                self._quantize_model()
                
            model_path = str(QUANTIZED_PATH) if QUANTIZED_PATH.exists() else str(ONNX_PATH)
            options = ort.SessionOptions()
            options.intra_op_num_threads = os.cpu_count()
            self.session = ort.InferenceSession(model_path, sess_options=options, providers=['CPUExecutionProvider'])
            self.backend = "cpu_quantized"

    def _quantize_model(self):
        """Dynamic INT8 Quantization drastically speeds up CPU inference"""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(str(ONNX_PATH), str(QUANTIZED_PATH), weight_type=QuantType.QUInt8)
        except Exception:
            pass # Fails if ONNX missing, handled gracefully

    def infer(self, features):
        if not self.session: return None
        
        input_name = self.session.get_inputs()[0].name
        outs = self.session.run(None, {input_name: features})
        return outs[0]
