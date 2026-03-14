#!/bin/bash
# deployment/deploy.sh — One-Click Automated Deployment (Feature 13)
# Auto-detects hardware and dynamically optimizes the EchoPose pipeline

echo "🚀 Booting EchoPose Ultimate Production Stack..."

# 1. Auto-detect Hardware
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l 2>/dev/null || echo 0)
CPU_CORES=$(nproc)

echo "🔍 Detected $GPU_COUNT NVIDIA GPUs and $CPU_CORES CPU cores."

# 2. Auto-optimize Backend
if [ "$GPU_COUNT" -gt 0 ]; then
    echo "⚡ Activating TensorRT Multi-GPU Execution Provider [BATCH_SIZE=32]"
    export INFERENCE_BACKEND=tensorrt
    export MAX_BATCH_SIZE=32
    export WORKER_COUNT=$GPU_COUNT
else
    echo "⚡ Activating ONNX QUInt8 CPU Execution Provider [BATCH_SIZE=8]"
    export INFERENCE_BACKEND=onnx_quantized
    export MAX_BATCH_SIZE=8
    # Leave 1 core for the OS
    export WORKER_COUNT=$((CPU_CORES - 1))
fi

# 3. Deploy with Auto-Scaling
echo "🐳 Spinning up $WORKER_COUNT optimized inference workers via Docker Compose..."
docker-compose up -d --scale inference=$WORKER_COUNT

# 4. Wait for Health Checks
echo "⏳ Waiting for services to come online..."
sleep 5
until curl -s http://localhost:8765/health > /dev/null; do
    echo "Waiting for Inference Engine..."
    sleep 2
done

# 5. First-Run Auto-Calibration
echo "🎯 Performing Room Environment Calibration..."
curl -X POST http://localhost:3000/calibrate
echo "✅ Calibration complete. Dashboard is live at http://localhost:8000"
