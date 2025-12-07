#!/bin/bash

# ============================================
# FAST ENDPOINT - CogVideoX-5B I2V Model
# ~2x faster than Wan 14B, proper Image-to-Video!
# ============================================

set -e

echo "============================================"
echo "FAST ENDPOINT - Downloading CogVideoX-5B I2V..."
echo "This will download ~21.6 GB total"
echo "============================================"

# ============================================
# Download from zai-org repository (working mirror)
# https://huggingface.co/zai-org/CogVideoX-5b-I2V
# Using huggingface-cli for proper caching
# ============================================

# Set HuggingFace cache directory
export HF_HOME=/root/.cache/huggingface
mkdir -p $HF_HOME

# Check if model already cached
if [ -d "$HF_HOME/hub/models--zai-org--CogVideoX-5b-I2V" ]; then
    echo "✓ CogVideoX-5B I2V already cached"
else
    echo "📦 Downloading CogVideoX-5B I2V via huggingface-cli..."
    
    # Download full model repo to HuggingFace cache
    # This ensures ComfyUI's from_pretrained() finds it
    python -c "
from huggingface_hub import snapshot_download
import os

print('Downloading zai-org/CogVideoX-5b-I2V...')
snapshot_download(
    repo_id='zai-org/CogVideoX-5b-I2V',
    local_dir_use_symlinks=False,
    resume_download=True
)
print('✓ Download complete!')
"
    
    if [ $? -eq 0 ]; then
        echo "✓ CogVideoX-5B I2V downloaded successfully"
    else
        echo "⚠ HuggingFace download failed, ComfyUI will download on first run"
    fi
fi

echo "============================================"
echo "Model check complete. Starting ComfyUI..."
echo "============================================"

# CUDA Performance Optimizations
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export TORCH_CUDNN_V8_API_ENABLED=1

# Start ComfyUI in the background with optimizations
echo "Starting ComfyUI (CogVideoX FAST mode)..."
python /ComfyUI/main.py --listen --fast &

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to be ready..."
max_wait=300
wait_count=0
while [ $wait_count -lt $max_wait ]; do
    if curl -s http://127.0.0.1:8188/ > /dev/null 2>&1; then
        echo "ComfyUI is ready!"
        break
    fi
    echo "Waiting for ComfyUI... ($wait_count/$max_wait)"
    sleep 2
    wait_count=$((wait_count + 2))
done

if [ $wait_count -ge $max_wait ]; then
    echo "Error: ComfyUI failed to start within $max_wait seconds"
    exit 1
fi

# Start the handler
echo "Starting FAST handler (CogVideoX-5B I2V)..."
exec python handler.py
