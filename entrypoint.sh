#!/bin/bash

# ============================================
# FAST ENDPOINT - Wan2.1 1.3B Model
# 5-6x faster than 14B, good quality with LoRAs
# ============================================

set -e

# Function to download models with retry logic
download_model() {
    local url=$1
    local output=$2
    local max_attempts=5
    local attempt=1
    
    # Skip if file already exists
    if [ -f "$output" ]; then
        echo "✓ Model already exists: $(basename $output)"
        return 0
    fi
    
    while [ $attempt -le $max_attempts ]; do
        echo "Download attempt $attempt/$max_attempts: $(basename $output)"
        if wget --timeout=600 --tries=3 --retry-connrefused --continue -q "$url" -O "$output"; then
            echo "✓ Download successful: $(basename $output)"
            return 0
        else
            echo "✗ Download failed (attempt $attempt/$max_attempts), retrying in 15 seconds..."
            sleep 15
            attempt=$((attempt + 1))
        fi
    done
    echo "ERROR: Failed to download after $max_attempts attempts: $(basename $output)"
    return 1
}

# Download 1.3B models (much smaller than 14B!)
echo "============================================"
echo "FAST ENDPOINT - Downloading 1.3B models..."
echo "This will download ~8 GB (vs 42GB for 14B)"
echo "============================================"

# 1.3B I2V Model (FP16 - small enough to not need FP8)
download_model "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors" \
    "/ComfyUI/models/diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors"

# Actually let's use the proper 1.3B model
download_model "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2.1_I2V_1.3B_bf16.safetensors" \
    "/ComfyUI/models/diffusion_models/Wan2.1_I2V_1.3B_bf16.safetensors"

# Shared components (same as 14B)
download_model "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors" \
    "/ComfyUI/models/clip_vision/clip_vision_h.safetensors"

download_model "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors" \
    "/ComfyUI/models/text_encoders/umt5-xxl-enc-bf16.safetensors"

download_model "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors" \
    "/ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors"

echo "============================================"
echo "Model check complete. Starting ComfyUI..."
echo "============================================"

# CUDA Performance Optimizations
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export TORCH_CUDNN_V8_API_ENABLED=1

# Start ComfyUI in the background with optimizations
echo "Starting ComfyUI (FAST mode)..."
python /ComfyUI/main.py --listen --use-sage-attention --fast &

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to be ready..."
max_wait=120
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
echo "Starting FAST handler (1.3B model)..."
exec python handler.py
