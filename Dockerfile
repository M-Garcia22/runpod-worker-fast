# ============================================
# FAST ENDPOINT - CogVideoX-5B I2V Model
# ~2x faster than Wan 14B, proper Image-to-Video!
# VRAM: ~20GB (fits RTX 4090 / A10)
# ============================================
FROM wlsdml1114/multitalk-base:1.7 AS runtime

RUN pip install -U "huggingface_hub[hf_transfer]"
RUN pip install runpod websocket-client

WORKDIR /

RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd /ComfyUI && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && \
    pip install -r requirements.txt

# CogVideoX Wrapper - Main extension for CogVideoX models
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-CogVideoXWrapper && \
    cd ComfyUI-CogVideoXWrapper && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    cd ComfyUI-KJNodes && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    cd ComfyUI-VideoHelperSuite && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/city96/ComfyUI-GGUF && \
    cd ComfyUI-GGUF && \
    pip install -r requirements.txt

# ============================================
# SPEED OPTIMIZATIONS (added by OpenMind)
# ============================================

# Torch compile optimization (enable in environment)
ENV TORCH_COMPILE_ENABLED=1
ENV TORCH_COMPILE_MODE=reduce-overhead

# CUDA optimization flags
ENV CUDA_MODULE_LOADING=LAZY
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create model directories
RUN mkdir -p /ComfyUI/models/CogVideo && \
    mkdir -p /ComfyUI/models/diffusion_models && \
    mkdir -p /ComfyUI/models/loras && \
    mkdir -p /ComfyUI/models/clip_vision && \
    mkdir -p /ComfyUI/models/text_encoders && \
    mkdir -p /ComfyUI/models/vae

COPY . .
COPY extra_model_paths.yaml /ComfyUI/extra_model_paths.yaml
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
