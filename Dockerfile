# ============================================
# FAST ENDPOINT - Wan2.1 1.3B Model
# 5-6x faster than 14B, lower VRAM requirements
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
    
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/city96/ComfyUI-GGUF && \
    cd ComfyUI-GGUF && \
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
    git clone https://github.com/kael558/ComfyUI-GGUF-FantasyTalking && \
    cd ComfyUI-GGUF-FantasyTalking && \
    pip install -r requirements.txt
    
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/orssorbit/ComfyUI-wanBlockswap

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper && \
    cd ComfyUI-WanVideoWrapper && \
    pip install -r requirements.txt

# ============================================
# SPEED OPTIMIZATIONS (added by OpenMind)
# ============================================

# TeaCache - 40-60% faster inference by caching redundant computations
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/welltop-cn/ComfyUI-TeaCache || true

# Torch compile optimization (enable in environment)
ENV TORCH_COMPILE_ENABLED=1
ENV TORCH_COMPILE_MODE=reduce-overhead

# CUDA optimization flags
ENV CUDA_MODULE_LOADING=LAZY
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/eddyhhlure1Eddy/IntelligentVRAMNode && \
    git clone https://github.com/eddyhhlure1Eddy/auto_wan2.2animate_freamtowindow_server && \
    git clone https://github.com/eddyhhlure1Eddy/ComfyUI-AdaptiveWindowSize && \
    cd ComfyUI-AdaptiveWindowSize/ComfyUI-AdaptiveWindowSize && \
    mv * ../

# Create model directories (models will be downloaded at runtime to reduce image size)
RUN mkdir -p /ComfyUI/models/diffusion_models && \
    mkdir -p /ComfyUI/models/loras && \
    mkdir -p /ComfyUI/models/clip_vision && \
    mkdir -p /ComfyUI/models/text_encoders && \
    mkdir -p /ComfyUI/models/vae

COPY . .
COPY extra_model_paths.yaml /ComfyUI/extra_model_paths.yaml
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]