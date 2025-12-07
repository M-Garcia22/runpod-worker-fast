import runpod
from runpod.serverless.utils import rp_upload
import os
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import binascii
import subprocess
import time
import torch

# ============================================
# FAST ENDPOINT - Wan2.2 5B I2V Model
# 2-3x faster than 14B, with Image-to-Video!
# ============================================

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server_address = os.getenv('SERVER_ADDRESS', '127.0.0.1')
client_id = str(uuid.uuid4())

def to_nearest_multiple_of_16(value):
    """Round to nearest multiple of 16, minimum 16"""
    try:
        numeric_value = float(value)
    except Exception:
        raise Exception(f"width/height must be numeric: {value}")
    adjusted = int(round(numeric_value / 16.0) * 16)
    if adjusted < 16:
        adjusted = 16
    return adjusted

def process_input(input_data, temp_dir, output_filename, input_type):
    """Process input data and return file path"""
    if input_type == "path":
        logger.info(f"📁 Path input: {input_data}")
        return input_data
    elif input_type == "url":
        logger.info(f"🌐 URL input: {input_data}")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        return download_file_from_url(input_data, file_path)
    elif input_type == "base64":
        logger.info(f"🔢 Base64 input")
        return save_base64_to_file(input_data, temp_dir, output_filename)
    else:
        raise Exception(f"Unsupported input type: {input_type}")
        
def download_file_from_url(url, output_path):
    """Download file from URL"""
    try:
        result = subprocess.run([
            'wget', '-O', output_path, '--no-verbose', url
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ Downloaded: {url} -> {output_path}")
            return output_path
        else:
            logger.error(f"❌ wget failed: {result.stderr}")
            raise Exception(f"Download failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise Exception("Download timeout")
    except Exception as e:
        raise Exception(f"Download error: {e}")

def save_base64_to_file(base64_data, temp_dir, output_filename):
    """Save base64 data to file"""
    try:
        decoded_data = base64.b64decode(base64_data)
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f:
            f.write(decoded_data)
        logger.info(f"✅ Saved base64 to: {file_path}")
        return file_path
    except (binascii.Error, ValueError) as e:
        raise Exception(f"Base64 decode failed: {e}")
    
def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(url, data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())

def get_videos(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_videos = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break
        else:
            continue

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        videos_output = []
        if 'gifs' in node_output:
            for video in node_output['gifs']:
                with open(video['fullpath'], 'rb') as f:
                    video_data = base64.b64encode(f.read()).decode('utf-8')
                videos_output.append(video_data)
        output_videos[node_id] = videos_output

    return output_videos

def load_workflow(workflow_path):
    with open(workflow_path, 'r') as file:
        return json.load(file)

def handler(job):
    job_input = job.get("input", {})
    logger.info(f"🚀 FAST ENDPOINT (5B I2V) - Received job")

    task_id = f"task_{uuid.uuid4()}"

    # Process image input
    image_path = None
    if "image_path" in job_input:
        image_path = process_input(job_input["image_path"], task_id, "input_image.jpg", "path")
    elif "image_url" in job_input:
        image_path = process_input(job_input["image_url"], task_id, "input_image.jpg", "url")
    elif "image_base64" in job_input:
        image_path = process_input(job_input["image_base64"], task_id, "input_image.jpg", "base64")
    else:
        image_path = "/example_image.png"
        logger.info("Using default image: /example_image.png")

    # Load workflow (1.3B uses single workflow, no FLF2V variant needed)
    prompt = load_workflow("/new_Wan22_api.json")
    
    # ============================================
    # FAST DEFAULTS for 5B TI2V model
    # ============================================
    # Check if Lightning LoRA exists (allows fewer steps)
    lightning_lora_path = "/ComfyUI/models/loras/wan22_5b_lightning.safetensors"
    has_lightning = os.path.exists(lightning_lora_path)
    
    length = job_input.get("length", 65)      # 4 seconds at 16fps
    steps = job_input.get("steps", 6 if has_lightning else 20)  # 6 with Lightning, 20 without
    cfg = job_input.get("cfg", 3.0 if has_lightning else 5.0)   # Lower CFG with Lightning
    
    # Context optimization for speed
    context_frames = job_input.get("context_frames", 65)
    context_overlap = job_input.get("context_overlap", 24)
    context_stride = job_input.get("context_stride", 4)
    
    logger.info(f"🎞️ FAST 5B I2V: {length} frames, {steps} steps, cfg={cfg}")
    logger.info(f"📐 Context: {context_frames}f/{context_overlap}overlap/{context_stride}stride")

    # Apply to workflow
    prompt["244"]["inputs"]["image"] = image_path
    prompt["541"]["inputs"]["num_frames"] = length
    prompt["135"]["inputs"]["positive_prompt"] = job_input.get("prompt", "a person moving")
    prompt["135"]["inputs"]["negative_prompt"] = job_input.get("negative_prompt", 
        "blurry, distorted, low quality, ugly, deformed, static, worst quality")
    
    prompt["220"]["inputs"]["seed"] = job_input.get("seed", int(time.time() * 1000) % (2**32))
    prompt["220"]["inputs"]["cfg"] = cfg
    prompt["220"]["inputs"]["steps"] = steps
    
    # Resolution (16x multiple)
    original_width = job_input.get("width", 480)
    original_height = job_input.get("height", 832)
    adjusted_width = to_nearest_multiple_of_16(original_width)
    adjusted_height = to_nearest_multiple_of_16(original_height)
    
    prompt["235"]["inputs"]["value"] = adjusted_width
    prompt["236"]["inputs"]["value"] = adjusted_height
    
    # Context options
    prompt["498"]["inputs"]["context_frames"] = context_frames
    prompt["498"]["inputs"]["context_overlap"] = context_overlap
    prompt["498"]["inputs"]["context_stride"] = context_stride
    
    # LoRA support
    lora_slot = 0
    
    # Apply Lightning LoRA if available
    if has_lightning:
        prompt["279"]["inputs"]["lora_0"] = "wan22_5b_lightning.safetensors"
        prompt["279"]["inputs"]["strength_0"] = 1.0
        logger.info("⚡ Lightning LoRA enabled - using 6 steps")
        lora_slot = 1
    else:
        logger.info("⚠️ No Lightning LoRA - using 20 steps")
    
    # Apply NSFW LoRA if available and enabled
    nsfw_lora_path = "/ComfyUI/models/loras/wan22_5b_nsfw.safetensors"
    use_nsfw_lora = job_input.get("use_nsfw_lora", True)  # Enabled by default
    if use_nsfw_lora and os.path.exists(nsfw_lora_path):
        nsfw_strength = job_input.get("nsfw_lora_strength", 0.8)
        prompt["279"]["inputs"][f"lora_{lora_slot}"] = "wan22_5b_nsfw.safetensors"
        prompt["279"]["inputs"][f"strength_{lora_slot}"] = nsfw_strength
        logger.info(f"🔥 NSFW LoRA enabled @ {nsfw_strength}")
        lora_slot += 1
    
    # Apply custom LoRAs
    lora_pairs = job_input.get("lora_pairs", [])
    if lora_pairs:
        for i, lora_pair in enumerate(lora_pairs[:4 - lora_slot]):
            lora_name = lora_pair.get("name") or lora_pair.get("high")
            lora_weight = lora_pair.get("weight") or lora_pair.get("high_weight", 1.0)
            if lora_name:
                prompt["279"]["inputs"][f"lora_{i + lora_slot}"] = lora_name
                prompt["279"]["inputs"][f"strength_{i + lora_slot}"] = lora_weight
                logger.info(f"LoRA {i + lora_slot}: {lora_name} @ {lora_weight}")
                
    # Connect to ComfyUI
    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    http_url = f"http://{server_address}:8188/"
    
    # Wait for HTTP
    max_http_attempts = 180
    for http_attempt in range(max_http_attempts):
        try:
            urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"HTTP connected (attempt {http_attempt+1})")
            break
        except Exception as e:
            if http_attempt == max_http_attempts - 1:
                raise Exception("ComfyUI server not reachable")
            time.sleep(1)
    
    ws = websocket.WebSocket()
    max_attempts = 36
    for attempt in range(max_attempts):
        try:
            ws.connect(ws_url)
            logger.info(f"WebSocket connected (attempt {attempt+1})")
            break
        except Exception as e:
            if attempt == max_attempts - 1:
                raise Exception("WebSocket connection timeout")
            time.sleep(5)
    
    start_time = time.time()
    videos = get_videos(ws, prompt)
    generation_time = time.time() - start_time
    
    ws.close()

    logger.info(f"⚡ FAST 5B I2V generation complete in {generation_time:.1f}s")

    for node_id in videos:
        if videos[node_id]:
            return {
                "video": videos[node_id][0],
                "generation_time": generation_time,
                "model": "5B_I2V",
                "frames": length,
                "steps": steps
            }
    
    return {"error": "No video generated"}

runpod.serverless.start({"handler": handler})
