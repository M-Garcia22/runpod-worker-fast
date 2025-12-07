import runpod
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
# FAST ENDPOINT - CogVideoX-5B I2V Model
# ~2x faster than Wan 14B, proper Image-to-Video!
# Uses THUDM/CogVideoX-5b-I2V from HuggingFace
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
    logger.info(f"🚀 FAST ENDPOINT (CogVideoX-5B I2V) - Received job")

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

    # Load CogVideoX workflow
    prompt = load_workflow("/new_CogVideoX_api.json")
    
    # ============================================
    # CogVideoX-5B I2V Settings
    # From https://huggingface.co/THUDM/CogVideoX-5b-I2V
    # - Resolution: 720x480 (fixed!)
    # - Frames: 81 (2.7 seconds at 30fps)
    # - Steps: 50 recommended
    # ============================================
    num_frames = job_input.get("length", 81)      # 81 frames = 2.7 seconds at 30fps
    steps = job_input.get("steps", 50)            # CogVideoX needs 50 steps
    cfg = job_input.get("cfg", 6.0)               # CFG scale
    seed = job_input.get("seed", int(time.time() * 1000) % (2**32))
    
    logger.info(f"🎞️ CogVideoX-5B I2V: {num_frames} frames @ 30fps, {steps} steps, cfg={cfg}, seed={seed}")

    # Apply to workflow
    # Node 5: Load Image
    prompt["5"]["inputs"]["image"] = image_path
    
    # Node 3: Positive prompt
    prompt["3"]["inputs"]["prompt"] = job_input.get("prompt", "a person moving naturally")
    
    # Node 4: Negative prompt
    prompt["4"]["inputs"]["prompt"] = job_input.get("negative_prompt", 
        "blurry, distorted, low quality, static, ugly, deformed, worst quality")
    
    # Node 6: Resolution (CogVideoX is fixed at 720x480!)
    # We still resize input image but output is always 720x480
    prompt["6"]["inputs"]["width"] = 720
    prompt["6"]["inputs"]["height"] = 480
    
    # Node 8: Sampler settings
    prompt["8"]["inputs"]["num_frames"] = num_frames
    prompt["8"]["inputs"]["steps"] = steps
    prompt["8"]["inputs"]["cfg"] = cfg
    prompt["8"]["inputs"]["seed"] = seed

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

    logger.info(f"⚡ CogVideoX-5B I2V generation complete in {generation_time:.1f}s")

    for node_id in videos:
        if videos[node_id]:
            return {
                "video": videos[node_id][0],
                "generation_time": generation_time,
                "model": "CogVideoX-5B-I2V",
                "frames": num_frames,
                "fps": 30,
                "duration_seconds": round(num_frames / 30, 2),
                "steps": steps,
                "resolution": "720x480"
            }
    
    return {"error": "No video generated"}

runpod.serverless.start({"handler": handler})
