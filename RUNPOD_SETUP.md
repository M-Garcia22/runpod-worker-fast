# RunPod Serverless Endpoint Setup Guide

This guide will walk you through setting up your Wan2.2 video generation worker as a RunPod Serverless endpoint.

## Prerequisites

- RunPod account (sign up at https://runpod.io)
- GitHub repository with your code (or use RunPod's template)
- RunPod API key (for testing the endpoint)

## Step-by-Step Setup

### 1. Push Your Code to GitHub

Make sure your `runpod-worker` directory is in a GitHub repository:

```bash
cd runpod-worker
git init  # if not already a git repo
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 2. Create Serverless Endpoint on RunPod

1. **Go to RunPod Console**: https://www.runpod.io/console/serverless
2. **Click "New Endpoint"** or **"Create Endpoint"**
3. **Fill in the details**:

   - **Template**: Select "Docker Hub" or "GitHub"
   - **Repository**: 
     - If using GitHub: `your-username/your-repo` (e.g., `mgarcia8324/openmind`)
     - If using Docker Hub: `mgarcia8324/openmind-wan22:optimized`
   - **Dockerfile Path**: `runpod-worker/Dockerfile` (if repo is not in root)
   - **Handler Path**: `handler.py`
   - **Container Disk**: **At least 100 GB** (models are ~42 GB + base image)

### 3. Configure Endpoint Settings

**Important Settings:**

- **Container Disk**: **100-150 GB** (required for ~42 GB models)
- **GPU Type**: Select based on your needs:
  - **RTX 3090 / A5000**: Good balance
  - **RTX 4090 / A6000**: Faster inference
  - **A100**: Best performance
- **Max Workers**: Start with 1-2, increase as needed
- **Idle Timeout**: 5-10 minutes (models load on cold start)
- **Flashboot**: Enable for faster cold starts

### 4. Network Volume (Recommended for Model Persistence)

**Important**: Models download on first run (~42 GB). To avoid re-downloading on every cold start:

1. **Create a Network Volume**:
   - Go to "Network Volumes" in RunPod console
   - Create a new volume (S3-backed, 100+ GB)
   - Note the volume path (e.g., `/runpod-volume`)

2. **Mount Volume to Endpoint**:
   - In endpoint settings, add the network volume
   - Mount point: `/ComfyUI/models`
   - This will persist models between cold starts

3. **Update entrypoint.sh** (Optional - for volume mounting):
   - Models will be downloaded to the volume on first run
   - Subsequent cold starts will use cached models

### 5. Environment Variables (Optional)

Add any environment variables your handler needs:
- `TORCH_COMPILE_ENABLED=1`
- `TORCH_COMPILE_MODE=reduce-overhead`
- `CUDA_MODULE_LOADING=LAZY`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

(These are already set in the Dockerfile, but you can override them here)

### 6. Build and Deploy

1. **Click "Deploy"** or **"Save"**
2. RunPod will:
   - Clone your repository
   - Build the Docker image (takes 10-20 minutes)
   - Push to RunPod's registry
   - Deploy the endpoint

3. **Monitor the build** in the endpoint logs
4. **Wait for "Active" status**

### 7. Test Your Endpoint

Once active, test with a simple request:

```python
import runpod

# Your endpoint ID (found in RunPod console)
endpoint_id = "your-endpoint-id-here"

# Test input
test_input = {
    "input": {
        "prompt": "a cat walking",
        "image_url": "https://example.com/test-image.jpg",
        "width": 480,
        "height": 832,
        "length": 81,
        "steps": 10
    }
}

# Submit job
job = runpod.submit_job(endpoint_id, test_input)
print(f"Job ID: {job['id']}")

# Wait for result
result = runpod.wait_for_job(job['id'])
print(result)
```

Or use curl:

```bash
curl -X POST https://api.runpod.io/v2/your-endpoint-id/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "input": {
      "prompt": "a cat walking",
      "image_url": "https://example.com/test-image.jpg",
      "width": 480,
      "height": 832,
      "length": 81,
      "steps": 10
    }
  }'
```

## Important Notes

### First Cold Start

- **First request will be slow** (~5-10 minutes):
  - Downloads ~42 GB of models
  - Loads models into GPU memory
  - Subsequent requests will be faster

### Model Download Behavior

- Models download automatically via `entrypoint.sh`
- If using a **Network Volume**: Models persist between cold starts
- If **no volume**: Models re-download on each cold start (slow!)

### Cost Optimization

1. **Use Network Volume**: Avoids re-downloading models
2. **Set appropriate idle timeout**: Balance between cost and responsiveness
3. **Monitor usage**: Check RunPod dashboard for costs
4. **Use Flashboot**: Reduces cold start time

### Troubleshooting

**Build fails:**
- Check Dockerfile syntax
- Ensure all dependencies are in requirements.txt
- Check build logs in RunPod console

**Endpoint times out:**
- Increase idle timeout
- Check if models are downloading (first run takes time)
- Verify container disk is large enough (100+ GB)

**Models not persisting:**
- Ensure Network Volume is mounted at `/ComfyUI/models`
- Check volume has enough space
- Verify volume is connected to endpoint

**Out of memory errors:**
- Use smaller GPU or reduce batch size
- Check model loading in logs
- Verify GPU type supports the models

## Next Steps

1. **Test with your client code**: Use `generate_video_client.py`
2. **Monitor performance**: Check RunPod metrics
3. **Optimize settings**: Adjust based on usage patterns
4. **Scale as needed**: Increase workers for higher throughput

## Support

- RunPod Docs: https://docs.runpod.io
- RunPod Discord: https://discord.gg/runpod
- Check endpoint logs in RunPod console for detailed error messages

