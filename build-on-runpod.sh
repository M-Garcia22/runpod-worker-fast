#!/bin/bash
# Build script to run on RunPod or any cloud instance with Docker

set -e

echo "=== Building Docker image on remote machine ==="
echo "This script should be run on a machine with sufficient disk space (50+ GB free)"

# Check Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Check available disk space (requires at least 50 GB)
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 50 ]; then
    echo "Warning: Less than 50 GB available. Build may fail."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build and push
echo "Building image..."
docker buildx build \
    --platform linux/amd64 \
    -t mgarcia8324/openmind-wan22:optimized \
    --push \
    .

echo "✓ Build complete and pushed to registry!"

