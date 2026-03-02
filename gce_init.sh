#!/bin/bash

# Configuration
DATA_DIR="/Volumes/stock_data" # Change this if you have a separate persistent disk
CONTAINER_NAME="stock_app"

echo ">>> Starting GCE Initialization / Data Recovery <<<"

# 1. Pull latest code
echo ">>> Pulling latest code from GitHub..."
git pull origin main

# 2. Check if persistent data mount exists (Optional, depends on GCE setup)
if [ ! -d "$DATA_DIR" ]; then
    echo ">>> Note: Persistent disk $DATA_DIR not found. Using local directory /app/data."
    DATA_DIR="$(pwd)/data"
fi
mkdir -p "$DATA_DIR/market_data"
mkdir -p "$DATA_DIR/signal_cache"

# 3. Build and Run Container
# We map the host directory to the container volume for persistence
echo ">>> Rebuilding and restarting container..."
docker stop $CONTAINER_NAME || true
docker rm $CONTAINER_NAME || true
docker build -t stock_app .
docker run -d --name $CONTAINER_NAME \
    -p 8080:8080 \
    -v "$DATA_DIR:/app/data" \
    --restart always \
    stock_app

# 4. Trigger Data Download
echo ">>> Triggering initial data download (This will take time)..."
docker exec $CONTAINER_NAME python3 download_data.py

# 5. Build Cache
echo ">>> Building signal cache..."
docker exec $CONTAINER_NAME python3 rebuild_cache.py

echo ">>> Setup complete! Application should be available on port 8080."
echo ">>> Note: You may need to refresh the UI once download finishes."
