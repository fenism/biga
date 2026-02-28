#!/bin/bash

# Configuration
APP_NAME="stock_app"
DATA_DIR="/home/$(whoami)/stock_data"

# Create data directory on host if it doesn't exist
mkdir -p $DATA_DIR

echo ">>> Building Docker image..."
docker build -t $APP_NAME .

echo ">>> Stopping existing container..."
docker stop $APP_NAME || true
docker rm $APP_NAME || true

echo ">>> Starting new container..."
docker run -d \
    --name $APP_NAME \
    --restart unless-stopped \
    -p 8501:8501 \
    -v $DATA_DIR:/app/data/market_data \
    $APP_NAME

echo ">>> Deployment complete. Service running on port 8501."
echo ">>> Data persisted in $DATA_DIR"
