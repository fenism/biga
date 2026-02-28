#!/bin/bash

# Update data daily at 4:30 PM (after market close)
# Add this to crontab: 30 16 * * 1-5 /app/update_data.sh

echo ">>> Starting daily data update..."
docker exec stock_app python3 download_data.py
docker exec stock_app python3 rebuild_cache.py
echo ">>> Data update and cache rebuild finished."
