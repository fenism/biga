# Use a slim Python image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libta-lib0-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install TA-Lib wrapper separately if needed, or ensure requirements.txt covers it
# Note: ta-lib usually needs C library installed, which we did above.

# Copy the rest of the application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Create volume for data persistence
VOLUME ["/app/data/market_data"]

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Entrypoint
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
