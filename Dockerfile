# Use a slim Python image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C Library from source
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install TA-Lib wrapper separately if needed, or ensure requirements.txt covers it
# Note: ta-lib usually needs C library installed, which we did above.

# Copy the rest of the application
COPY . .

# Expose default port (8080 for Cloud Run, 8501 for local/GCE)
EXPOSE 8080

# Create volume for data persistence (Note: Cloud Run is stateless, volume will be empty on each request unless using Cloud Storage FUSE)
VOLUME ["/app/data/market_data"]

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

# Entrypoint - using sh -c to expand $PORT
ENTRYPOINT ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]
