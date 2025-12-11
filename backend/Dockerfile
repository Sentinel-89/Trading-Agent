# Use the Python 3.14 base image
FROM python:3.13-slim

# Install system dependencies needed for compilation
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

# --- INSTALL TA-LIB FROM SOURCE (Necessary for pandas-ta) ---
RUN wget https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz -O /tmp/ta-lib.tar.gz && \
    mkdir -p /usr/local/src/ta-lib && \
    tar -xzf /tmp/ta-lib.tar.gz -C /usr/local/src/ta-lib --strip-components 1 && \
    cd /usr/local/src/ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    rm -f /tmp/ta-lib.tar.gz

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY Requirements.txt .

# --- INSTALL PYTHON DEPENDENCIES ---

# Step 1: Install PyTorch and related packages (needs special index)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install remaining application dependencies (where pandas-ta lives)
# This will likely fail, but we must try this explicit step to see the error.
RUN pip install --no-cache-dir -r Requirements.txt 

# Copy the rest of the application code (main.py, models.py, etc.)
COPY . .

# Expose the port that Uvicorn will listen on
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]