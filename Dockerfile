FROM python:3.11-slim

WORKDIR /app

# Install minimal OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy full application codebase
COPY . .

# Expose ports for FastAPI (8000) and Streamlit Dashboard (8501)
EXPOSE 8000
EXPOSE 8501

# Runtime environment settings
ENV PYTHONUNBUFFERED=1
ENV OFFLINE_MODE=1

# Default execution starts the self-contained demo environment
CMD ["python", "run_demo.py", "--offline"]
