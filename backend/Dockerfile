FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by yt-dlp and ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency installation
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY backend/ ./backend/

# Expose the API port
EXPOSE 8000

# Start the FastAPI server
CMD ["uv", "run", "uvicorn", "backend.src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
