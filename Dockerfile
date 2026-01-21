# Use a supported Python version (3.11 is stable for TensorFlow)
FROM python:3.11-slim

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy project definition files
COPY pyproject.toml uv.lock* ./

# Install dependencies
# --no-cache keeps the image size down
RUN uv sync --no-cache

# Copy the pipeline model explicitly
COPY data/text_classification_pipeline.pkl data/

# Copy the rest of the application code and data
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Run the application using uv to ensure the environment is correct
CMD ["uv", "run", "uvicorn", "fastapi_service.main:app", "--host", "0.0.0.0", "--port", "8000"]