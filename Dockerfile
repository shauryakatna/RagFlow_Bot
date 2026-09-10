FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the placeholder document.
COPY graph.py main.py sample.pdf ./

# GOOGLE_API_KEY and DOC_PATH are provided at runtime (e.g. --env-file .env).
EXPOSE 8000
CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8000"]
