FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre‑download models from Hugging Face during the image build
# They will be stored inside the image permanently
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download('PrudhviManikanta/skin-disease-efficientnet-multiclass', 'skin_disease_efficientnet_ft.keras'); \
    hf_hub_download('PrudhviManikanta/skin-disease-efficientnet-multiclass', 'model.keras')"

# Copy the application code
COPY main.py .

# Expose port (Render provides $PORT)
EXPOSE 8000

# Start command
CMD uvicorn main:app --host 0.0.0.0 --port $PORT