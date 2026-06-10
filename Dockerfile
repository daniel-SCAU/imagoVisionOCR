# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile – XM2 OCR Inspection System
# Target: NVIDIA Jetson (ARM64) with JetPack 5.x / L4T base image
# ──────────────────────────────────────────────────────────────────────────────
ARG L4T_VERSION=r35.4.1
FROM nvcr.io/nvidia/l4t-ml:${L4T_VERSION}-py3 AS base

LABEL maintainer="imago-xm2-ocr" \
      description="XM2 Jetson conveyor OCR inspection system"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PADDLE_ON_GPU=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-glx \
        libgomp1 \
        tesseract-ocr \
        tesseract-ocr-eng \
        cifs-utils \
        nfs-common \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .

# Install PaddlePaddle for Jetson (CUDA 11.x / ARM64)
# Note: Use the Jetson-specific wheel from paddle. The generic PyPI wheel is
#       x86-only. Swap the URL below for the exact JetPack/CUDA version in use.
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install \
        numpy>=1.24.0 \
        "opencv-python>=4.8.0" \
        "fastapi>=0.110.0" \
        "uvicorn[standard]>=0.29.0" \
        "pydantic>=2.0.0" \
        "nicegui>=1.4.0" \
        "smbprotocol>=1.12.0" \
        "pytesseract>=0.3.10" \
        "python-multipart>=0.0.9" && \
    pip3 install paddlepaddle paddleocr || \
        echo "WARNING: paddlepaddle/paddleocr install failed; use Jetson-specific wheel"

# Optional: Ultralytics YOLOv8
# RUN pip3 install ultralytics>=8.0.0

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# Pre-download PaddleOCR models at build time to avoid cold-start delays
RUN python3 -c "from paddleocr import PaddleOCR; PaddleOCR(use_gpu=False, show_log=False)" \
    || echo "WARNING: PaddleOCR model pre-download failed (check network)"

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8000 8080

VOLUME ["/archive", "/config"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/status || exit 1

ENTRYPOINT ["python3", "app/main.py"]
CMD ["--loop", "--api-port", "8000", "--ui-port", "8080"]
