# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile – XM2 OCR Inspection System
# Target: Imago XM2 all-in-one smart camera (Jetson Orin Nano Super, ARM64)
#         Base image: NVIDIA L4T r36.x = JetPack 6.x, CUDA 12.x
# ──────────────────────────────────────────────────────────────────────────────
ARG L4T_VERSION=r36.4.0
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

# Install PaddlePaddle for Jetson (CUDA 12.x / ARM64 / JetPack 6.x)
# Note: Use the Jetson-specific wheel from paddle. The generic PyPI wheel is
#       x86-only. Swap the URL below for the exact JetPack/CUDA version in use.
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install \
        "numpy>=1.24.0" \
        "fastapi>=0.110.0" \
        "uvicorn[standard]>=0.29.0" \
        "pydantic>=2.0.0" \
        "nicegui>=1.4.0" \
        "smbprotocol>=1.12.0" \
        "pytesseract>=0.3.10" \
        "python-multipart>=0.0.9"
# NOTE: opencv-python is intentionally omitted above.  The L4T base image
# ships a CUDA-enabled OpenCV build; installing the PyPI wheel would replace
# it with a CPU-only build and break GPU-accelerated vision pipelines.

# Install PaddlePaddle for Jetson (CUDA 12.6 / cuDNN 9.x / ARM64 / JetPack 6.1).
# The generic PyPI wheel is x86-only and does not support CUDA 12.6/cuDNN 9.x.
# Verify the URL at: https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip_en.html
# and select: Linux · Python 3.10 · CUDA 12.6 · Jetson / aarch64.
RUN pip3 install \
        "https://paddle-inference-lib.bj.bcebos.com/3.0.0/python3.10/Jetson/jetpack6.1_aarch64/paddlepaddle_gpu-3.0.0-cp310-cp310-linux_aarch64.whl" \
        "paddleocr>=2.9.0" || \
    echo "WARNING: paddlepaddle/paddleocr install failed; verify the wheel URL above"

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
