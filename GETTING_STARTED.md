# Getting Started – imagoVisionOCR on the Imago XM2

This guide walks through setting up the imagoVisionOCR software on the **Imago XM2** smart camera from scratch. The XM2 is an all-in-one unit: the camera sensor and a **NVIDIA Jetson Orin Nano Super** edge computer are housed in the same industrial enclosure. Everything in this guide runs directly on the device — no external PC or server is required.

---

## What you need

| Item | Notes |
|---|---|
| Imago XM2 smart camera | Shipped with JetPack 6.x and `xm2sdk` pre-installed |
| Ethernet or Wi-Fi connection (initial setup only) | For SSH access and cloning the repository; not needed at runtime |
| USB drive or SSD (optional) | For the local inspection archive if internal storage is limited |
| Network-attached storage (optional) | For NAS sync; see [NAS sync](#optional-nas-sync) below |

---

## Step 1 — Power on and connect

1. Mount the XM2 in position and connect power (see Imago hardware guide).
2. Connect an Ethernet cable to your network switch, or configure Wi-Fi using the Imago setup wizard on first boot.
3. Find the device's IP address (printed on the label, or via your router's DHCP table).
4. Open an SSH session:

```bash
ssh jetson@<XM2_IP_ADDRESS>
# Default password is set during Imago first-boot wizard
```

All remaining steps are run inside this SSH session (or a terminal directly on the device).

---

## Step 2 — Verify JetPack version

The software requires **JetPack 6.x (L4T r36.x)**. Confirm this on the XM2:

```bash
cat /etc/nv_tegra_release
# Expected output contains: R36 (release), REVISION: 4.0 (or higher)
```

Also confirm the vendor SDK is present:

```bash
python3 -c "import xm2sdk; print('xm2sdk OK')"
# Expected: xm2sdk OK
```

If `xm2sdk` is not found, contact Imago support — it is pre-installed at the factory and should not need manual installation.

---

## Step 3 — Clone the repository

```bash
cd /opt
sudo mkdir xm2-ocr
sudo chown jetson:jetson xm2-ocr
cd xm2-ocr

git clone https://github.com/daniel-SCAU/imagoVisionOCR .
```

---

## Step 4 — Install Python dependencies

```bash
cd /opt/xm2-ocr
pip3 install -r requirements.txt
```

> **Note on PaddlePaddle:** The generic PyPI `paddlepaddle` wheel is x86-only. On the XM2 (ARM64), install the Jetson-specific wheel from Paddle's release page or NVIDIA's JetPack ecosystem. If the standard install fails, the application will automatically fall back to Tesseract OCR.
>
> For the best GPU-accelerated OCR performance, install the correct ARM64 wheel:
> ```bash
> # Example – replace the URL with the wheel matching your JetPack version
> pip3 install https://paddle-wheel.bj.bcebos.com/2.6.0/jetson/jetpack6/paddlepaddle_gpu-2.6.0-cp310-cp310-linux_aarch64.whl
> ```

---

## Step 5 — Prepare your detection model

The XM2 runs object detection **entirely on-device** (no cloud API). You need either a Roboflow local model or YOLO weights file.

### Option A — Roboflow local model (recommended)

1. Download your trained model from Roboflow (export as "Roboflow Inference" format).
2. Copy the model directory to the XM2:

```bash
# From your development machine
scp -r ./my_model jetson@<XM2_IP>:/opt/xm2-ocr/models/
```

3. Set `roboflow_model_path` in your config (Step 6).

### Option B — YOLOv8 local weights

1. Copy your `.pt` weights file to the XM2:

```bash
scp yolov8n.pt jetson@<XM2_IP>:/opt/xm2-ocr/models/
```

2. In config (Step 6), set `detection_model` to `"yolo"` and `yolo_weights` to the path.

---

## Step 6 — Configure the application

```bash
cd /opt/xm2-ocr
cp config.example.json config/config.json
nano config/config.json
```

Minimum configuration for the XM2 with Roboflow local model:

```json
{
  "use_xm2_sdk": true,
  "camera_exposure": 2000,
  "trigger_mode": false,

  "detection_model": "roboflow",
  "roboflow_model_path": "/opt/xm2-ocr/models/my_model",

  "ocr_engine": "paddleocr",
  "ocr_use_gpu": true,

  "validation_rules": "{\"lot_number\": \"LOT\\\\d{6}[A-Z]\"}",

  "storage_path": "/opt/xm2-ocr/archive",
  "nas_enabled": false
}
```

Key settings explained:

| Key | XM2 recommendation |
|---|---|
| `use_xm2_sdk` | **Always `true`** on the XM2 — uses the integrated Imago camera |
| `camera_exposure` | Start at `2000` µs; tune based on your lighting |
| `trigger_mode` | Set `true` if using hardware trigger input; `false` for software/timed loop |
| `ocr_use_gpu` | `true` — the Orin Nano Super has a dedicated GPU for inference |
| `storage_path` | Use `/opt/xm2-ocr/archive` or mount a USB drive at `/mnt/usb` |

---

## Step 7 — Run a test capture

Before enabling auto-start, verify the pipeline works end-to-end:

```bash
cd /opt/xm2-ocr
python3 app/main.py --config config/config.json --once
```

Expected output:
```
2024-06-15T14:30:22 INFO     app.camera.camera_interface – Camera connected (sdk=True)
2024-06-15T14:30:22 INFO     app.ocr.reader – PaddleOCR loaded (lang=en, gpu=True)
2024-06-15T14:30:23 INFO     app.pipeline – OCR text: 'LOT240615A'
2024-06-15T14:30:23 INFO     app.pipeline – Validation: PASS – All rules matched
[2024-06-15T14:30:23+00:00] PASS – 'LOT240615A'
```

Exit code `0` = PASS, `1` = FAIL, `2` = pipeline error.

---

## Step 8 — Access the web UI

Start the application with both API and UI:

```bash
python3 app/main.py --config config/config.json --loop --interval 1.0
```

Then open a browser on any machine on the same network:

| Interface | URL |
|---|---|
| Live UI (camera feed, OCR result, PASS/FAIL, config panel) | `http://<XM2_IP>:8080` |
| REST API | `http://<XM2_IP>:8000` |

---

## Step 9 — Enable auto-start on boot (systemd)

The XM2 should start the inspection system automatically on power-up:

```bash
sudo cp /opt/xm2-ocr/xm2-ocr.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now xm2-ocr

# Check it started correctly
sudo systemctl status xm2-ocr
sudo journalctl -fu xm2-ocr
```

The service will:
- Start automatically at boot (after the filesystem is ready)
- Restart automatically on failure
- Log all output to the system journal

---

## Optional — Docker deployment

If you prefer running the system in a container:

```bash
cd /opt/xm2-ocr
docker compose up --build -d
docker compose logs -f xm2-ocr
```

The Docker image is built from `nvcr.io/nvidia/l4t-ml:r36.4.0-py3` (JetPack 6.x / CUDA 12.x), matching the XM2's firmware. Build requires internet access; runtime is fully offline.

---

## Optional — NAS sync

To archive inspection results to a network share:

```json
{
  "nas_enabled": true,
  "nas_path": "//192.168.1.100/inspections",
  "nas_username": "xm2user",
  "nas_password": "yourpassword"
}
```

Or, if the share is pre-mounted on the XM2 host:

```json
{
  "nas_enabled": true,
  "nas_path": "/mnt/nas/inspections"
}
```

When NAS sync is enabled, also ensure the systemd service can reach the network before starting. The default `xm2-ocr.service` already declares `Wants=network-online.target` for this purpose — the service will start whether or not the network is available, and fall back gracefully to local-only archiving if the NAS is unreachable.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `xm2sdk` import error | SDK not installed | Contact Imago support; ensure JetPack 6.x image is loaded |
| `Camera connected (sdk=False)` | SDK not found; using OpenCV fallback | Check `xm2sdk` install, or set `"use_xm2_sdk": false` explicitly for testing |
| `No print_area detected` | Detection model not configured | Set `roboflow_model_path` or `yolo_weights` in config |
| PaddleOCR fails, falls back to Tesseract | ARM64 wheel not installed | Install Jetson-specific `paddlepaddle` wheel (see Step 4) |
| Archive not written | `storage_path` directory not writable | `sudo chown -R jetson:jetson /opt/xm2-ocr/archive` |
| UI not reachable | Firewall blocking ports | `sudo ufw allow 8000 && sudo ufw allow 8080` |
| Service fails to start | Python path wrong | Confirm `which python3` returns `/usr/bin/python3`; edit `xm2-ocr.service` if needed |

---

## Quick reference

```bash
# Single test capture
python3 app/main.py --config config/config.json --once

# Continuous loop (1 s interval)
python3 app/main.py --config config/config.json --loop --interval 1.0

# API + UI only (trigger via HTTP POST /capture)
python3 app/main.py --config config/config.json

# Check service status
sudo systemctl status xm2-ocr

# Tail live logs
sudo journalctl -fu xm2-ocr

# Trigger a capture via API
curl -X POST http://localhost:8000/capture

# Read last 10 results
curl http://localhost:8000/results?limit=10
```
