# imagoVisionOCR

Production-grade edge AI OCR inspection system running on an **Imago XM2 (NVIDIA Jetson)**.

Captures images from an industrial camera trigger, detects and normalises the print area geometry, performs OCR, validates extracted text against configurable rules, archives results, and exposes a live web UI.

---

## Architecture

```
app/
  camera/        CameraInterface – XM2 SDK + OpenCV/GStreamer fallback
  detection/     ObjectDetector  – Roboflow Inference (local) / YOLOv8 (fallback)
  geometry/      transform       – rotation correction + perspective warp
  ocr/           OCRReader       – PaddleOCR (+ Tesseract fallback)
  validation/    rules           – exact match + regex validation engine
  storage/       ArchiveManager  – dated local archive + optional SMB/NFS sync
  api/           FastAPI service – /capture /status /config /results
  ui/            NiceGUI         – live feed, OCR result, PASS/FAIL, config panel
  config/        settings        – SQLite-backed config + result store
  pipeline.py    Orchestrator    – wires all modules together
  main.py        Entry point
```

**Runtime flow:**

```
Trigger Event
  → capture_frame()
  → detect_objects(frame)          ← YOLO / Roboflow (no fixed ROI)
  → select print_area
  → normalize_print_area()         ← rotation + perspective correction
  → extract_text()                 ← PaddleOCR on normalised crop
  → validate_text()                ← exact / regex rules
  → archive.save()                 ← YYYY/MM/DD/ dated tree + optional NAS
  → db.save_result()               ← SQLite
```

---

## Requirements

- Python 3.10+
- NVIDIA Jetson Orin (ARM64) with JetPack 6.x and CUDA
- Camera accessible via XM2 SDK, V4L2 (`/dev/video*`), or GStreamer pipeline

---

## Installation

```bash
# Clone
git clone https://github.com/daniel-SCAU/imagoVisionOCR
cd imagoVisionOCR

# Install dependencies
pip install -r requirements.txt

# Copy example config
cp config.example.json config.json
# Edit config.json — set detection credentials, storage path, validation rules
```

For **Roboflow** local detection, set `roboflow_model_path` to the path of your local model files in `config.json`.
For **local YOLO**, set `detection_model` to `"yolo"` and provide `yolo_weights`.

---

## Run

### Single capture
```bash
python app/main.py --config config.json --once
```

### Continuous loop (every 1 s)
```bash
python app/main.py --config config.json --loop --interval 1.0
```

### API + UI only (trigger via HTTP POST)
```bash
python app/main.py --config config.json
# API:  http://localhost:8000
# UI:   http://localhost:8080
```

---

## API Endpoints

| Method | Path        | Description                              |
|--------|-------------|------------------------------------------|
| POST   | `/capture`  | Trigger one inspection cycle             |
| GET    | `/status`   | System status + last result              |
| GET    | `/config`   | Read current config                      |
| POST   | `/config`   | Update config (persisted to SQLite)      |
| GET    | `/results`  | Last N inspection results                |

---

## Configuration

Key fields in `config.json` / SQLite:

| Key                    | Default          | Description                                   |
|------------------------|------------------|-----------------------------------------------|
| `detection_model`      | `"roboflow"`     | `"roboflow"` or `"yolo"`                      |
| `roboflow_model_path`  | `""`             | Path to local Roboflow model files            |
| `roboflow_project`     | `""`             | Roboflow project slug (fallback to local cache) |
| `yolo_weights`         | `"yolov8n.pt"`   | Path or model name for local YOLO             |
| `confidence_threshold` | `0.5`            | Minimum detection confidence                  |
| `ocr_engine`           | `"paddleocr"`    | `"paddleocr"` (GPU) or `"tesseract"` (fallback) |
| `camera_exposure`      | `2000`           | Exposure in µs (XM2 SDK) or OpenCV units      |
| `validation_rules`     | `"{}"`           | JSON string of field → expected/regex pattern |
| `storage_path`         | `"./archive"`    | Local archive root directory                  |
| `nas_enabled`          | `false`          | Enable NAS sync                               |
| `nas_path`             | `"//nas/ocr"`    | UNC path or mount point                       |

---

## Docker Deployment (Jetson ARM64)

```bash
# Build and run
docker compose up --build -d

# Logs
docker compose logs -f xm2-ocr
```

---

## systemd Service

```bash
sudo cp xm2-ocr.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now xm2-ocr
sudo journalctl -fu xm2-ocr
```

---

## Output Archive Structure

```
archive/
  2024/06/15/
    image_20240615_143022_000000_PASS.jpg
    image_20240615_143022_000000_PASS.json
```

JSON metadata example:
```json
{
  "timestamp": "2024-06-15T14:30:22.000000+00:00",
  "ocr_text": "LOT240615A",
  "validation_result": "PASS",
  "bbox": {"x": 420, "y": 310, "w": 200, "h": 50},
  "reason": "All rules matched"
}
```

