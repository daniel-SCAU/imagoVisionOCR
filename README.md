# imagoVisionOCR

Trigger-based OCR capture application for an Imago XM2 camera on Jetson Orin Super 16GB.

## Features

- Capture image on trigger (`once`, `stdin`, or timed interval).
- ROI cropping with configurable **anchor point**.
- OCR extraction using Tesseract.
- Compare extracted text with configured variables/expected values.
- Save full image (and optional ROI crops/reports) to a remote folder path.
- Configurable camera settings to improve image clarity.

## Prerequisites

- Python 3.10+
- Tesseract OCR engine installed on device
- Camera available through OpenCV (`/dev/video*` or GStreamer pipeline)
- Remote folder mounted and writable on the Jetson (for example via NFS/SMB mount)

## Install

```bash
pip install -r requirements.txt
```

## Configuration

Copy and edit the example configuration:

```bash
cp /home/runner/work/imagoVisionOCR/imagoVisionOCR/daniel-SCAU/imagoVisionOCR/config.example.json /home/runner/work/imagoVisionOCR/imagoVisionOCR/daniel-SCAU/imagoVisionOCR/config.json
```

Important fields:

- `anchor`: base point used to offset all ROI boxes.
- `rois`: each ROI has offsets and size, plus either:
  - `variable`: key looked up in `variables`, or
  - `expected_values`: explicit list of accepted values.
- `storage.remote_folder`: mounted remote destination for images/results.
- `camera.settings`: optional OpenCV capture properties (focus, exposure, gain, etc.).

## Run

One-shot trigger:

```bash
python /home/runner/work/imagoVisionOCR/imagoVisionOCR/daniel-SCAU/imagoVisionOCR/imago_vision_ocr_app.py --config /home/runner/work/imagoVisionOCR/imagoVisionOCR/daniel-SCAU/imagoVisionOCR/config.json --once
```

Interactive trigger (press Enter each capture):

```bash
python /home/runner/work/imagoVisionOCR/imagoVisionOCR/daniel-SCAU/imagoVisionOCR/imago_vision_ocr_app.py --config /home/runner/work/imagoVisionOCR/imagoVisionOCR/daniel-SCAU/imagoVisionOCR/config.json
```

Timed trigger:

```bash
python /home/runner/work/imagoVisionOCR/imagoVisionOCR/daniel-SCAU/imagoVisionOCR/imago_vision_ocr_app.py --config /home/runner/work/imagoVisionOCR/imagoVisionOCR/daniel-SCAU/imagoVisionOCR/config.json --interval 2.0
```

## Output

For each trigger, the app writes to `storage.remote_folder`:

- Full captured image (`capture_<timestamp>.jpg`)
- Optional ROI crops (`roi_<name>_<timestamp>.jpg`)
- Optional OCR comparison report (`report_<timestamp>.json`)

It also prints OCR and comparison status to stdout.