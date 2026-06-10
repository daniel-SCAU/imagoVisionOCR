#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract


BACKENDS: dict[str, int] = {
    "any": cv2.CAP_ANY,
    "v4l2": cv2.CAP_V4L2,
    "gstreamer": cv2.CAP_GSTREAMER,
}


@dataclass
class Roi:
    name: str
    x_offset: int
    y_offset: int
    width: int
    height: int
    variable: str | None = None
    expected_values: list[str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Imago XM2 trigger-based OCR application")
    parser.add_argument("--config", required=True, help="Path to JSON configuration file")
    parser.add_argument("--once", action="store_true", help="Capture once and exit")
    parser.add_argument("--interval", type=float, help="Capture interval in seconds")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_capture(camera_cfg: dict[str, Any]) -> cv2.VideoCapture:
    pipeline = camera_cfg.get("gstreamer_pipeline")
    backend_name = str(camera_cfg.get("backend", "any")).lower()
    backend = BACKENDS.get(backend_name, cv2.CAP_ANY)
    source = pipeline if pipeline else camera_cfg.get("source", 0)

    capture = cv2.VideoCapture(source, backend)
    if not capture.isOpened():
        raise RuntimeError("Unable to open camera source")

    for prop_name, prop_id in (
        ("width", cv2.CAP_PROP_FRAME_WIDTH),
        ("height", cv2.CAP_PROP_FRAME_HEIGHT),
        ("fps", cv2.CAP_PROP_FPS),
    ):
        if prop_name in camera_cfg:
            capture.set(prop_id, float(camera_cfg[prop_name]))

    settings = camera_cfg.get("settings", {})
    property_map = {
        "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
        "exposure": cv2.CAP_PROP_EXPOSURE,
        "focus": cv2.CAP_PROP_FOCUS,
        "gain": cv2.CAP_PROP_GAIN,
        "brightness": cv2.CAP_PROP_BRIGHTNESS,
        "contrast": cv2.CAP_PROP_CONTRAST,
        "sharpness": cv2.CAP_PROP_SHARPNESS,
    }
    for key, value in settings.items():
        prop = property_map.get(key)
        if prop is not None:
            capture.set(prop, float(value))

    return capture


def build_rois(config: dict[str, Any]) -> list[Roi]:
    rois = []
    for roi in config.get("rois", []):
        rois.append(
            Roi(
                name=roi["name"],
                x_offset=int(roi["x_offset"]),
                y_offset=int(roi["y_offset"]),
                width=int(roi["width"]),
                height=int(roi["height"]),
                variable=roi.get("variable"),
                expected_values=roi.get("expected_values"),
            )
        )
    if not rois:
        raise ValueError("No ROIs configured")
    return rois


def clamp_roi(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x = max(0, x)
    y = max(0, y)
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h


def preprocess_image(img: np.ndarray, mode: str) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == "none":
        return gray
    if mode == "otsu":
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    if mode == "adaptive_threshold":
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)
    return gray


def run_ocr(img: np.ndarray, ocr_cfg: dict[str, Any]) -> str:
    psm = int(ocr_cfg.get("psm", 7))
    oem = int(ocr_cfg.get("oem", 3))
    whitelist = ocr_cfg.get("whitelist")
    lang = ocr_cfg.get("lang", "eng")
    mode = str(ocr_cfg.get("preprocess", "adaptive_threshold"))
    processed = preprocess_image(img, mode)

    config = f"--oem {oem} --psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"
    text = pytesseract.image_to_string(processed, lang=lang, config=config)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compare_text(actual: str, expected_values: list[str], case_sensitive: bool) -> bool:
    if not case_sensitive:
        actual = actual.lower()
        expected_values = [v.lower() for v in expected_values]
    return actual in expected_values


def save_image(path: Path, image: np.ndarray, image_format: str, jpeg_quality: int) -> None:
    if image_format.lower() in {"jpg", "jpeg"}:
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    else:
        cv2.imwrite(str(path), image)


def resolve_expected_values(roi: Roi, variables: dict[str, str]) -> list[str]:
    if roi.expected_values:
        return [str(v) for v in roi.expected_values]
    if roi.variable:
        if roi.variable not in variables:
            raise KeyError(f"ROI '{roi.name}' references missing variable '{roi.variable}'")
        return [str(variables[roi.variable])]
    raise ValueError(f"ROI '{roi.name}' must define either 'variable' or 'expected_values'")


def process_trigger(
    capture: cv2.VideoCapture,
    config: dict[str, Any],
    rois: list[Roi],
    output_dir: Path,
) -> dict[str, Any]:
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError("Failed to capture frame")

    anchor_cfg = config.get("anchor", {})
    anchor_x = int(anchor_cfg.get("x", 0))
    anchor_y = int(anchor_cfg.get("y", 0))
    variables = config.get("variables", {})
    ocr_cfg = config.get("ocr", {})
    case_sensitive = bool(config.get("comparison", {}).get("case_sensitive", False))
    storage = config.get("storage", {})
    image_format = str(storage.get("image_format", "jpg")).lower()
    jpeg_quality = int(storage.get("jpeg_quality", 95))
    save_roi_images = bool(storage.get("save_roi_images", True))
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")

    capture_name = output_dir / f"capture_{timestamp}.{image_format}"
    save_image(capture_name, frame, image_format=image_format, jpeg_quality=jpeg_quality)

    img_h, img_w = frame.shape[:2]
    roi_results: list[dict[str, Any]] = []
    all_match = True
    for roi in rois:
        x = anchor_x + roi.x_offset
        y = anchor_y + roi.y_offset
        x, y, w, h = clamp_roi(x, y, roi.width, roi.height, img_w, img_h)
        crop = frame[y : y + h, x : x + w]
        text = run_ocr(crop, ocr_cfg)
        expected_values = resolve_expected_values(roi, variables)
        matched = compare_text(text, expected_values, case_sensitive=case_sensitive)
        all_match = all_match and matched

        if save_roi_images:
            roi_name = output_dir / f"roi_{roi.name}_{timestamp}.{image_format}"
            save_image(roi_name, crop, image_format=image_format, jpeg_quality=jpeg_quality)

        roi_results.append(
            {
                "name": roi.name,
                "anchor_x": anchor_x,
                "anchor_y": anchor_y,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "text": text,
                "expected_values": expected_values,
                "matched": matched,
            }
        )

    result = {
        "timestamp_utc": timestamp,
        "capture_image": str(capture_name),
        "all_match": all_match,
        "rois": roi_results,
    }

    if storage.get("save_report", True):
        report_path = output_dir / f"report_{timestamp}.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result


def warmup(capture: cv2.VideoCapture, frames: int) -> None:
    for _ in range(max(0, frames)):
        capture.read()


def print_result(result: dict[str, Any]) -> None:
    print(f"[{result['timestamp_utc']}] all_match={result['all_match']}")
    for roi in result["rois"]:
        print(
            f"  - {roi['name']}: text='{roi['text']}' expected={roi['expected_values']} matched={roi['matched']}"
        )


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    rois = build_rois(config)

    storage = config.get("storage", {})
    remote_folder = Path(storage.get("remote_folder", "./output")).expanduser().resolve()
    remote_folder.mkdir(parents=True, exist_ok=True)

    capture = build_capture(config.get("camera", {}))
    warmup_frames = int(config.get("runtime", {}).get("warmup_frames", 5))
    warmup(capture, warmup_frames)

    trigger_cfg = config.get("trigger", {})
    mode = str(trigger_cfg.get("mode", "stdin")).lower()
    if args.once:
        mode = "once"
    if args.interval is not None:
        mode = "interval"
        trigger_cfg["interval_seconds"] = args.interval

    try:
        if mode == "once":
            print_result(process_trigger(capture, config, rois, remote_folder))
            return 0

        if mode == "interval":
            interval = float(trigger_cfg.get("interval_seconds", 1.0))
            while True:
                print_result(process_trigger(capture, config, rois, remote_folder))
                time.sleep(max(0.01, interval))

        print("Press Enter to trigger capture, Ctrl+C to exit.")
        while True:
            input()
            print_result(process_trigger(capture, config, rois, remote_folder))
    except KeyboardInterrupt:
        return 0
    finally:
        capture.release()


if __name__ == "__main__":
    raise SystemExit(main())
