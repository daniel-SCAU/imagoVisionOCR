"""
Inspection pipeline orchestrator.

Runtime flow::

    capture_frame()
      → detect_objects(frame)
      → select print_area
      → normalize_print_area()   ← geometry correction
      → extract_text()           ← PaddleOCR
      → validate_text()
      → archive.save()
      → db.save_result()
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.camera.camera_interface import CameraInterface
from app.config.settings import save_result
from app.detection.detector import ObjectDetector, draw_detections
from app.geometry.transform import normalize_print_area
from app.ocr.reader import OCRReader
from app.storage.archive import ArchiveManager
from app.validation.rules import validate_text

logger = logging.getLogger(__name__)

# Module-level singletons (lazy-initialised per config hash to avoid reload)
_camera: CameraInterface | None = None
_detector: ObjectDetector | None = None
_ocr: OCRReader | None = None
_archive: ArchiveManager | None = None
_config_hash: int = 0


def _get_singletons(cfg: dict[str, Any]) -> tuple[
    CameraInterface, ObjectDetector, OCRReader, ArchiveManager
]:
    global _camera, _detector, _ocr, _archive, _config_hash

    h = hash(str(sorted(cfg.items())))
    if h == _config_hash and _camera is not None:
        return _camera, _detector, _ocr, _archive  # type: ignore[return-value]

    # Re-initialise on config change
    if _camera is not None:
        try:
            _camera.disconnect()
        except Exception:
            pass

    _camera = CameraInterface(cfg)
    _camera.connect()
    _camera.set_trigger_mode(bool(cfg.get("trigger_mode", False)))
    exposure = cfg.get("camera_exposure")
    if exposure is not None:
        _camera.set_exposure(int(exposure))

    _detector = ObjectDetector(cfg)
    _ocr = OCRReader(cfg)
    _archive = ArchiveManager(cfg)
    _config_hash = h
    return _camera, _detector, _ocr, _archive


def run_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Execute one full inspection cycle.

    Returns a dict with keys:
      timestamp, ocr_text, validation_result, passed, bbox, image_path, image
    """
    ts = datetime.now(timezone.utc).isoformat()

    camera, detector, ocr_reader, archive = _get_singletons(cfg)

    # 1. Capture
    frame = camera.capture_frame(trigger=True)
    logger.debug("Frame captured: %s", frame.shape)

    # 2. Detect
    detections = detector.detect(frame)
    logger.debug("Detections: %d", len(detections))

    # 3. Select print area
    print_det = detector.select_print_area(detections)

    annotated = draw_detections(frame, detections)

    if print_det is None:
        logger.warning("No print_area detected in frame")
        ocr_text = ""
        validation = "FAIL"
        passed = False
        bbox: dict[str, Any] = {}
        image_path = ""
        img_path, _ = archive.save(
            annotated, ocr_text, validation, bbox,
            extra={"note": "no_detection", "timestamp": ts},
        )
        image_path = str(img_path)
    else:
        bbox = {
            "x": print_det.bbox[0],
            "y": print_det.bbox[1],
            "w": print_det.bbox[2],
            "h": print_det.bbox[3],
        }

        # 4. Geometry normalisation
        normalised = normalize_print_area(frame, print_det)
        logger.debug("Normalised crop: %s", normalised.shape)

        # 5. OCR
        ocr_text = ocr_reader.extract_text(normalised)
        logger.info("OCR text: %r", ocr_text)

        # 6. Validate
        rules = cfg.get("validation_rules", "{}")
        vresult = validate_text(ocr_text, rules)
        validation = "PASS" if vresult.passed else "FAIL"
        passed = vresult.passed
        logger.info("Validation: %s – %s", validation, vresult.reason)

        # 7. Archive
        img_path, _ = archive.save(
            annotated, ocr_text, validation, bbox,
            extra={"timestamp": ts, "reason": vresult.reason},
        )
        image_path = str(img_path)

    # 8. Persist to DB
    save_result(
        timestamp_utc=ts,
        ocr_text=ocr_text,
        validation=validation,
        bbox=bbox,
        image_path=image_path,
    )

    return {
        "timestamp": ts,
        "ocr_text": ocr_text,
        "validation_result": validation,
        "passed": passed,
        "bbox": bbox,
        "image_path": image_path,
        "image": annotated,
    }


def shutdown_pipeline() -> None:
    """Release camera and GPU resources."""
    global _camera
    if _camera is not None:
        try:
            _camera.disconnect()
        except Exception:
            pass
        _camera = None
