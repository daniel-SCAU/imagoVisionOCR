"""
Object detection module supporting:
  Option A (default): Roboflow Inference API / local runtime
  Option B:           Ultralytics YOLOv8/v11 local model

Output ``Detection`` dataclass:
  class_name   – e.g. "print_area", "label", "seal"
  confidence   – float 0–1
  bbox         – (x, y, w, h) in pixels
  points       – optional 4-corner polygon for homography (list of [x, y])
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, w, h
    points: list[list[int]] = field(default_factory=list)  # optional 4-corner polygon


def _center_to_xywh(cx: float, cy: float, w: float, h: float) -> tuple[int, int, int, int]:
    """Convert center-x, center-y, width, height to top-left x, y, width, height."""
    return int(cx - w / 2), int(cy - h / 2), int(w), int(h)


class ObjectDetector:
    """
    Unified detector.  Backend is selected at construction time based on config.

    config keys:
      detection_model       – "roboflow" | "yolo"
      roboflow_api_key      – Roboflow API key (Option A)
      roboflow_project      – Roboflow project ID
      roboflow_version      – model version integer
      yolo_weights          – path or model name, e.g. "yolov8n.pt" (Option B)
      confidence_threshold  – minimum confidence to keep a detection
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._threshold = float(config.get("confidence_threshold", 0.5))
        self._backend = str(config.get("detection_model", "roboflow")).lower()
        self._model: Any = None
        self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Run detection on *image* (BGR numpy array).

        Returns a list of :class:`Detection` objects filtered by confidence.
        """
        if self._backend == "roboflow":
            return self._detect_roboflow(image)
        return self._detect_yolo(image)

    def select_print_area(self, detections: list[Detection]) -> Detection | None:
        """Return the highest-confidence ``print_area`` detection, or None."""
        candidates = [d for d in detections if d.class_name == "print_area"]
        if not candidates:
            return None
        return max(candidates, key=lambda d: d.confidence)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._backend == "roboflow":
            self._load_roboflow()
        else:
            self._load_yolo()

    def _load_roboflow(self) -> None:
        api_key = self._config.get("roboflow_api_key", "")
        project = self._config.get("roboflow_project", "")
        version = int(self._config.get("roboflow_version", 1))

        if not api_key or not project:
            logger.warning(
                "Roboflow credentials not set; detector will return empty results. "
                "Set roboflow_api_key and roboflow_project in config."
            )
            return

        try:
            from inference import get_roboflow_model  # type: ignore[import]

            self._model = get_roboflow_model(
                model_id=f"{project}/{version}", api_key=api_key
            )
            logger.info("Roboflow model loaded: %s/%s", project, version)
        except ImportError:
            logger.warning(
                "inference package not installed; falling back to YOLOv8 local model."
            )
            self._backend = "yolo"
            self._load_yolo()
        except Exception as exc:
            logger.error("Failed to load Roboflow model: %s", exc)

    def _load_yolo(self) -> None:
        weights = self._config.get("yolo_weights", "yolov8n.pt")
        try:
            from ultralytics import YOLO  # type: ignore[import]

            self._model = YOLO(weights)
            logger.info("YOLO model loaded: %s", weights)
        except ImportError:
            logger.error(
                "ultralytics package not installed. "
                "Install with: pip install ultralytics"
            )
        except Exception as exc:
            logger.error("Failed to load YOLO model: %s", exc)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _detect_roboflow(self, image: np.ndarray) -> list[Detection]:
        if self._model is None:
            return []
        try:
            results = self._model.infer(image)[0]
            detections: list[Detection] = []
            for pred in results.predictions:
                conf = float(pred.confidence)
                if conf < self._threshold:
                    continue
                x, y, w, h = _center_to_xywh(pred.x, pred.y, pred.width, pred.height)

                # Extract 4-corner polygon when the model returns instance/keypoint data.
                # Roboflow polygon predictions expose a ``points`` attribute (list of
                # objects with ``.x`` / ``.y`` properties).  We sample exactly 4 corners
                # so the geometry module can use the preferred homography path.
                points: list[list[int]] = []
                raw_pts = getattr(pred, "points", None)
                if raw_pts and len(raw_pts) >= 4:
                    sampled = _sample_4_corners(
                        [[int(p.x), int(p.y)] for p in raw_pts]
                    )
                    if sampled:
                        points = sampled

                detections.append(
                    Detection(
                        class_name=pred.class_name,
                        confidence=conf,
                        bbox=(x, y, w, h),
                        points=points,
                    )
                )
            return detections
        except Exception as exc:
            logger.error("Roboflow inference error: %s", exc)
            return []

    def _detect_yolo(self, image: np.ndarray) -> list[Detection]:
        if self._model is None:
            return []
        try:
            results = self._model.predict(
                image,
                conf=self._threshold,
                verbose=False,
                device="cuda:0" if _cuda_available() else "cpu",
            )
            detections: list[Detection] = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                # Pre-build per-index polygon lookup from segmentation masks (YOLOv8-seg)
                # or oriented bounding boxes (YOLOv8-obb).
                seg_points: dict[int, list[list[int]]] = {}
                if result.masks is not None:
                    # result.masks.xy is a list (one per detection) of (N,2) float arrays
                    for idx, poly in enumerate(result.masks.xy):
                        if poly is not None and len(poly) >= 4:
                            sampled = _sample_4_corners(
                                [[int(p[0]), int(p[1])] for p in poly]
                            )
                            if sampled:
                                seg_points[idx] = sampled
                elif result.obb is not None:
                    # result.obb.xyxyxyxy shape: (N, 4, 2)
                    try:
                        for idx, corners in enumerate(result.obb.xyxyxyxy):
                            seg_points[idx] = [[int(p[0]), int(p[1])] for p in corners]
                    except Exception:
                        pass

                for i, box in enumerate(boxes):
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = result.names.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                    detections.append(
                        Detection(
                            class_name=class_name,
                            confidence=conf,
                            bbox=(x1, y1, x2 - x1, y2 - y1),
                            points=seg_points.get(i, []),
                        )
                    )
            return detections
        except Exception as exc:
            logger.error("YOLO inference error: %s", exc)
            return []


def _sample_4_corners(points: list[list[int]]) -> list[list[int]]:
    """
    Reduce an arbitrary polygon to exactly 4 corner points suitable for a
    homography transform.

    Strategy:
      * If the polygon already has exactly 4 points, return them as-is.
      * Otherwise find the 4 points that are geometrically closest to the
        corners of the polygon's axis-aligned bounding box (top-left,
        top-right, bottom-right, bottom-left order).

    Returns an empty list if *points* has fewer than 4 entries.
    """
    if len(points) < 4:
        return []
    if len(points) == 4:
        return points

    import math

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    corners_ref = [
        (min_x, min_y),  # top-left
        (max_x, min_y),  # top-right
        (max_x, max_y),  # bottom-right
        (min_x, max_y),  # bottom-left
    ]
    result: list[list[int]] = []
    for rx, ry in corners_ref:
        best = min(points, key=lambda p: math.hypot(p[0] - rx, p[1] - ry))
        result.append(best)
    return result


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore[import]

        return torch.cuda.is_available()
    except ImportError:
        return False


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Return a copy of *image* with detection bounding boxes drawn."""
    overlay = image.copy()
    for det in detections:
        x, y, w, h = det.bbox
        color = (0, 255, 0) if det.class_name == "print_area" else (255, 128, 0)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(overlay, label, (x, max(y - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    return overlay
