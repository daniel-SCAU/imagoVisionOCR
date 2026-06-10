"""
Geometry normalisation module.

Handles:
  * Rotation angle estimation from bounding box or minAreaRect
  * Perspective / homography warp to a canonical upright rectangle
  * 4-point detection support (preferred) with minAreaRect fallback
"""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Target width/height for the normalised print-area crop
CANONICAL_WIDTH = 640
CANONICAL_HEIGHT = 160


def compute_rotation(bbox: tuple[int, int, int, int], contour: np.ndarray | None = None) -> float:
    """
    Estimate the rotation angle (degrees) of the print area.

    Args:
        bbox:    (x, y, w, h) axis-aligned bounding box from detection.
        contour: Optional contour array (Nx1x2).  When provided, minAreaRect
                 is used for a more accurate estimate.

    Returns:
        Rotation angle in degrees.  Positive = counter-clockwise tilt.
    """
    if contour is not None and len(contour) >= 5:
        _, _, angle = cv2.minAreaRect(contour)
        # OpenCV minAreaRect returns angle in [-90, 0); normalise to [-45, 45]
        if angle < -45:
            angle += 90
        return float(angle)
    # Fallback: axis-aligned bbox → assume no rotation
    return 0.0


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 corner points as: top-left, top-right, bottom-right, bottom-left.
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def warp_image(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    angle: float = 0.0,
    target_size: tuple[int, int] = (CANONICAL_WIDTH, CANONICAL_HEIGHT),
) -> np.ndarray:
    """
    Deskew and crop the region described by *bbox* using a rotation + warp.

    This is the fallback path when 4-corner points are not available.

    Args:
        image:       Full BGR frame.
        bbox:        (x, y, w, h) bounding box of the print area.
        angle:       Rotation angle in degrees (from :func:`compute_rotation`).
        target_size: (width, height) of the output crop.

    Returns:
        Warped BGR image of size *target_size*.
    """
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    tw, th = target_size

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    # Re-extract bbox after rotation (coordinates haven't moved much for small angles)
    x1 = max(0, int(cx - w / 2))
    y1 = max(0, int(cy - h / 2))
    x2 = min(image.shape[1], int(cx + w / 2))
    y2 = min(image.shape[0], int(cy + h / 2))
    cropped = rotated[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = rotated
    return cv2.resize(cropped, (tw, th), interpolation=cv2.INTER_LINEAR)


def normalize_print_area(
    image: np.ndarray,
    detection: Any,
    target_size: tuple[int, int] = (CANONICAL_WIDTH, CANONICAL_HEIGHT),
) -> np.ndarray:
    """
    Return a canonical, upright crop of the print area.

    Preferred path  – 4-point homography (when ``detection.points`` is set):
        A perspective transform is applied so that the parallelogram / tilted
        quad is mapped to a rectangle of *target_size*.

    Fallback path – minAreaRect-based deskew:
        The bounding box is deskewed by the estimated rotation angle and then
        resized to *target_size*.

    Args:
        image:       Full BGR frame.
        detection:   :class:`~app.detection.detector.Detection` instance.
        target_size: Desired output (width, height).

    Returns:
        Normalised BGR crop.
    """
    tw, th = target_size

    # ── Option 1: 4-corner homography ────────────────────────────────────
    if detection.points and len(detection.points) == 4:
        try:
            src = np.array(detection.points, dtype=np.float32)
            src = _order_points(src)
            dst = np.array(
                [[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]],
                dtype=np.float32,
            )
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is not None:
                warped = cv2.warpPerspective(image, H, (tw, th))
                return warped
        except Exception as exc:
            logger.warning("Homography failed: %s; using minAreaRect fallback", exc)

    # ── Option 2: minAreaRect rotation + crop ────────────────────────────
    x, y, w, h = detection.bbox
    # Build a contour from the bbox for minAreaRect estimation
    pts = np.array(
        [
            [[x, y]],
            [[x + w, y]],
            [[x + w, y + h]],
            [[x, y + h]],
        ],
        dtype=np.int32,
    )
    angle = compute_rotation(detection.bbox, contour=pts)
    return warp_image(image, detection.bbox, angle=angle, target_size=target_size)
