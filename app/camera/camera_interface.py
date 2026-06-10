"""
Camera abstraction layer for the Imago XM2 (Jetson-based) system.

Priority order:
  1. XM2 vendor SDK  (imported as ``xm2sdk`` if available)
  2. GStreamer pipeline via OpenCV (for Jetson CSI / V4L2 cameras)
  3. Plain OpenCV VideoCapture fallback
"""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_XM2_SDK_AVAILABLE = False
try:
    import xm2sdk  # type: ignore[import]  # vendor SDK – may not be installed

    _XM2_SDK_AVAILABLE = True
except ImportError:
    pass

_BACKEND_MAP: dict[str, int] = {
    "any": cv2.CAP_ANY,
    "v4l2": cv2.CAP_V4L2,
    "gstreamer": cv2.CAP_GSTREAMER,
}


class CameraInterface:
    """
    Unified camera interface that supports the XM2 vendor SDK and OpenCV
    (GStreamer or plain V4L2).

    Backend selection order (first available wins unless overridden):
      1. XM2 vendor SDK  – when ``use_xm2_sdk: true`` (default) and ``xm2sdk``
                           is importable.
      2. OpenCV            – GStreamer pipeline if ``gstreamer_pipeline`` is set,
                             otherwise plain VideoCapture.

    Usage::

        cam = CameraInterface(config)
        cam.connect()
        cam.set_trigger_mode(True)
        cam.set_exposure(2000)
        frame = cam.capture_frame(trigger=True)
        cam.disconnect()
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._capture: cv2.VideoCapture | None = None
        self._xm2_device: Any = None
        self._use_sdk = _XM2_SDK_AVAILABLE and config.get("use_xm2_sdk", True)
        self._trigger_enabled = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open connection to the camera (SDK or OpenCV)."""
        if self._use_sdk:
            self._connect_xm2()
        else:
            self._connect_opencv()
        logger.info("Camera connected (sdk=%s)", self._use_sdk)

    def disconnect(self) -> None:
        """Release all camera resources."""
        if self._use_sdk and self._xm2_device is not None:
            try:
                self._xm2_device.close()
            except Exception:
                pass
            self._xm2_device = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        logger.info("Camera disconnected")

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_trigger_mode(self, enabled: bool) -> None:
        """Enable or disable hardware trigger mode."""
        self._trigger_enabled = enabled
        if self._use_sdk and self._xm2_device is not None:
            try:
                self._xm2_device.set_trigger_mode(enabled)
            except Exception as exc:
                logger.warning("set_trigger_mode SDK error: %s", exc)
        logger.debug("Trigger mode set to %s", enabled)

    def set_exposure(self, value: int) -> None:
        """Set camera exposure (µs for SDK; OpenCV units for fallback)."""
        if self._use_sdk and self._xm2_device is not None:
            try:
                self._xm2_device.set_exposure(value)
                return
            except Exception as exc:
                logger.warning("set_exposure SDK error: %s", exc)
        if self._capture is not None:
            self._capture.set(cv2.CAP_PROP_EXPOSURE, float(value))

    def set_gain(self, value: float) -> None:
        if self._capture is not None:
            self._capture.set(cv2.CAP_PROP_GAIN, value)

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------

    def capture_frame(self, trigger: bool = True) -> np.ndarray:
        """
        Capture a single frame.

        Args:
            trigger: If True and trigger mode is active, wait for hardware
                     trigger before grabbing (SDK path only).  Ignored by the
                     OpenCV fallback which always reads the next available frame.

        Returns:
            BGR image as ``np.ndarray``.

        Raises:
            RuntimeError: If frame acquisition fails.
        """
        if self._use_sdk and self._xm2_device is not None:
            return self._capture_xm2(trigger)
        return self._capture_opencv()

    # ------------------------------------------------------------------
    # Private – XM2 SDK path
    # ------------------------------------------------------------------

    def _connect_xm2(self) -> None:
        try:
            self._xm2_device = xm2sdk.Device()
            self._xm2_device.open(self._config.get("device_serial", ""))
            width = int(self._config.get("camera_width", 1920))
            height = int(self._config.get("camera_height", 1080))
            fps = int(self._config.get("camera_fps", 30))
            self._xm2_device.set_resolution(width, height)
            self._xm2_device.set_frame_rate(fps)
        except Exception as exc:
            logger.warning("XM2 SDK connect failed (%s), falling back to OpenCV", exc)
            self._use_sdk = False
            self._connect_opencv()

    def _capture_xm2(self, trigger: bool) -> np.ndarray:
        try:
            raw = self._xm2_device.grab(wait_trigger=trigger and self._trigger_enabled)
            # SDK returns bytes or numpy array depending on version
            if isinstance(raw, bytes):
                arr = np.frombuffer(raw, dtype=np.uint8)
                w = int(self._config.get("camera_width", 1920))
                h = int(self._config.get("camera_height", 1080))
                frame = arr.reshape((h, w, 3))
            else:
                frame = np.asarray(raw)
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.ndim == 3 else frame
        except Exception as exc:
            raise RuntimeError(f"XM2 SDK frame capture error: {exc}") from exc

    # ------------------------------------------------------------------
    # Private – OpenCV path
    # ------------------------------------------------------------------

    def _connect_opencv(self) -> None:
        pipeline = self._config.get("gstreamer_pipeline", "")
        backend_name = str(self._config.get("camera_backend", "any")).lower()
        backend = _BACKEND_MAP.get(backend_name, cv2.CAP_ANY)
        source: Any = pipeline if pipeline else self._config.get("camera_source", 0)

        self._capture = cv2.VideoCapture(source, backend)
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open camera source: {source!r}")

        for prop_name, prop_id in (
            ("camera_width", cv2.CAP_PROP_FRAME_WIDTH),
            ("camera_height", cv2.CAP_PROP_FRAME_HEIGHT),
            ("camera_fps", cv2.CAP_PROP_FPS),
        ):
            if prop_name in self._config:
                self._capture.set(prop_id, float(self._config[prop_name]))

        exposure = self._config.get("camera_exposure")
        if exposure is not None:
            self._capture.set(cv2.CAP_PROP_EXPOSURE, float(exposure))

        # Warm up – discard initial frames that may be dark/blurry
        warmup = int(self._config.get("warmup_frames", 5))
        for _ in range(warmup):
            self._capture.read()

    def _capture_opencv(self) -> np.ndarray:
        if self._capture is None:
            raise RuntimeError("Camera not connected")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError("OpenCV failed to read frame")
        return frame
