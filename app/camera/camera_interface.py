"""
Camera abstraction layer for the Imago XM2 (Jetson-based) system.

Priority order:
  1. XM2 vendor SDK  (imported as ``xm2sdk`` if available)
  2. GenICam via Harvester (``harvesters`` package, optional)
  3. GStreamer pipeline via OpenCV (for Jetson CSI / V4L2 / GigE cameras)
  4. Plain OpenCV VideoCapture fallback
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

_HARVESTER_AVAILABLE = False
try:
    from harvesters.core import Harvester  # type: ignore[import]

    _HARVESTER_AVAILABLE = True
except ImportError:
    pass

_BACKEND_MAP: dict[str, int] = {
    "any": cv2.CAP_ANY,
    "v4l2": cv2.CAP_V4L2,
    "gstreamer": cv2.CAP_GSTREAMER,
    "dshow": cv2.CAP_DSHOW,
}


class CameraInterface:
    """
    Unified camera interface that supports the XM2 vendor SDK, GenICam via
    Harvester, and OpenCV (GStreamer or plain V4L2/DirectShow).

    Backend selection order (first available wins unless overridden):
      1. XM2 vendor SDK  – when ``use_xm2_sdk: true`` (default) and ``xm2sdk``
                           is importable.
      2. GenICam/Harvester – when ``use_genicam: true`` and ``harvesters`` is
                             installed.  Requires a valid ``.cti`` producer file
                             path in ``genicam_cti``.
      3. OpenCV            – GStreamer pipeline if ``gstreamer_pipeline`` is set,
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
        self._harvester: Any = None          # Harvester instance
        self._h_acquirer: Any = None         # ImageAcquirer from Harvester
        self._use_sdk = _XM2_SDK_AVAILABLE and config.get("use_xm2_sdk", True)
        self._use_genicam = (
            not self._use_sdk
            and _HARVESTER_AVAILABLE
            and bool(config.get("use_genicam", False))
        )
        self._trigger_enabled = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open connection to the camera (SDK, GenICam/Harvester, or OpenCV)."""
        if self._use_sdk:
            self._connect_xm2()
        elif self._use_genicam:
            self._connect_harvester()
        else:
            self._connect_opencv()
        logger.info(
            "Camera connected (sdk=%s, genicam=%s)", self._use_sdk, self._use_genicam
        )

    def disconnect(self) -> None:
        """Release all camera resources."""
        if self._use_sdk and self._xm2_device is not None:
            try:
                self._xm2_device.close()
            except Exception:
                pass
            self._xm2_device = None
        if self._h_acquirer is not None:
            try:
                self._h_acquirer.stop_acquisition()
                self._h_acquirer.destroy()
            except Exception:
                pass
            self._h_acquirer = None
        if self._harvester is not None:
            try:
                self._harvester.reset()
            except Exception:
                pass
            self._harvester = None
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
        elif self._use_genicam and self._h_acquirer is not None:
            try:
                node_map = self._h_acquirer.remote_device.node_map
                node_map.TriggerMode.value = "On" if enabled else "Off"
            except Exception as exc:
                logger.warning("set_trigger_mode GenICam error: %s", exc)
        logger.debug("Trigger mode set to %s", enabled)

    def set_exposure(self, value: int) -> None:
        """Set camera exposure (µs for SDK/GenICam; OpenCV units for fallback)."""
        if self._use_sdk and self._xm2_device is not None:
            try:
                self._xm2_device.set_exposure(value)
                return
            except Exception as exc:
                logger.warning("set_exposure SDK error: %s", exc)
        elif self._use_genicam and self._h_acquirer is not None:
            try:
                node_map = self._h_acquirer.remote_device.node_map
                node_map.ExposureTime.value = float(value)
                return
            except Exception as exc:
                logger.warning("set_exposure GenICam error: %s", exc)
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
                     trigger before grabbing (SDK / GenICam paths).  Ignored
                     by the OpenCV fallback which always reads the next
                     available frame.

        Returns:
            BGR image as ``np.ndarray``.

        Raises:
            RuntimeError: If frame acquisition fails.
        """
        if self._use_sdk and self._xm2_device is not None:
            return self._capture_xm2(trigger)
        if self._use_genicam and self._h_acquirer is not None:
            return self._capture_harvester()
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
    # Private – GenICam / Harvester path
    # ------------------------------------------------------------------

    def _connect_harvester(self) -> None:
        cti_file = self._config.get("genicam_cti", "")
        if not cti_file:
            logger.warning(
                "GenICam/Harvester selected but 'genicam_cti' not set in config; "
                "falling back to OpenCV."
            )
            self._use_genicam = False
            self._connect_opencv()
            return
        try:
            self._harvester = Harvester()
            self._harvester.add_file(cti_file)
            self._harvester.update()
            self._h_acquirer = self._harvester.create_image_acquirer(0)
            self._h_acquirer.start_acquisition()
            logger.info("GenICam/Harvester connected via CTI: %s", cti_file)
        except Exception as exc:
            logger.warning(
                "Harvester connect failed (%s); falling back to OpenCV.", exc
            )
            self._use_genicam = False
            if self._harvester is not None:
                try:
                    self._harvester.reset()
                except Exception:
                    pass
                self._harvester = None
            self._h_acquirer = None
            self._connect_opencv()

    def _capture_harvester(self) -> np.ndarray:
        try:
            with self._h_acquirer.fetch_buffer() as buf:
                component = buf.payload.components[0]
                width = component.width
                height = component.height
                data = component.data.reshape(height, width, -1)
                frame = np.array(data, dtype=np.uint8)
                # GenICam typically delivers Mono8 or BayerRG8; convert to BGR
                if frame.ndim == 2 or frame.shape[2] == 1:
                    frame = cv2.cvtColor(
                        frame.squeeze(), cv2.COLOR_GRAY2BGR
                    )
                elif frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame
        except Exception as exc:
            raise RuntimeError(f"Harvester frame capture error: {exc}") from exc

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
