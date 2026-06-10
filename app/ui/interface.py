"""
NiceGUI-based web interface for the XM2 OCR inspection system.

Provides:
  * Live camera feed (MJPEG via periodic base64 refresh)
  * Last detection overlay (bounding box)
  * OCR result + PASS/FAIL indicator
  * Configuration panel (exposure, threshold, validation rules, storage path)

Run standalone::

    python -m app.ui.interface

Or via main.py (which starts both the API and UI in the same process).
"""
from __future__ import annotations

import asyncio
import base64
import logging
import threading
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from nicegui import ui  # type: ignore[import]
    _NICEGUI_AVAILABLE = True
except ImportError:
    _NICEGUI_AVAILABLE = False
    logger.error("nicegui is not installed. Run: pip install nicegui")


def _encode_frame(frame: np.ndarray | None, overlay: dict[str, Any] | None = None) -> str:
    """Encode a BGR numpy frame to a base64 JPEG data URI."""
    if frame is None:
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No Frame", (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
        frame = placeholder

    if overlay:
        frame = frame.copy()
        x, y, w, h = (
            int(overlay.get("x", 0)),
            int(overlay.get("y", 0)),
            int(overlay.get("w", 0)),
            int(overlay.get("h", 0)),
        )
        color = (0, 200, 0) if overlay.get("passed") else (0, 0, 220)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    b64 = base64.b64encode(buf.tobytes()).decode()
    return f"data:image/jpeg;base64,{b64}"


def build_ui(shared_state: dict[str, Any]) -> None:
    """
    Build and register NiceGUI pages.

    Args:
        shared_state: A mutable dict shared with the pipeline thread, containing:
            "frame"      – latest BGR numpy frame (or None)
            "ocr_text"   – str
            "passed"     – bool | None
            "bbox"       – {x, y, w, h} dict
            "config"     – current config dict (writable)
            "save_config"– callable(updates: dict)
    """
    if not _NICEGUI_AVAILABLE:
        return

    @ui.page("/")
    def main_page() -> None:
        ui.label("XM2 Jetson OCR Inspection").classes("text-2xl font-bold mb-4")

        with ui.row().classes("w-full gap-4"):
            # ── Left column: live feed + result ──────────────────────────
            with ui.column().classes("flex-1"):
                ui.label("Live Feed").classes("text-lg font-semibold")
                feed_img = ui.image("").classes("w-full rounded border")

                with ui.row().classes("items-center gap-4 mt-2"):
                    result_label = ui.label("—").classes("text-xl font-bold")
                    ocr_label = ui.label("").classes("text-base text-gray-600")

            # ── Right column: config ──────────────────────────────────────
            with ui.column().classes("w-72"):
                ui.label("Configuration").classes("text-lg font-semibold")

                cfg = shared_state.get("config", {})

                exposure_input = ui.number(
                    "Camera Exposure (µs)", value=cfg.get("camera_exposure", 2000), min=1, max=100000
                ).classes("w-full")

                threshold_input = ui.number(
                    "Detection Threshold", value=cfg.get("confidence_threshold", 0.5),
                    min=0.0, max=1.0, step=0.05, format="%.2f"
                ).classes("w-full")

                rules_input = ui.textarea(
                    "Validation Rules (JSON)",
                    value=str(cfg.get("validation_rules", "{}")),
                ).classes("w-full h-32")

                storage_input = ui.input(
                    "Storage Path", value=str(cfg.get("storage_path", "./archive"))
                ).classes("w-full")

                def apply_config() -> None:
                    updates: dict[str, Any] = {
                        "camera_exposure": int(exposure_input.value),
                        "confidence_threshold": float(threshold_input.value),
                        "validation_rules": rules_input.value,
                        "storage_path": storage_input.value,
                    }
                    save_fn = shared_state.get("save_config")
                    if callable(save_fn):
                        save_fn(updates)
                        shared_state["config"].update(updates)
                    ui.notify("Config saved", color="positive")

                ui.button("Apply", on_click=apply_config).classes("mt-2 w-full")

        # ── Feed refresh timer ────────────────────────────────────────────
        async def refresh_feed() -> None:
            frame = shared_state.get("frame")
            bbox = shared_state.get("bbox", {})
            passed = shared_state.get("passed")

            overlay = {**bbox, "passed": passed} if bbox else None
            feed_img.source = _encode_frame(frame, overlay)

            if passed is None:
                result_label.text = "—"
                result_label.classes(replace="text-xl font-bold text-gray-400")
            elif passed:
                result_label.text = "✔ PASS"
                result_label.classes(replace="text-xl font-bold text-green-600")
            else:
                result_label.text = "✘ FAIL"
                result_label.classes(replace="text-xl font-bold text-red-600")

            ocr_label.text = shared_state.get("ocr_text", "")

        ui.timer(0.5, refresh_feed)

    @ui.page("/health")
    def health_page() -> None:
        ui.label("OK").classes("text-green-600")


def start_ui(shared_state: dict[str, Any], host: str = "0.0.0.0", port: int = 8080) -> None:
    """Start the NiceGUI server (blocking call – run in a thread)."""
    if not _NICEGUI_AVAILABLE:
        logger.warning("NiceGUI not available; UI server not started")
        return
    build_ui(shared_state)
    ui.run(host=host, port=port, title="XM2 OCR", reload=False, show=False)


def start_ui_thread(
    shared_state: dict[str, Any], host: str = "0.0.0.0", port: int = 8080
) -> threading.Thread:
    """Launch the NiceGUI server in a daemon thread."""
    t = threading.Thread(
        target=start_ui, args=(shared_state, host, port), daemon=True, name="nicegui"
    )
    t.start()
    return t
