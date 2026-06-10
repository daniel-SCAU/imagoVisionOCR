"""
FastAPI service for the XM2 OCR inspection system.

Endpoints:
  POST /capture          – simulate / trigger a capture cycle
  GET  /status           – system status
  GET  /config           – read config
  POST /config           – update config
  GET  /results          – last N inspection results
  GET  /roi              – deprecated debug endpoint (kept for compatibility)

Run with::

    uvicorn app.api.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import base64
import logging
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config.settings import get_recent_results, load_config, save_config
from app.pipeline import run_pipeline

logger = logging.getLogger(__name__)

app = FastAPI(
    title="XM2 OCR Inspection API",
    description="Edge AI OCR inspection system for Imago XM2 / Jetson",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared mutable state ──────────────────────────────────────────────────────
_last_result: dict[str, Any] = {}
_pipeline_running: bool = False


# ── Pydantic models ───────────────────────────────────────────────────────────

class ConfigUpdate(BaseModel):
    updates: dict[str, Any]


class CaptureResponse(BaseModel):
    ocr_text: str
    validation_result: str
    passed: bool
    image_b64: str | None = None
    bbox: dict[str, Any] = {}
    timestamp: str = ""


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/status")
async def get_status() -> dict[str, Any]:
    """Return system readiness and last result summary."""
    return {
        "status": "running",
        "pipeline_running": _pipeline_running,
        "last_result": _last_result,
    }


@app.get("/config")
async def get_config() -> dict[str, Any]:
    """Return the current merged configuration."""
    cfg = load_config()
    # Use an allowlist for safe config keys to avoid leaking any credential variants
    _SENSITIVE_SUBSTRINGS = {"password", "key", "secret", "token", "credential"}
    safe = {
        k: v for k, v in cfg.items()
        if not any(s in k.lower() for s in _SENSITIVE_SUBSTRINGS)
    }
    return safe


@app.post("/config")
async def post_config(body: ConfigUpdate) -> dict[str, Any]:
    """Persist configuration updates to SQLite."""
    save_config(body.updates)
    return {"saved": list(body.updates.keys())}


@app.post("/capture", response_model=CaptureResponse)
async def post_capture(include_image: bool = False) -> CaptureResponse:
    """
    Trigger one inspection cycle.

    Runs the full pipeline: capture → detect → geometry → OCR → validate → archive.
    """
    global _pipeline_running, _last_result
    if _pipeline_running:
        raise HTTPException(status_code=409, detail="Pipeline already running")

    _pipeline_running = True
    try:
        cfg = load_config()
        result = run_pipeline(cfg)
        _last_result = result

        image_b64: str | None = None
        if include_image and "image" in result:
            img: np.ndarray = result["image"]
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            image_b64 = base64.b64encode(buf.tobytes()).decode()

        return CaptureResponse(
            ocr_text=result.get("ocr_text", ""),
            validation_result=str(result.get("validation_result", "")),
            passed=bool(result.get("passed", False)),
            image_b64=image_b64,
            bbox=result.get("bbox", {}),
            timestamp=result.get("timestamp", ""),
        )
    except Exception as exc:
        logger.exception("Capture pipeline error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _pipeline_running = False


@app.get("/results")
async def get_results(limit: int = 20) -> list[dict[str, Any]]:
    """Return the *limit* most recent inspection results from SQLite."""
    return get_recent_results(limit=limit)
