"""
XM2 OCR Inspection System – main entry point.

Starts:
  * FastAPI inspection API (uvicorn, default port 8000)
  * NiceGUI configuration / monitoring UI (default port 8080)
  * Optional continuous pipeline loop (--loop)

Usage::

    python app/main.py [--config config.json] [--api-port 8000] [--ui-port 8080]
    python app/main.py --once                 # single capture then exit
    python app/main.py --loop --interval 2.0  # continuous every 2 s
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn

from app.api.server import app as fastapi_app
from app.config.settings import load_config, save_config
from app.pipeline import run_pipeline, shutdown_pipeline
from app.ui.interface import start_ui_thread

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Shared state between pipeline thread and UI
shared_state: dict[str, Any] = {
    "frame": None,
    "ocr_text": "",
    "passed": None,
    "bbox": {},
    "config": {},
    "save_config": save_config,
}

_stop_event = threading.Event()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XM2 Jetson OCR Inspection System")
    p.add_argument("--config", default=None, help="Path to JSON config file")
    p.add_argument("--once", action="store_true", help="Run one capture cycle and exit")
    p.add_argument("--loop", action="store_true", help="Run continuous capture loop")
    p.add_argument("--interval", type=float, default=1.0, help="Loop interval in seconds")
    p.add_argument("--api-port", type=int, default=8000)
    p.add_argument("--ui-port", type=int, default=8080)
    p.add_argument("--no-ui", action="store_true", help="Disable NiceGUI UI")
    p.add_argument("--no-api", action="store_true", help="Disable FastAPI server")
    return p.parse_args()


def _pipeline_loop(cfg: dict[str, Any], interval: float) -> None:
    """Continuous inspection loop – runs in its own thread."""
    while not _stop_event.is_set():
        try:
            result = run_pipeline(cfg)
            shared_state["ocr_text"] = result.get("ocr_text", "")
            shared_state["passed"] = result.get("passed")
            shared_state["bbox"] = result.get("bbox", {})
            shared_state["frame"] = result.get("image")
            logger.info(
                "[%s] %s – %r",
                result["timestamp"],
                result["validation_result"],
                result["ocr_text"],
            )
        except Exception as exc:
            logger.error("Pipeline error: %s", exc)
        _stop_event.wait(timeout=max(0.05, interval))


def _handle_signal(signum: int, _frame: Any) -> None:
    logger.info("Signal %d received – shutting down", signum)
    _stop_event.set()


def main() -> int:
    args = parse_args()

    json_path = Path(args.config) if args.config else None
    cfg = load_config(json_path=json_path)
    shared_state["config"] = cfg

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ── Single-shot mode ──────────────────────────────────────────────────
    if args.once:
        try:
            result = run_pipeline(cfg)
            print(
                f"[{result['timestamp']}] {result['validation_result']} – {result['ocr_text']!r}"
            )
            return 0 if result["passed"] else 1
        except Exception as exc:
            logger.error("Pipeline failed: %s", exc)
            return 2
        finally:
            shutdown_pipeline()

    # ── Start UI ──────────────────────────────────────────────────────────
    if not args.no_ui:
        start_ui_thread(shared_state, port=args.ui_port)
        logger.info("NiceGUI UI started on http://0.0.0.0:%d", args.ui_port)

    # ── Start API ─────────────────────────────────────────────────────────
    api_thread: threading.Thread | None = None
    if not args.no_api:
        api_config = uvicorn.Config(
            fastapi_app,
            host="0.0.0.0",
            port=args.api_port,
            log_level="warning",
        )
        api_server = uvicorn.Server(api_config)

        def _run_api() -> None:
            api_server.run()

        api_thread = threading.Thread(target=_run_api, daemon=True, name="fastapi")
        api_thread.start()
        logger.info("FastAPI started on http://0.0.0.0:%d", args.api_port)

    # ── Continuous loop or idle ───────────────────────────────────────────
    if args.loop:
        loop_thread = threading.Thread(
            target=_pipeline_loop,
            args=(cfg, args.interval),
            daemon=True,
            name="pipeline-loop",
        )
        loop_thread.start()
        logger.info("Continuous pipeline loop started (interval=%.1fs)", args.interval)
        loop_thread.join()
    else:
        logger.info(
            "Services running. "
            "POST http://localhost:%d/capture to trigger an inspection.",
            args.api_port,
        )
        _stop_event.wait()

    shutdown_pipeline()
    logger.info("Shutdown complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
