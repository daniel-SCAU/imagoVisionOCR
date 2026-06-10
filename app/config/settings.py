"""Configuration management backed by SQLite with a JSON file fallback."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

_DEFAULT_CONFIG: dict[str, Any] = {
    "camera_source": 0,
    "camera_backend": "any",
    "camera_width": 1920,
    "camera_height": 1080,
    "camera_fps": 30,
    "camera_exposure": 2000,
    "ocr_engine": "paddleocr",
    "detection_model": "roboflow",
    "roboflow_model_path": "",
    "roboflow_project": "",
    "roboflow_version": 1,
    "yolo_weights": "yolov8n.pt",
    "confidence_threshold": 0.5,
    "storage_path": "./archive",
    "nas_enabled": False,
    "nas_path": "//nas/ocr",
    "nas_username": "",
    "nas_password": "",
    "validation_rules": "{}",
    "gstreamer_pipeline": "",
}

_DB_PATH = Path("xm2_ocr_config.db")


def _get_connection(db_path: Path = _DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS config (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_utc TEXT NOT NULL,
            ocr_text      TEXT,
            validation    TEXT,
            bbox          TEXT,
            image_path    TEXT
        )
        """
    )
    conn.commit()


def load_config(db_path: Path = _DB_PATH, json_path: Path | None = None) -> dict[str, Any]:
    """Return the merged config: defaults → JSON file (optional) → SQLite overrides."""
    cfg: dict[str, Any] = dict(_DEFAULT_CONFIG)

    if json_path and json_path.exists():
        with json_path.open("r", encoding="utf-8") as fh:
            cfg.update(json.load(fh))

    conn = _get_connection(db_path)
    _ensure_table(conn)
    rows = conn.execute("SELECT key, value FROM config").fetchall()
    conn.close()
    for row in rows:
        try:
            cfg[row["key"]] = json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            cfg[row["key"]] = row["value"]

    return cfg


def save_config(updates: dict[str, Any], db_path: Path = _DB_PATH) -> None:
    """Persist key/value pairs to SQLite."""
    conn = _get_connection(db_path)
    _ensure_table(conn)
    for key, value in updates.items():
        conn.execute(
            "INSERT INTO config (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )
    conn.commit()
    conn.close()


def save_result(
    timestamp_utc: str,
    ocr_text: str,
    validation: str,
    bbox: dict[str, Any],
    image_path: str,
    db_path: Path = _DB_PATH,
) -> None:
    conn = _get_connection(db_path)
    _ensure_table(conn)
    conn.execute(
        "INSERT INTO results (timestamp_utc, ocr_text, validation, bbox, image_path) "
        "VALUES (?, ?, ?, ?, ?)",
        (timestamp_utc, ocr_text, validation, json.dumps(bbox), image_path),
    )
    conn.commit()
    conn.close()


def get_recent_results(limit: int = 20, db_path: Path = _DB_PATH) -> list[dict[str, Any]]:
    conn = _get_connection(db_path)
    _ensure_table(conn)
    rows = conn.execute(
        "SELECT * FROM results ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
