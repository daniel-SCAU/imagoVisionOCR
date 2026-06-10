"""
Archive / storage module.

Saves images and JSON metadata to a dated local directory tree, and
optionally syncs to a NAS via SMB (smbprotocol) or a mounted filesystem path.

Local file structure::

    <storage_path>/YYYY/MM/DD/
        image_<TIMESTAMP>_PASS.jpg
        image_<TIMESTAMP>_PASS.json
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _dated_dir(base: Path, ts: datetime) -> Path:
    return base / ts.strftime("%Y") / ts.strftime("%m") / ts.strftime("%d")


def _stem(ts: datetime, result: str) -> str:
    return f"image_{ts.strftime('%Y%m%d_%H%M%S_%f')}_{result.upper()}"


class ArchiveManager:
    """
    Handles local archiving and optional NAS synchronisation.

    config keys:
      storage_path   – local base directory (default: ./archive)
      nas_enabled    – bool, whether to push to NAS
      nas_path       – UNC path or mount point, e.g. "//nas/ocr" or "/mnt/nas/ocr"
      nas_username   – SMB username (only used when smbprotocol is available)
      nas_password   – SMB password
      image_format   – "jpg" | "png" (default: "jpg")
      jpeg_quality   – 0–100 (default: 95)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._base = Path(config.get("storage_path", "./archive")).expanduser().resolve()
        self._nas_enabled = bool(config.get("nas_enabled", False))
        self._nas_path = str(config.get("nas_path", ""))
        self._nas_user = str(config.get("nas_username", ""))
        self._nas_pass = str(config.get("nas_password", ""))
        self._fmt = str(config.get("image_format", "jpg")).lower()
        self._quality = int(config.get("jpeg_quality", 95))
        self._base.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        image: np.ndarray,
        ocr_text: str,
        validation_result: str,
        bbox: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        """
        Save image + JSON metadata to the local archive.

        Returns:
            (image_path, json_path) – absolute Paths.
        """
        ts = datetime.now(timezone.utc)
        dest_dir = _dated_dir(self._base, ts)
        dest_dir.mkdir(parents=True, exist_ok=True)

        stem = _stem(ts, validation_result)
        img_path = dest_dir / f"{stem}.{self._fmt}"
        json_path = dest_dir / f"{stem}.json"

        # Write image
        self._write_image(img_path, image)

        # Write metadata
        metadata: dict[str, Any] = {
            "timestamp": ts.isoformat(),
            "ocr_text": ocr_text,
            "validation_result": validation_result,
            "bbox": bbox,
        }
        if extra:
            metadata.update(extra)
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        logger.info("Archived: %s", img_path)

        if self._nas_enabled:
            self._sync_to_nas(img_path, json_path, ts)

        return img_path, json_path

    # ------------------------------------------------------------------
    # Image writing
    # ------------------------------------------------------------------

    def _write_image(self, path: Path, image: np.ndarray) -> None:
        if self._fmt in {"jpg", "jpeg"}:
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        else:
            cv2.imwrite(str(path), image)

    # ------------------------------------------------------------------
    # NAS sync
    # ------------------------------------------------------------------

    def _sync_to_nas(self, img_path: Path, json_path: Path, ts: datetime) -> None:
        nas = Path(self._nas_path)

        # If NAS path is a mounted filesystem, just copy files
        if nas.is_absolute() and (nas.exists() or not self._nas_user):
            try:
                dest = _dated_dir(nas, ts)
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(img_path), str(dest / img_path.name))
                shutil.copy2(str(json_path), str(dest / json_path.name))
                logger.info("Synced to NAS mount: %s", dest)
                return
            except Exception as exc:
                logger.warning("NAS mount copy failed: %s", exc)

        # SMB via smbprotocol
        self._sync_smb(img_path, json_path, ts)

    def _sync_smb(self, img_path: Path, json_path: Path, ts: datetime) -> None:
        try:
            import smbclient  # type: ignore[import]

            server = self._nas_path.lstrip("/\\").split("/")[0].split("\\")[0]
            share_and_path = self._nas_path.lstrip("/\\")[len(server):].lstrip("/\\")

            smbclient.register_session(
                server,
                username=self._nas_user or None,
                password=self._nas_pass or None,
            )
            remote_dir = f"\\\\{server}\\{share_and_path}\\"
            remote_dir += ts.strftime("%Y\\%m\\%d")

            smbclient.makedirs(remote_dir, exist_ok=True)

            for local_path in (img_path, json_path):
                remote_file = f"{remote_dir}\\{local_path.name}"
                with local_path.open("rb") as src, smbclient.open_file(remote_file, mode="wb") as dst:
                    shutil.copyfileobj(src, dst)
            logger.info("Synced to SMB: %s", remote_dir)
        except ImportError:
            logger.warning("smbprotocol not installed; skipping SMB sync")
        except Exception as exc:
            logger.error("SMB sync failed: %s", exc)
