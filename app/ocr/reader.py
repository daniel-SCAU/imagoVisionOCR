"""
OCR module using PaddleOCR with a Tesseract fallback.

Provides :func:`extract_text` which returns a clean, uppercase string.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_NOISE_PATTERN = re.compile(r"[^A-Z0-9\-./: ]")


class OCRReader:
    """
    Wraps PaddleOCR (preferred) or pytesseract (fallback).

    config keys:
      ocr_engine   – "paddleocr" | "tesseract"
      ocr_lang     – language code, e.g. "en" (PaddleOCR) / "eng" (Tesseract)
      ocr_use_gpu  – bool, enable GPU for PaddleOCR
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._engine = str(config.get("ocr_engine", "paddleocr")).lower()
        self._lang = str(config.get("ocr_lang", "en"))
        self._use_gpu = bool(config.get("ocr_use_gpu", True))
        self._paddle: Any = None
        self._load_engine()

    # ------------------------------------------------------------------
    # Engine loading
    # ------------------------------------------------------------------

    def _load_engine(self) -> None:
        if self._engine == "paddleocr":
            self._load_paddle()
        else:
            logger.info("OCR engine: tesseract")

    def _load_paddle(self) -> None:
        try:
            from paddleocr import PaddleOCR  # type: ignore[import]

            self._paddle = PaddleOCR(
                use_angle_cls=True,
                lang=self._lang,
                use_gpu=self._use_gpu,
                show_log=False,
            )
            logger.info("PaddleOCR loaded (lang=%s, gpu=%s)", self._lang, self._use_gpu)
        except ImportError:
            logger.warning(
                "paddleocr not installed; falling back to tesseract. "
                "Install with: pip install paddlepaddle paddleocr"
            )
            self._engine = "tesseract"
        except Exception as exc:
            logger.error("PaddleOCR init error: %s; falling back to tesseract", exc)
            self._engine = "tesseract"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_text(self, image: np.ndarray) -> str:
        """
        Run OCR on *image* (BGR or grayscale numpy array).

        Returns:
            Cleaned, uppercase, single-line string.
        """
        processed = self._preprocess(image)
        if self._engine == "paddleocr" and self._paddle is not None:
            raw = self._run_paddle(processed)
        else:
            raw = self._run_tesseract(processed)
        return self._clean(raw)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        """
        Light preprocessing to improve OCR accuracy:
          * Convert to grayscale
          * CLAHE histogram equalization
          * Mild sharpening
          * Scale up small images (helps PaddleOCR)
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Laplacian-based sharpening kernel: amplifies centre pixel, suppresses neighbours
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        gray = cv2.filter2D(gray, -1, kernel)

        h, w = gray.shape
        if w < 200:
            scale = 200 / w
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        # Convert back to BGR for PaddleOCR (expects colour)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------------
    # Engine-specific runners
    # ------------------------------------------------------------------

    def _run_paddle(self, image: np.ndarray) -> str:
        try:
            result = self._paddle.ocr(image, cls=True)
            if not result or not result[0]:
                return ""
            lines: list[str] = []
            for line in result[0]:
                if line and len(line) == 2:
                    text_conf = line[1]
                    if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 1:
                        lines.append(str(text_conf[0]))
            return " ".join(lines)
        except Exception as exc:
            logger.error("PaddleOCR run error: %s", exc)
            return ""

    def _run_tesseract(self, image: np.ndarray) -> str:
        try:
            import pytesseract  # type: ignore[import]

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return pytesseract.image_to_string(gray, config="--oem 3 --psm 7")
        except ImportError:
            logger.error("pytesseract not installed")
            return ""
        except Exception as exc:
            logger.error("Tesseract run error: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Text cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(text: str) -> str:
        """
        Normalise extracted text:
          * Strip surrounding whitespace
          * Collapse internal whitespace
          * Uppercase
          * Remove noise characters (keep A-Z 0-9 - . / : space)
        """
        text = text.strip().upper()
        text = re.sub(r"\s+", " ", text)
        text = _NOISE_PATTERN.sub("", text)
        return text.strip()


# Module-level convenience function (creates a single-use reader)
def extract_text(image: np.ndarray, config: dict[str, Any] | None = None) -> str:
    """Convenience wrapper – creates an :class:`OCRReader` and returns cleaned text."""
    return OCRReader(config or {}).extract_text(image)
