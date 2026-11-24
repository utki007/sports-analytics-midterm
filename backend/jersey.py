"""Jersey number recognition via OCR, with graceful degradation.

OCR backends (EasyOCR / PaddleOCR) are heavy and optional. If none is
installed, this module cleanly reports numbers as unavailable ("?") instead of
failing — the rest of the pipeline is unaffected. When a backend IS present we
aggregate digit predictions across frames per track and emit a confidence.

We never hallucinate a number: unreadable or low-agreement tracks stay "?".
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_BACKEND = None
_READER = None


def _init_backend() -> Optional[str]:
    global _BACKEND, _READER
    if _BACKEND is not None:
        return _BACKEND if _BACKEND != "none" else None
    try:
        import easyocr  # type: ignore
        _READER = easyocr.Reader(["en"], gpu=False, verbose=False)
        _BACKEND = "easyocr"
        return _BACKEND
    except Exception:
        pass
    try:
        from paddleocr import PaddleOCR  # type: ignore
        _READER = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        _BACKEND = "paddleocr"
        return _BACKEND
    except Exception:
        pass
    _BACKEND = "none"
    return None


def backend_available() -> Optional[str]:
    return _init_backend()


def _preprocess(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return gray


def read_number(crop: np.ndarray) -> Optional[Tuple[str, float]]:
    """OCR a single jersey crop. Returns (digits, confidence) or None."""
    backend = _init_backend()
    if backend is None or crop is None or crop.size == 0:
        return None
    img = _preprocess(crop)
    try:
        if backend == "easyocr":
            res = _READER.readtext(img, allowlist="0123456789")
            best = None
            for _box, text, conf in res:
                t = "".join(ch for ch in text if ch.isdigit())
                if 1 <= len(t) <= 2 and (best is None or conf > best[1]):
                    best = (t, float(conf))
            return best
        else:  # paddleocr
            res = _READER.ocr(img, cls=False)
            if not res or not res[0]:
                return None
            best = None
            for line in res[0]:
                text, conf = line[1][0], line[1][1]
                t = "".join(ch for ch in text if ch.isdigit())
                if 1 <= len(t) <= 2 and (best is None or conf > best[1]):
                    best = (t, float(conf))
            return best
    except Exception:
        return None


class JerseyAggregator:
    """Collect per-frame OCR votes and resolve one number per track."""

    def __init__(self, min_votes: int = 2, min_agreement: float = 0.5):
        self.min_votes = min_votes
        self.min_agreement = min_agreement
        self._votes: Dict[int, List[Tuple[str, float]]] = defaultdict(list)

    def add(self, track_id: int, result: Optional[Tuple[str, float]]):
        if result is not None:
            self._votes[track_id].append(result)

    def resolve(self) -> Dict[int, Tuple[Optional[str], float]]:
        out: Dict[int, Tuple[Optional[str], float]] = {}
        for tid, votes in self._votes.items():
            if len(votes) < self.min_votes:
                out[tid] = (None, 0.0)
                continue
            counter = Counter(v[0] for v in votes)
            num, count = counter.most_common(1)[0]
            agreement = count / len(votes)
            avg_conf = float(np.mean([v[1] for v in votes if v[0] == num]))
            if agreement >= self.min_agreement and avg_conf > 0.3:
                out[tid] = (num, round(agreement * avg_conf, 3))
            else:
                out[tid] = (None, 0.0)
        return out
