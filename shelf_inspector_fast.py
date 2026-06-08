"""Core image inspection pipeline for the library shelf OCR project.

This module contains the main non-Web logic: image rotation, red-label and
red-band detection, crop generation, official PaddleOCR recognition,
call-number normalization, shelf-order checking, diagnostics export, and result
cache loading for the Web service.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CUSTOM_REC_MODEL_ENV = "BOOK_OCR_REC_MODEL_DIR"


# Data objects shared by detection, OCR, sorting, report export, and Web JSON.
@dataclass
class CropOcrAttempt:
    variant: str
    crop_box: tuple[int, int, int, int]
    crop_path: Path | None = None
    raw_text: str = ""
    clean_text: str = ""
    confidence: float = 0.0
    parse_ok: bool = False
    score: float = 0.0
    selected: bool = False


@dataclass
class Detection:
    index: int
    red_box: tuple[int, int, int, int]
    crop_box: tuple[int, int, int, int]
    crop_path: Path | None = None
    raw_text: str = ""
    clean_text: str = ""
    confidence: float = 0.0
    parse_ok: bool = False
    status: str = "yellow"
    reason: str = ""
    recommended_position: int | None = None
    ocr_attempts: list[CropOcrAttempt] = field(default_factory=list)

    @property
    def center_y(self) -> float:
        x, y, w, h = self.red_box
        return y + h / 2

    @property
    def center_x(self) -> float:
        x, y, w, h = self.crop_box
        return x + w / 2


@dataclass
class OcrLine:
    text: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass
class OcrCandidate:
    raw_text: str
    clean_text: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass
class ImageRunResult:
    image_path: Path
    output_dir: Path
    detections: list[Detection]
    ocr_enabled: bool
    rotate_mode: str = ""
    elapsed_seconds: float = 0.0
    result_dir: Path | None = None
    from_cache: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def annotated_path(self) -> Path:
        return (self.result_dir or self.output_dir / self.image_path.stem) / "annotated.jpg"

    @property
    def summary_path(self) -> Path:
        return (self.result_dir or self.output_dir / self.image_path.stem) / "summary.json"


# Basic image I/O and orientation helpers.
def read_image(path: Path) -> np.ndarray:
    """Read image safely on Windows paths that may contain non-ASCII chars."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    """Write image safely on Windows paths that may contain non-ASCII chars."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".jpg"
    ok, data = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    data.tofile(str(path))


def rotate_image(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "left":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mode == "right":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if mode == "none":
        return image
    raise ValueError(f"Unknown rotate mode: {mode}")


def score_orientation(image: np.ndarray, mode: str, max_side: int) -> float:
    rotated = rotate_image(image, mode)
    working, _ = resize_to_max_side(rotated, max_side)
    mask = build_red_mask(working)
    boxes = detect_red_label_boxes(mask)
    column = find_main_red_column(mask)

    score = min(len(boxes), 35) * 1.5
    if column is None:
        return score

    x0, x1 = column
    width_ratio = (x1 - x0) / max(1, working.shape[1])
    column_mask = mask[:, x0:x1]
    row_has_red = np.any(column_mask > 0, axis=1)
    height_ratio = float(np.count_nonzero(row_has_red)) / max(1, working.shape[0])

    # A useful call-number strip usually has a narrow red marker column that spans much of the shelf.
    score += max(0.0, 80.0 - abs(width_ratio - 0.06) * 260.0)
    score += height_ratio * 40.0
    if width_ratio > 0.20:
        score -= 35.0
    return score


def choose_rotation(image: np.ndarray, max_side: int) -> str:
    candidates = ["none", "left", "right"]
    scores = {mode: score_orientation(image, mode, max_side) for mode in candidates}
    return max(candidates, key=lambda mode: scores[mode])


def resize_to_max_side(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / longest
    resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


# Red-label and red-band detection. These functions find visual anchors used to
# crop call numbers from horizontal shelf photos.
def build_red_mask(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Red wraps around the hue axis in HSV, so it needs two ranges.
    lower_red_1 = np.array([0, 45, 45])
    upper_red_1 = np.array([12, 255, 255])
    lower_red_2 = np.array([168, 45, 45])
    upper_red_2 = np.array([180, 255, 255])

    mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask_1, mask_2)

    h, w = mask.shape[:2]
    k = max(3, int(min(h, w) * 0.003))
    if k % 2 == 0:
        k += 1

    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    return mask


def find_main_red_column(mask: np.ndarray) -> tuple[int, int] | None:
    h, _ = mask.shape[:2]
    col_counts = np.count_nonzero(mask, axis=0)
    threshold = max(int(h * 0.02), int(col_counts.max() * 0.25))
    flags = col_counts > threshold
    segments = find_segments(flags, min_length=3, gap=4)
    if not segments:
        return None

    def score(segment: tuple[int, int]) -> int:
        x0, x1 = segment
        return int(col_counts[x0:x1].sum())

    return max(segments, key=score)


def boxes_from_main_red_column(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    column = find_main_red_column(mask)
    if column is None:
        return []

    h, w = mask.shape[:2]
    x0, x1 = column
    pad_x = max(2, int((x1 - x0) * 0.15))
    x0 = max(0, x0 - pad_x)
    x1 = min(w, x1 + pad_x)

    roi = mask[:, x0:x1]
    row_counts = np.count_nonzero(roi, axis=1)
    threshold = max(2, int((x1 - x0) * 0.08))
    flags = row_counts > threshold

    min_length = max(10, int(h * 0.01))
    gap = max(2, int(h * 0.003))
    segments = find_segments(flags, min_length=min_length, gap=gap)

    boxes: list[tuple[int, int, int, int]] = []
    for y0, y1 in segments:
        sub = roi[y0:y1, :]
        points = cv2.findNonZero(sub)
        if points is None:
            continue
        sx, sy, sw, sh = cv2.boundingRect(points)
        if sh < min_length:
            continue
        boxes.append((x0 + sx, y0 + sy, sw, sh))
    return boxes


def find_segments(flags: np.ndarray, min_length: int, gap: int) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    last_true: int | None = None

    for i, value in enumerate(flags):
        if value:
            if start is None:
                start = i
            last_true = i
        elif start is not None and last_true is not None and i - last_true > gap:
            if last_true - start + 1 >= min_length:
                segments.append((start, last_true + 1))
            start = None
            last_true = None

    if start is not None and last_true is not None and last_true - start + 1 >= min_length:
        segments.append((start, last_true + 1))

    return segments


def split_tall_red_box(mask: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    x, y, w, h = box
    roi = mask[y : y + h, x : x + w]
    row_counts = np.count_nonzero(roi, axis=1)
    threshold = max(4, int(w * 0.08))
    flags = row_counts > threshold
    min_length = max(8, int(mask.shape[0] * 0.006))
    gap = max(2, int(mask.shape[0] * 0.002))
    segments = find_segments(flags, min_length=min_length, gap=gap)

    if len(segments) <= 1:
        return [box]

    split_boxes: list[tuple[int, int, int, int]] = []
    for y0, y1 in segments:
        sub = roi[y0:y1, :]
        points = cv2.findNonZero(sub)
        if points is None:
            continue
        sx, sy, sw, sh = cv2.boundingRect(points)
        split_boxes.append((x + sx, y + y0 + sy, sw, sh))
    return split_boxes or [box]


def merge_overlapping_boxes(boxes: Iterable[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    sorted_boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged: list[tuple[int, int, int, int]] = []

    for box in sorted_boxes:
        x, y, w, h = box
        if not merged:
            merged.append(box)
            continue

        px, py, pw, ph = merged[-1]
        y_overlap = min(y + h, py + ph) - max(y, py)
        close_y = abs((y + h / 2) - (py + ph / 2)) < max(h, ph) * 0.45
        close_x = abs((x + w / 2) - (px + pw / 2)) < max(w, pw) * 2.5

        if (y_overlap > 0 or close_y) and close_x:
            nx0 = min(px, x)
            ny0 = min(py, y)
            nx1 = max(px + pw, x + w)
            ny1 = max(py + ph, y + h)
            merged[-1] = (nx0, ny0, nx1 - nx0, ny1 - ny0)
        else:
            merged.append(box)

    return merged


def detect_red_label_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = max(80, int(h * w * 0.000015))
    min_w = max(3, int(w * 0.002))
    min_h = max(6, int(h * 0.002))

    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if area < min_area or bw < min_w or bh < min_h:
            continue
        if bw > w * 0.7 or bh > h * 0.45:
            continue
        boxes.extend(split_tall_red_box(mask, (x, y, bw, bh)))

    column_boxes = boxes_from_main_red_column(mask)
    if len(column_boxes) > len(boxes):
        boxes = column_boxes

    boxes = merge_overlapping_boxes(boxes)
    boxes = [b for b in boxes if b[2] * b[3] >= min_area]
    return boxes


def make_crop_box(
    image_shape: tuple[int, int, int],
    red_box: tuple[int, int, int, int],
    crop_right_ratio: float,
    y_padding_ratio: float,
) -> tuple[int, int, int, int] | None:
    h, w = image_shape[:2]
    x, y, bw, bh = red_box

    pad_y = max(4, int(bh * y_padding_ratio))
    crop_x0 = min(w - 1, x + bw)
    crop_w = max(int(w * crop_right_ratio), bw * 4)
    crop_x1 = min(w, crop_x0 + crop_w)
    crop_y0 = max(0, y - pad_y)
    crop_y1 = min(h, y + bh + pad_y)

    if crop_x1 - crop_x0 < 20 or crop_y1 - crop_y0 < 8:
        return None
    return (crop_x0, crop_y0, crop_x1 - crop_x0, crop_y1 - crop_y0)


def save_crops(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    out_dir: Path,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        crop = image[y : y + h, x : x + w]
        path = out_dir / f"crop_{i:03d}.jpg"
        write_image(path, crop)
        paths.append(path)
    return paths


# OCR model management. The final Web version calls load_paddle_ocr(use_env=False)
# so experimental recognition models are not accidentally used.
def resolve_rec_model_dir(
    rec_model_dir: str | os.PathLike[str] | None = None,
    *,
    use_env: bool = True,
) -> Path | None:
    source = rec_model_dir if rec_model_dir is not None else os.environ.get(CUSTOM_REC_MODEL_ENV, "") if use_env else ""
    raw = str(source).strip().strip('"')
    if not raw:
        return None

    path = Path(raw).expanduser().absolute()
    nested = path / "inference"
    if not (path / "inference.yml").exists() and (nested / "inference.yml").exists():
        path = nested

    has_params = (path / "inference.pdiparams").exists()
    has_model = (path / "inference.json").exists() or (path / "inference.pdmodel").exists()
    if not (path / "inference.yml").exists() or not has_params or not has_model:
        raise FileNotFoundError(
            "Invalid OCR recognition model directory. Expected inference.yml, "
            f"inference.pdiparams and inference.json/pdmodel under: {path}"
        )
    return path


def ocr_model_cache_tag(
    rec_model_dir: str | os.PathLike[str] | None = None,
    *,
    use_env: bool = True,
) -> str:
    path = resolve_rec_model_dir(rec_model_dir, use_env=use_env)
    if path is None:
        return "rec_official"

    params = path / "inference.pdiparams"
    try:
        stat = params.stat()
        payload = f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        payload = str(path)
    digest = hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"rec_{digest}"


def load_paddle_ocr(
    rec_model_dir: str | os.PathLike[str] | None = None,
    *,
    use_env: bool = True,
) -> Any | None:
    try:
        from paddleocr import PaddleOCR
    except Exception:
        return None

    local_model_base = Path.home() / ".paddlex" / "official_models"
    local_mobile_det = local_model_base / "PP-OCRv5_mobile_det"
    local_mobile_rec = local_model_base / "PP-OCRv5_mobile_rec"
    custom_mobile_rec = resolve_rec_model_dir(rec_model_dir, use_env=use_env)
    active_mobile_rec = custom_mobile_rec or local_mobile_rec

    candidates = [
        {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "PP-OCRv5_mobile_rec",
            "text_detection_model_dir": str(local_mobile_det),
            "text_recognition_model_dir": str(active_mobile_rec),
            "text_det_limit_side_len": 960,
            "enable_mkldnn": False,
            "enable_hpi": False,
            "device": "cpu",
        }
    ]
    if custom_mobile_rec is None:
        candidates.extend(
            [
                {"use_textline_orientation": True, "lang": "ch"},
                {"use_angle_cls": True, "lang": "ch"},
                {"lang": "ch"},
            ]
        )
    last_error: Exception | None = None
    for kwargs in candidates:
        try:
            return PaddleOCR(**kwargs)
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    return None


# PaddleOCR result parsing. Different PaddleOCR versions return different nested
# shapes, so these helpers reduce them to text/confidence/box records.
def flatten_ocr_result(result: Any) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []

    def walk(node: Any) -> None:
        if isinstance(node, (list, tuple)):
            if len(node) >= 2 and isinstance(node[-1], (list, tuple)):
                maybe_text = node[-1][0] if len(node[-1]) > 0 else None
                maybe_conf = node[-1][1] if len(node[-1]) > 1 else None
                if isinstance(maybe_text, str):
                    try:
                        conf = float(maybe_conf)
                    except Exception:
                        conf = 0.0
                    items.append((maybe_text, conf))
                    return
            for child in node:
                walk(child)

    walk(result)
    return items


def run_ocr(ocr: Any | None, crop_path: Path) -> tuple[str, float]:
    if ocr is None:
        return "", 0.0

    try:
        result = ocr.predict(str(crop_path))
    except Exception:
        result = ocr.ocr(str(crop_path), cls=True)
    return ocr_text_confidence_from_result(result)


def ocr_text_confidence_from_result(result: Any) -> tuple[str, float]:
    lines = extract_ocr_lines(result)
    if lines:
        text = " ".join(line.text for line in lines)
        confidence = sum(line.confidence for line in lines) / len(lines)
        return text, confidence

    items = flatten_ocr_result(result)
    if not items:
        return "", 0.0

    text = " ".join(item[0] for item in items)
    confidence = sum(item[1] for item in items) / len(items)
    return text, confidence


def run_ocr_batch(ocr: Any | None, crop_paths: list[Path]) -> list[tuple[str, float]]:
    if ocr is None:
        return [("", 0.0) for _ in crop_paths]
    if not crop_paths:
        return []

    try:
        results = ocr.predict([str(path) for path in crop_paths])
    except Exception:
        return [run_ocr(ocr, path) for path in crop_paths]

    if isinstance(results, dict):
        if len(crop_paths) == 1:
            return [ocr_text_confidence_from_result(results)]
        return [run_ocr(ocr, path) for path in crop_paths]
    if not isinstance(results, list):
        try:
            results = list(results)
        except Exception:
            return [run_ocr(ocr, path) for path in crop_paths]

    if len(results) != len(crop_paths):
        return [run_ocr(ocr, path) for path in crop_paths]
    return [ocr_text_confidence_from_result(result) for result in results]


def is_ocr_box(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return False
    return True


def extract_ocr_lines(result: Any) -> list[OcrLine]:
    lines: list[OcrLine] = []

    def add_v3_result(node: dict[str, Any]) -> bool:
        texts = node.get("rec_texts")
        scores = node.get("rec_scores")
        boxes = node.get("rec_boxes")
        polys = node.get("rec_polys") or node.get("dt_polys")
        if not isinstance(texts, list):
            return False

        for i, text in enumerate(texts):
            if not isinstance(text, str):
                continue
            try:
                confidence = float(scores[i]) if scores is not None else 0.0
            except Exception:
                confidence = 0.0

            box: tuple[int, int, int, int] | None = None
            if boxes is not None and i < len(boxes):
                raw_box = boxes[i]
                try:
                    x0, y0, x1, y1 = [int(v) for v in raw_box]
                    box = (x0, y0, x1 - x0, y1 - y0)
                except Exception:
                    box = None
            if box is None and polys is not None and i < len(polys):
                poly = polys[i]
                try:
                    xs = [int(point[0]) for point in poly]
                    ys = [int(point[1]) for point in poly]
                    x0, y0 = min(xs), min(ys)
                    x1, y1 = max(xs), max(ys)
                    box = (x0, y0, x1 - x0, y1 - y0)
                except Exception:
                    box = None

            if box is not None:
                lines.append(OcrLine(text=text, confidence=confidence, box=box))
        return bool(lines)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if add_v3_result(node):
                return
            for child in node.values():
                walk(child)
            return

        if not isinstance(node, (list, tuple)):
            return

        if len(node) >= 2 and is_ocr_box(node[0]) and isinstance(node[1], (list, tuple)):
            text = node[1][0] if len(node[1]) > 0 else None
            confidence = node[1][1] if len(node[1]) > 1 else 0.0
            if isinstance(text, str):
                xs = [int(point[0]) for point in node[0]]
                ys = [int(point[1]) for point in node[0]]
                x0, y0 = min(xs), min(ys)
                x1, y1 = max(xs), max(ys)
                lines.append(OcrLine(text=text, confidence=float(confidence), box=(x0, y0, x1 - x0, y1 - y0)))
                return

        for child in node:
            walk(child)

    walk(result)
    return lines


def run_ocr_lines(ocr: Any | None, image_path: Path) -> list[OcrLine]:
    if ocr is None:
        return []
    try:
        result = ocr.predict(str(image_path))
    except Exception:
        result = ocr.ocr(str(image_path), cls=True)
    return extract_ocr_lines(result)


# Orientation selection combines fast red-label geometry with a small OCR check
# only when the image is ambiguous.
def score_orientation_with_ocr(
    image: np.ndarray,
    mode: str,
    max_side: int,
    crop_right_ratio: float,
    ocr: Any,
    temp_dir: Path,
) -> float:
    rotated = rotate_image(image, mode)
    working, _ = resize_to_max_side(rotated, max_side)
    mask = build_red_mask(working)
    red_column = find_main_red_column(mask)
    strip_box = make_code_strip_box(working.shape, red_column, crop_right_ratio)
    fast_score = score_orientation(image, mode, max_side)
    if strip_box is None:
        return fast_score * 0.05

    sx, sy, sw, sh = strip_box
    strip = working[sy : sy + sh, sx : sx + sw]
    strip_path = temp_dir / f"orientation_{mode}.jpg"
    write_image(strip_path, strip)
    lines = run_ocr_lines(ocr, strip_path)

    clean_values = [normalize_ocr_text(line.text) for line in lines]
    valid = [value for value in clean_values if parse_call_number(value) is not None]
    confidences = [line.confidence for line in lines if line.confidence > 0]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    strip_score = len(valid) * 50.0 + avg_confidence * 30.0 + min(len(lines), 40) * 1.5 + fast_score * 0.05

    if mode != "none":
        return strip_score

    band_box = make_horizontal_code_band_box(working.shape, mask) or make_bottom_code_band_box(working.shape)
    if band_box is None:
        return strip_score
    bx, by, bw, bh = band_box
    band = working[by : by + bh, bx : bx + bw]
    band_path = temp_dir / "orientation_none_bottom_band.jpg"
    write_image(band_path, band)
    band_lines = run_ocr_lines(ocr, band_path)
    band_candidates = build_ocr_candidates(band_lines, allow_recovery=False)
    band_valid = [candidate for candidate in band_candidates if parse_call_number(candidate.clean_text) is not None]
    band_confidences = [candidate.confidence for candidate in band_candidates if candidate.confidence > 0]
    band_avg_confidence = sum(band_confidences) / len(band_confidences) if band_confidences else 0.0
    band_score = len(band_valid) * 55.0 + band_avg_confidence * 30.0 + min(len(band_lines), 40) * 1.5
    return max(strip_score, band_score)


def choose_rotation_with_ocr(
    image: np.ndarray,
    max_side: int,
    crop_right_ratio: float,
    ocr: Any | None,
) -> str:
    # Fast edition: avoid running OCR three times just to choose orientation.
    # For wide photos, only run one quick bottom-band OCR when red-label geometry is ambiguous.
    h, w = image.shape[:2]
    none_score = score_orientation(image, "none", max_side)
    left_score = score_orientation(image, "left", max_side)
    right_score = score_orientation(image, "right", max_side)

    if w >= h * 1.15:
        working, _ = resize_to_max_side(image, max_side)
        mask = build_red_mask(working)
        band_box = make_horizontal_code_band_box(working.shape, mask)
        if band_box is not None:
            _, by, _, bh = band_box
            band_density = float(np.count_nonzero(mask[by : by + bh, :])) / max(1, bh * working.shape[1])
            if band_density >= 0.03:
                return "none"

    if w >= h * 1.15 and none_score >= 90:
        return "none"

    if w >= h * 1.15 and ocr is not None:
        working, _ = resize_to_max_side(image, max_side)
        mask = build_red_mask(working)
        band_box = make_horizontal_code_band_box(working.shape, mask) or make_bottom_code_band_box(working.shape)
        if band_box is not None:
            bx, by, bw, bh = band_box
            with tempfile.TemporaryDirectory() as tmp:
                band_path = Path(tmp) / "orientation_bottom_band.jpg"
                write_image(band_path, working[by : by + bh, bx : bx + bw])
                band_lines = run_ocr_lines(ocr, band_path)
            band_candidates = build_ocr_candidates(band_lines, allow_recovery=False)
            valid_count = sum(1 for item in band_candidates if parse_call_number(item.clean_text) is not None)
            if valid_count >= 6:
                return "none"

    scores = {"left": left_score, "right": right_score, "none": none_score}
    return max(scores, key=lambda mode: scores[mode])


# OCR text normalization and call-number parsing. Common OCR confusions such as
# I/1, O/0, slash variants, and missing decimal points are fixed here.
def normalize_ocr_text(text: str) -> str:
    text = text.upper()
    replacements = {
        " ": "",
        "\t": "",
        "／": "/",
        "\\": "/",
        "|": "/",
        "丨": "/",
        "—": "-",
        "–": "-",
        "_": "-",
        "．": ".",
        "。": ".",
        "，": "",
        ",": "",
        "：": ":",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^A-Z0-9./:-]", "", text)
    match = re.search(r"[A-Z][A-Z0-9.-]*/[A-Z0-9:-]+", text)
    if match:
        text = match.group(0)
    text = re.sub(r"^R(?=[A-Z][0-9])", "", text)

    # Common OCR fixes in the classification-number part before '/'.
    if "/" in text:
        left, right = text.split("/", 1)
        left = re.sub(r"^II(?=\d)", "H", left)
        contaminated_left = re.fullmatch(r"[A-Z]+?(D\d+(?:\.\d+)?(?:-\d+)?)", left)
        if contaminated_left and not left.startswith("D"):
            left = contaminated_left.group(1)
        fixed_left = []
        for i, ch in enumerate(left):
            if i > 0 and ch == "O":
                fixed_left.append("0")
            else:
                fixed_left.append(ch)
        left = "".join(fixed_left)
        left = re.sub(r"^1(?=2\d{2}(?:\d|\.|-|$))", "I", left)
        right = re.sub(r"^II(?=[A-Z0-9]|$)", "H", right)
        if right and right[0].isdigit():
            first_digit_as_letter = {
                "0": "O",
                "1": "L",
                "2": "Z",
                "4": "A",
                "5": "S",
                "6": "G",
                "7": "T",
                "8": "B",
                "9": "Q",
            }
            right = first_digit_as_letter.get(right[0], right[0]) + right[1:]
        if len(right) >= 2 and right[0] == "0" and right[1].isalpha():
            right = "O" + right[1:]
        right = re.sub(r"(?<=[A-Z])0(?=[A-Z])", "O", right)
        right = re.sub(r"-+$", "", right)
        decimal_match = re.fullmatch(r"([A-Z]+)(\d{3})(\d{1,2})", left)
        if decimal_match:
            cls, major, decimal = decimal_match.groups()
            left = f"{cls}{major}.{decimal}"
        dash_decimal_match = re.fullmatch(r"([A-Z]+\d{3})-(\d)", left)
        if dash_decimal_match:
            cls, decimal = dash_decimal_match.groups()
            left = f"{cls}.{decimal}"
        text = left + "/" + right

    return text


def natural_parts(value: str) -> tuple[tuple[int, Any], ...]:
    parts: list[tuple[int, Any]] = []
    for item in re.findall(r"\d+|[A-Z]+|[:-]", value):
        if item.isdigit():
            parts.append((0, int(item)))
        else:
            parts.append((1, item))
    return tuple(parts)


def parse_call_number(value: str) -> tuple[Any, ...] | None:
    match = re.match(r"^([A-Z]+)([0-9]+(?:\.[0-9]+)?)(?:-([0-9]+))?(?:/([A-Z][A-Z0-9:-]*))?$", value)
    if not match:
        return None

    letter, number, aux, suffix = match.groups()
    if letter == "II":
        return None
    if suffix and suffix.startswith("II"):
        return None
    number_parts = tuple(int(part) for part in number.split("."))
    aux_parts = tuple(int(part) for part in aux.split(".")) if aux else tuple()
    suffix_parts = natural_parts(suffix or "")

    return (
        letter,
        number_parts,
        1 if aux else 0,
        aux_parts,
        suffix_parts,
    )


def split_call_number_suffix(value: str) -> tuple[str, str] | None:
    if parse_call_number(value) is None or "/" not in value:
        return None
    prefix, suffix = value.rsplit("/", 1)
    return f"{prefix}/", suffix


def call_number_order_key(value: str) -> tuple[Any, ...] | None:
    parsed = parse_call_number(value)
    if parsed is None:
        return None

    letter, number_parts, _has_aux, _aux_parts, _suffix_parts = parsed
    # Prototype sorting focuses on CLC classification order. Real shelves may not
    # keep the suffix/cutter code in strict alphabetical order, as shown by the
    # newly supplied correct samples. Auxiliary-number groups are narrower, so
    # their suffix order remains useful for catching obvious local inversions.
    number_key = clc_number_hierarchy_key(value, number_parts)
    return (letter, number_key)


def clc_number_hierarchy_key(value: str, fallback_parts: tuple[int, ...]) -> tuple[Any, ...]:
    match = re.match(r"^[A-Z]+([0-9]+(?:\.[0-9]+)?)", value)
    if not match:
        return (fallback_parts[:1],)
    number = match.group(1)
    main, *decimal_parts = number.split(".")
    main_key = tuple(int(char) for char in main if char.isdigit())
    decimal_key = tuple(tuple(int(char) for char in part if char.isdigit()) for part in decimal_parts)
    return (main_key, decimal_key)


def local_aux_order_key(value: str) -> tuple[tuple[Any, ...], tuple[Any, ...]] | None:
    parsed = parse_call_number(value)
    if parsed is None:
        return None
    letter, number_parts, has_aux, aux_parts, suffix_parts = parsed
    if not has_aux:
        return None
    return (letter, number_parts, aux_parts), suffix_parts


# Shelf-order status rules. Green means accepted order, yellow means review, and
# red means a likely reorder suggestion.
def apply_sort_status(detections: list[Detection], confidence_threshold: float) -> None:
    parsed = [
        d
        for d in detections
        if d.parse_ok and "/" in d.clean_text and d.confidence >= confidence_threshold
    ]
    sorted_texts = [d.clean_text for d in sorted(parsed, key=lambda d: call_number_order_key(d.clean_text))]

    recommended_positions: dict[str, list[int]] = {}
    for pos, text in enumerate(sorted_texts, start=1):
        recommended_positions.setdefault(text, []).append(pos)

    used_positions: dict[str, int] = {}
    for actual_pos, detection in enumerate(detections, start=1):
        if not detection.clean_text:
            detection.status = "yellow"
            detection.reason = "OCR 未识别到内容"
            continue

        if detection.reason.startswith("根据相邻薄书编号"):
            detection.status = "yellow"
            detection.reason = "根据相邻薄书编号推测，建议人工复核"
            continue

        if detection.confidence < confidence_threshold:
            detection.status = "yellow"
            detection.reason = "OCR 置信度较低"
            continue

        if not detection.parse_ok:
            detection.status = "yellow"
            detection.reason = "索书号格式不符合规则"
            continue

        if "/" not in detection.clean_text:
            detection.status = "yellow"
            detection.reason = "索书号不完整，缺少辅助号"
            continue

        candidates = recommended_positions.get(detection.clean_text, [])
        cursor = used_positions.get(detection.clean_text, 0)
        recommended = candidates[cursor] if cursor < len(candidates) else None
        used_positions[detection.clean_text] = cursor + 1
        detection.recommended_position = recommended

        if recommended is None:
            detection.status = "yellow"
            detection.reason = "未找到推荐位置"
        elif recommended == actual_pos:
            detection.status = "green"
            detection.reason = "排序正确"
        else:
            detection.status = "red"
            detection.reason = f"推荐位置应为 {recommended}"

def longest_nondecreasing_indices(keys: list[tuple[Any, ...]]) -> set[int]:
    if not keys:
        return set()

    lengths = [1] * len(keys)
    previous = [-1] * len(keys)
    best_index = 0
    for i in range(len(keys)):
        for j in range(i):
            if keys[j] <= keys[i] and lengths[j] + 1 > lengths[i]:
                lengths[i] = lengths[j] + 1
                previous[i] = j
        if lengths[i] > lengths[best_index]:
            best_index = i

    indices: set[int] = set()
    cursor = best_index
    while cursor != -1:
        indices.add(cursor)
        cursor = previous[cursor]
    return indices


def apply_sort_status(detections: list[Detection], confidence_threshold: float) -> None:
    parsed = [
        d
        for d in detections
        if d.parse_ok and "/" in d.clean_text and d.confidence >= confidence_threshold
    ]
    sorted_texts = [d.clean_text for d in sorted(parsed, key=lambda d: call_number_order_key(d.clean_text))]
    parsed_positions = {id(detection): index for index, detection in enumerate(parsed)}
    stable_indices = longest_nondecreasing_indices(
        [call_number_order_key(d.clean_text) or tuple() for d in parsed]
    )

    recommended_positions: dict[str, list[int]] = {}
    for pos, text in enumerate(sorted_texts, start=1):
        recommended_positions.setdefault(text, []).append(pos)

    used_positions: dict[str, int] = {}
    for detection in detections:
        detection.recommended_position = None

    for actual_pos, detection in enumerate(detections, start=1):
        if not detection.clean_text:
            detection.status = "yellow"
            detection.reason = "OCR 未识别到内容"
            continue

        if detection.reason.startswith("根据相邻薄书编号"):
            detection.status = "yellow"
            detection.reason = "根据相邻薄书编号推测，建议人工复核"
            continue

        if detection.confidence < confidence_threshold:
            detection.status = "yellow"
            detection.reason = "OCR 置信度较低"
            continue

        if not detection.parse_ok:
            detection.status = "yellow"
            detection.reason = "索书号格式不符合规则"
            continue

        if "/" not in detection.clean_text:
            detection.status = "yellow"
            detection.reason = "索书号不完整，缺少辅助号"
            continue

        candidates = recommended_positions.get(detection.clean_text, [])
        cursor = used_positions.get(detection.clean_text, 0)
        recommended = candidates[cursor] if cursor < len(candidates) else None
        used_positions[detection.clean_text] = cursor + 1
        detection.recommended_position = recommended
        parsed_index = parsed_positions.get(id(detection))
        is_stable = parsed_index is not None and parsed_index in stable_indices

        if recommended is None:
            detection.status = "yellow"
            detection.reason = "未找到推荐位置"
        elif is_stable:
            detection.status = "green"
            detection.reason = "排序正确"
        else:
            detection.status = "red"
            detection.reason = f"推荐位置应为 {recommended}"

    previous_aux: tuple[tuple[Any, ...], tuple[Any, ...], Detection, int] | None = None
    for actual_pos, detection in enumerate(detections, start=1):
        if not detection.parse_ok or detection.confidence < confidence_threshold:
            continue
        aux_key = local_aux_order_key(detection.clean_text)
        if aux_key is None:
            previous_aux = None
            continue
        prefix, suffix_key = aux_key
        if previous_aux is not None:
            previous_prefix, previous_suffix, previous_detection, previous_pos = previous_aux
            if previous_prefix == prefix and previous_suffix > suffix_key:
                previous_detection.status = "red"
                previous_detection.reason = f"同一辅助号组内建议放到第 {actual_pos} 本之后"
                previous_detection.recommended_position = actual_pos
                detection.status = "red"
                detection.reason = f"同一辅助号组内建议放到第 {previous_pos} 本之前"
                detection.recommended_position = previous_pos
        previous_aux = (prefix, suffix_key, detection, actual_pos)


def downgrade_uncertain_sort_status(detections: list[Detection]) -> None:
    if not detections:
        return

    invalid_count = sum(1 for d in detections if not d.parse_ok)
    confidences = [d.confidence for d in detections if d.confidence > 0]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    invalid_ratio = invalid_count / len(detections)

    if invalid_ratio <= 0.25 and avg_confidence >= 0.85:
        return

    for detection in detections:
        if detection.status == "red":
            detection.status = "yellow"
            detection.reason = "图像质量或 OCR 结果不确定，建议人工复核"


def mark_detection_yellow(detection: Detection, reason: str) -> None:
    if detection.status == "red":
        return
    detection.status = "yellow"
    detection.reason = reason
    detection.recommended_position = None


def mark_uncertain_horizontal_detections(detections: list[Detection], image_width: int) -> None:
    if not detections:
        return

    widths = [float(item.crop_box[2]) for item in detections if item.crop_box[2] > 0]
    typical_w = median(widths) or 32.0

    for index, detection in enumerate(detections):
        parts = split_call_number_suffix(detection.clean_text)
        if parts is None:
            continue
        prefix, suffix = parts
        x, _, w, _ = detection.crop_box
        edge_pad = max(5.0, image_width * 0.006)

        if x + w >= image_width - edge_pad and (w < typical_w * 0.88 or len(suffix) <= 2):
            mark_detection_yellow(detection, "边缘 crop 可能截断书号，建议人工复核")
            continue

        if w > max(typical_w * 2.35, typical_w + 45):
            mark_detection_yellow(detection, "crop 宽度异常，可能一次框入多个书号，建议人工复核")
            continue

        if len(suffix) <= 1:
            mark_detection_yellow(detection, "辅助号过短，疑似 OCR 截断，建议人工复核")
            continue
        if "[context short suffix corrected:" in detection.raw_text:
            mark_detection_yellow(detection, "上下文修正的短辅助号，建议人工复核")
            continue
        prefix_body = prefix.rstrip("/")
        if "." in prefix_body and "-" in prefix_body and len(suffix) <= 2 and detection.confidence < 0.997:
            mark_detection_yellow(detection, "带副类号的短辅助号疑似截断，建议人工复核")
            continue

        if suffix_has_embedded_digit(suffix):
            mark_detection_yellow(detection, "辅助号中间疑似混入数字，建议人工复核")
            continue

        if x <= 2 and (detection.confidence < 0.96 or len(suffix) <= 3):
            mark_detection_yellow(detection, "边缘 crop 可能截断书号，建议人工复核")
            continue

        alpha = alpha_suffix_stem(detection.clean_text)
        if alpha is None:
            continue
        alpha_prefix, stem = alpha
        if index + 2 < len(detections):
            right_1 = suffix_stem_number(detections[index + 1].clean_text)
            right_2 = suffix_stem_number(detections[index + 2].clean_text)
            if (
                right_1 is not None
                and right_2 is not None
                and right_1[0] == alpha_prefix
                and right_2[0] == alpha_prefix
                and right_1[1] == stem
                and right_2[1] == stem
                and right_2[2] == right_1[2] + 1
                and right_1[2] > 1
            ):
                mark_detection_yellow(detection, f"疑似漏末位数字 {right_1[2] - 1}，建议人工复核")
                continue
        neighbors = []
        if index > 0:
            neighbors.append(detections[index - 1])
        if index + 1 < len(detections):
            neighbors.append(detections[index + 1])
        for neighbor in neighbors:
            numbered = suffix_stem_number(neighbor.clean_text)
            if numbered is None:
                continue
            neighbor_prefix, neighbor_stem, neighbor_number = numbered
            if neighbor_prefix == alpha_prefix and neighbor_stem == stem and neighbor_number >= 1:
                edge_pad = max(5.0, image_width * 0.006)
                near_edge = x <= edge_pad or x + w >= image_width - edge_pad
                unusually_narrow = w < typical_w * 0.72
                if near_edge or unusually_narrow:
                    mark_detection_yellow(detection, "边缘或窄 crop 临近编号书号，末位数字需人工复核")
                    break


def downgrade_isolated_prefix_outlier_status(detections: list[Detection]) -> None:
    if len(detections) < 5:
        return

    widths = [float(item.crop_box[2]) for item in detections if item.crop_box[2] > 0]
    typical_w = median(widths) or 0.0
    for index, detection in enumerate(detections):
        if detection.status != "red":
            continue
        parts = split_call_number_suffix(detection.clean_text)
        if parts is None:
            continue
        prefix, suffix = parts
        left = nearest_parsed_detection(detections, index, -1)
        right = nearest_parsed_detection(detections, index, 1)
        if left is None or right is None:
            continue
        left_parts = split_call_number_suffix(left.clean_text)
        right_parts = split_call_number_suffix(right.clean_text)
        if left_parts is None or right_parts is None:
            continue
        neighbor_prefix, left_suffix = left_parts
        right_prefix, right_suffix = right_parts
        if neighbor_prefix != right_prefix or neighbor_prefix == prefix:
            continue
        suffix_key = natural_parts(suffix)
        if natural_parts(left_suffix) <= suffix_key <= natural_parts(right_suffix):
            prefix_body = prefix.rstrip("/")
            neighbor_body = neighbor_prefix.rstrip("/")
            detection_key = call_number_order_key(detection.clean_text)
            left_key = call_number_order_key(left.clean_text)
            right_key = call_number_order_key(right.clean_text)
            if (
                prefix_body
                and neighbor_body
                and prefix_body[0] == neighbor_body[0]
                and detection.confidence >= 0.95
                and typical_w > 0
                and detection.crop_box[2] >= typical_w * 0.75
                and detection_key is not None
                and left_key is not None
                and right_key is not None
                and detection_key > max(left_key, right_key)
            ):
                continue
            detection.status = "yellow"
            detection.reason = "分类号与相邻书号不一致，疑似 OCR 前缀或馆内排架差异，建议人工复核"
            detection.recommended_position = None


def text_edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if char_a == char_b else 1),
                )
            )
        previous = current
    return previous[-1]


def correct_isolated_narrow_prefix_ocr(detections: list[Detection]) -> None:
    if len(detections) < 5:
        return

    widths = [float(item.crop_box[2]) for item in detections if item.crop_box[2] > 0]
    typical_w = median(widths) or 0.0
    if typical_w <= 0:
        return

    for index, detection in enumerate(detections):
        if not detection.parse_ok or "/" not in detection.clean_text:
            continue
        if detection.crop_box[2] > typical_w * 0.92:
            continue
        parts = split_call_number_suffix(detection.clean_text)
        if parts is None:
            continue
        prefix, suffix = parts
        left = nearest_parsed_detection(detections, index, -1)
        right = nearest_parsed_detection(detections, index, 1)
        if left is None or right is None:
            continue
        left_parts = split_call_number_suffix(left.clean_text)
        right_parts = split_call_number_suffix(right.clean_text)
        if left_parts is None or right_parts is None:
            continue
        neighbor_prefix, left_suffix = left_parts
        right_prefix, right_suffix = right_parts
        if neighbor_prefix != right_prefix or neighbor_prefix == prefix:
            continue
        suffix_key = natural_parts(suffix)
        if not (natural_parts(left_suffix) <= suffix_key <= natural_parts(right_suffix)):
            continue

        prefix_body = prefix.rstrip("/")
        neighbor_body = neighbor_prefix.rstrip("/")
        if not prefix_body or not neighbor_body:
            continue
        if prefix_body[0] != neighbor_body[0]:
            continue
        if text_edit_distance(prefix_body, neighbor_body) > 1:
            continue
        if (
            detection.confidence >= 0.95
            and typical_w > 0
            and detection.crop_box[2] >= typical_w * 0.75
            and prefix_body[0] == neighbor_body[0]
        ):
            continue

        corrected = f"{neighbor_prefix}{suffix}"
        if parse_call_number(corrected) is None:
            continue
        detection.raw_text = f"{detection.raw_text} [prefix corrected from narrow crop context: {detection.clean_text}]"
        detection.clean_text = corrected
        detection.parse_ok = True
        detection.confidence = min(detection.confidence, 0.92)


def nearest_parsed_detection(
    detections: list[Detection],
    start_index: int,
    step: int,
) -> Detection | None:
    index = start_index + step
    while 0 <= index < len(detections):
        detection = detections[index]
        if detection.parse_ok and "/" in detection.clean_text:
            return detection
        index += step
    return None


def remove_boundary_order_outliers(detections: list[Detection]) -> bool:
    if len(detections) < 10:
        return False

    changed = False
    while len(detections) >= 10:
        limit = max(3, int(len(detections) * 0.20))
        first = detections[0]
        last = detections[-1]

        if (
            last.status == "red"
            and last.recommended_position is not None
            and len(detections) - last.recommended_position >= limit
        ):
            detections.pop()
            changed = True
            continue

        if (
            first.status == "red"
            and first.recommended_position is not None
            and first.recommended_position - 1 >= limit
        ):
            detections.pop(0)
            changed = True
            continue

        break

    if changed:
        for index, detection in enumerate(detections, start=1):
            detection.index = index
    return changed


# Horizontal shelf helpers. They handle the Web/mobile photos where call numbers
# sit near a bottom red band rather than beside a vertical red marker.
def make_code_strip_box(
    image_shape: tuple[int, int, int],
    red_column: tuple[int, int] | None,
    crop_right_ratio: float,
) -> tuple[int, int, int, int] | None:
    if red_column is None:
        return None

    h, w = image_shape[:2]
    _, red_x1 = red_column
    x0 = min(w - 1, red_x1)
    strip_w = max(int(w * crop_right_ratio), 160)
    x1 = min(w, x0 + strip_w)
    if x1 - x0 < 40:
        return None
    return (x0, 0, x1 - x0, h)


def make_bottom_code_band_box(image_shape: tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    h, w = image_shape[:2]
    y0 = int(h * 0.58)
    y1 = int(h * 0.91)
    if y1 - y0 < 40:
        return None
    return (0, y0, w, y1 - y0)


def make_horizontal_code_band_box(
    image_shape: tuple[int, int, int],
    red_mask: np.ndarray,
) -> tuple[int, int, int, int] | None:
    h, w = image_shape[:2]
    if h <= 0 or w <= 0 or red_mask.size == 0:
        return None

    row_counts = (red_mask > 0).sum(axis=1).astype(float)
    window = max(5, h // 120)
    kernel = np.ones(window, dtype=float) / window
    smooth_counts = np.convolve(row_counts, kernel, mode="same")
    if smooth_counts.max() <= 0:
        return None

    # The new training samples are horizontal shelves: the useful call numbers sit
    # immediately below a long red label band. Ignore red shelves/table edges near
    # the bottom so they do not steal the band selection.
    threshold = max(w * 0.04, float(smooth_counts.max()) * 0.32)
    candidate_rows = np.where(smooth_counts >= threshold)[0]
    if len(candidate_rows) == 0:
        return None

    segments: list[tuple[int, int, float]] = []
    start = previous = int(candidate_rows[0])
    for value in candidate_rows[1:]:
        row = int(value)
        if row == previous + 1:
            previous = row
            continue
        density = float(smooth_counts[start : previous + 1].max()) / w
        segments.append((start, previous, density))
        start = previous = row
    density = float(smooth_counts[start : previous + 1].max()) / w
    segments.append((start, previous, density))

    useful_segments: list[tuple[int, int, float]] = []
    for y0, y1, density in segments:
        center_ratio = ((y0 + y1) / 2) / h
        if y1 - y0 < 6:
            continue
        if density < 0.08:
            continue
        if not 0.28 <= center_ratio <= 0.86:
            continue
        useful_segments.append((y0, y1, density))

    if not useful_segments:
        return None

    band_y0, band_y1, _ = max(useful_segments, key=lambda item: (item[2], item[1] - item[0]))
    crop_y0 = max(0, int(band_y1 - h * 0.03))
    crop_y1 = min(h, int(band_y1 + h * 0.37))
    if crop_y1 - crop_y0 < 40:
        return None
    return (0, crop_y0, w, crop_y1 - crop_y0)


def estimate_horizontal_red_band_rows(
    red_mask: np.ndarray,
    band_box: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    x, y, w, h = band_box
    if w <= 0 or h <= 0:
        return None
    band_mask = red_mask[y : y + h, x : x + w]
    if band_mask.size == 0:
        return None

    row_counts = np.count_nonzero(band_mask > 0, axis=1)
    if row_counts.size == 0 or row_counts.max() <= 0:
        return None
    threshold = max(w * 0.08, float(row_counts.max()) * 0.45)
    rows = np.where(row_counts >= threshold)[0]
    if len(rows) == 0:
        return None
    red_y0 = max(0, int(rows[0]) - 5)
    red_y1 = min(h, int(rows[-1]) + 6)
    if red_y1 - red_y0 < 8:
        return None
    return y + red_y0, y + red_y1


def detections_from_bottom_text_band(
    image: np.ndarray,
    red_mask: np.ndarray,
    band_box: tuple[int, int, int, int] | None,
    crops_dir: Path,
) -> list[Detection]:
    if band_box is None:
        return []

    image_h, image_w = image.shape[:2]
    red_rows = estimate_horizontal_red_band_rows(red_mask, band_box)
    if red_rows is None:
        return []
    red_y0, red_y1 = red_rows
    if red_y1 < image_h * 0.50:
        return []

    crop_y0 = min(image_h - 1, red_y1 + max(8, int(image_h * 0.003)))
    crop_y1 = min(image_h, int(red_y1 + image_h * 0.25), int(image_h * 0.94))
    if crop_y1 - crop_y0 < max(55, int(image_h * 0.035)):
        return []

    roi = image[crop_y0:crop_y1, :]
    if roi.size == 0:
        return []
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        11,
    )

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(threshold, 8)
    clean = np.zeros_like(threshold)
    roi_h, roi_w = threshold.shape[:2]
    for component_index in range(1, component_count):
        x, y, w, h, area = stats[component_index]
        if area < max(6, int(image_w * image_h * 0.000001)):
            continue
        if area > roi_h * roi_w * 0.05:
            continue
        if h > roi_h * 0.88 and w < max(4, int(image_w * 0.002)):
            continue
        if w > roi_w * 0.12 or h > roi_h * 0.75:
            continue
        clean[labels == component_index] = 255

    kernel_w = max(5, int(image_w * 0.0017))
    if kernel_w % 2 == 0:
        kernel_w += 1
    kernel_h = max(21, int(roi_h * 0.055))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    joined = cv2.dilate(clean, kernel, iterations=1)
    col_counts = np.count_nonzero(joined > 0, axis=0).astype(float)
    if col_counts.max() <= 0:
        return []

    smooth_window = max(7, int(image_w * 0.0027))
    smooth = np.convolve(col_counts, np.ones(smooth_window) / smooth_window, mode="same")
    flags = smooth >= max(4.0, roi_h * 0.20)
    segments = find_segments(
        flags,
        min_length=max(10, int(image_w * 0.004)),
        gap=max(6, int(image_w * 0.0044)),
    )

    detections: list[Detection] = []
    crops_dir.mkdir(parents=True, exist_ok=True)
    min_w = max(16, int(image_w * 0.008))
    max_w = max(95, int(image_w * 0.045))
    for seg_x0, seg_x1 in segments:
        text_w = seg_x1 - seg_x0
        if text_w < min_w or text_w > max_w:
            continue
        pad_x = max(8, int(text_w * 0.42), int(image_w * 0.004))
        x0 = max(0, int(seg_x0 - pad_x))
        x1 = min(image_w, int(seg_x1 + pad_x))
        if x1 - x0 < min_w:
            continue
        crop_box = (x0, crop_y0, x1 - x0, crop_y1 - crop_y0)
        ink = np.count_nonzero(clean[:, x0:x1] > 0)
        if ink < max(18, int((x1 - x0) * (crop_y1 - crop_y0) * 0.004)):
            continue
        red_box = (x0, red_y0, x1 - x0, max(1, red_y1 - red_y0))
        detection = Detection(
            index=len(detections) + 1,
            red_box=red_box,
            crop_box=crop_box,
            status="yellow",
            reason="底部横红带书号分割候选，建议人工复核",
        )
        crop = image[crop_y0:crop_y1, x0:x1]
        crop_path = crops_dir / f"bottom_text_crop_{detection.index:03d}.jpg"
        write_image(crop_path, crop)
        detection.crop_path = crop_path
        detections.append(detection)

    return detections


def should_use_bottom_text_band_detections(
    current: list[Detection],
    bottom_text: list[Detection],
    image_shape: tuple[int, int, int],
) -> bool:
    if len(bottom_text) < 6:
        return False
    current_valid = valid_detection_count(current)
    if current_valid >= max(6, int(len(bottom_text) * 0.65)):
        return False
    if current_valid <= max(2, int(len(bottom_text) * 0.25)):
        return True

    image_w = image_shape[1]
    wide_count = sum(1 for item in current if item.crop_box[2] > image_w * 0.22)
    thin_red_count = sum(1 for item in current if item.red_box[2] < image_w * 0.018)
    return wide_count >= max(4, int(len(current) * 0.35)) and thin_red_count >= max(4, int(len(current) * 0.35))


# Crop-bound adjustment for horizontal detections. The goal is to keep each book
# crop complete even when adjacent call numbers are close together.
def expand_horizontal_crop_y_bounds(
    detections: list[Detection],
    image_shape: tuple[int, int, int],
) -> None:
    if len(detections) < 4:
        return

    anchors = [
        item
        for item in detections
        if item.crop_box[3] > 0
        and item.confidence >= 0.70
        and (item.parse_ok or "/" in item.clean_text)
    ]
    if len(anchors) < 4:
        anchors = [item for item in detections if item.crop_box[3] > 0]
    if len(anchors) < 4:
        return

    image_h = image_shape[0]
    heights = [float(item.crop_box[3]) for item in anchors]
    typical_h = median(heights)
    if typical_h <= 0:
        return

    y0_values = [float(item.crop_box[1]) for item in anchors]
    y1_values = [float(item.crop_box[1] + item.crop_box[3]) for item in anchors]
    top_pad = max(4, int(typical_h * 0.06))
    bottom_pad = max(8, int(typical_h * 0.18))
    common_y0 = int(max(0, percentile(y0_values, 0.10) - top_pad))
    common_y1 = int(min(image_h, percentile(y1_values, 0.90) + bottom_pad))
    if common_y1 <= common_y0:
        return

    max_common_h = max(80, int(typical_h * 1.75))
    if common_y1 - common_y0 > max_common_h:
        centers = [item.crop_box[1] + item.crop_box[3] / 2 for item in anchors]
        center_y = int(median([float(value) for value in centers]))
        common_y0 = max(0, center_y - max_common_h // 2)
        common_y1 = min(image_h, common_y0 + max_common_h)

    for detection in detections:
        x, y, w, h = detection.crop_box
        y0 = min(y, common_y0)
        y1 = max(y + h, common_y1)
        if y1 <= y0:
            continue
        detection.crop_box = (x, y0, w, y1 - y0)


def detections_from_ocr_strip(
    image: np.ndarray,
    strip_box: tuple[int, int, int, int],
    ocr_lines: list[OcrLine],
    crops_dir: Path,
    order: str = "y_desc",
    allow_recovery: bool = True,
    keep_invalid: bool = True,
    min_confidence: float = 0.0,
) -> list[Detection]:
    strip_x, strip_y, _, _ = strip_box
    crops_dir.mkdir(parents=True, exist_ok=True)
    detections: list[Detection] = []

    for candidate in build_ocr_candidates(ocr_lines, allow_recovery=allow_recovery):
        clean_text = candidate.clean_text
        if not clean_text:
            continue
        if candidate.confidence < min_confidence:
            continue
        if order == "x_asc" and "/" not in clean_text and not clean_text.startswith(("D66", "D669")):
            continue
        parse_ok = parse_call_number(clean_text) is not None
        if not keep_invalid and not parse_ok:
            continue

        lx, ly, lw, lh = candidate.box
        if order == "x_asc":
            pad_x = max(3, int(lw * 0.08))
            pad_y = max(3, int(lh * 0.16))
        else:
            pad_x = max(8, int(lw * 0.18))
            pad_y = max(4, int(lh * 0.35))
        x0 = max(0, strip_x + lx - pad_x)
        y0 = max(0, strip_y + ly - pad_y)
        x1 = min(image.shape[1], strip_x + lx + lw + pad_x)
        y1 = min(image.shape[0], strip_y + ly + lh + pad_y)
        crop_box = (x0, y0, x1 - x0, y1 - y0)
        red_box = (max(0, strip_x - 8), y0, 8, y1 - y0)

        detection = Detection(
            index=len(detections) + 1,
            red_box=red_box,
            crop_box=crop_box,
            raw_text=candidate.raw_text,
            clean_text=clean_text,
            confidence=candidate.confidence,
            parse_ok=parse_ok,
        )

        detections.append(detection)

    if order == "x_asc":
        detections.sort(key=lambda d: d.center_x)
        correct_horizontal_suffix_digit_bleed(detections)
        expand_horizontal_crop_y_bounds(detections, image.shape)
        trim_horizontal_crop_x_overlaps(detections, image.shape[1])
        add_horizontal_crop_x_context(detections, image.shape[1])
    else:
        detections.sort(key=lambda d: d.center_y, reverse=True)
    for i, detection in enumerate(detections, start=1):
        detection.index = i
        crop = image[
            detection.crop_box[1] : detection.crop_box[1] + detection.crop_box[3],
            detection.crop_box[0] : detection.crop_box[0] + detection.crop_box[2],
        ]
        crop_path = crops_dir / f"ocr_crop_{i:03d}.jpg"
        write_image(crop_path, crop)
        detection.crop_path = crop_path
    return detections


def correct_horizontal_suffix_digit_bleed(detections: list[Detection]) -> None:
    if len(detections) < 2:
        return

    widths = [d.crop_box[2] for d in detections if d.crop_box[2] > 0]
    typical_w = median([float(width) for width in widths]) or 40.0

    for left, right in zip(detections, detections[1:]):
        left_parts = split_call_number_suffix(left.clean_text)
        right_parts = split_call_number_suffix(right.clean_text)
        if left_parts is None or right_parts is None:
            continue

        left_prefix, left_suffix = left_parts
        right_prefix, right_suffix = right_parts
        if left_prefix != right_prefix:
            continue

        lx, ly, lw, lh = left.crop_box
        rx, ry, rw, rh = right.crop_box
        horizontal_gap = rx - (lx + lw)
        vertical_gap = abs((ly + lh / 2) - (ry + rh / 2))
        same_row = vertical_gap <= max(lh, rh) * 0.55
        left_too_wide = lw >= max(rw * 1.15, typical_w * 1.10)
        boxes_touch = horizontal_gap <= typical_w * 0.35

        left_number_bleed = re.fullmatch(r"([A-Z]{2,5})([1-9]\d+)", left_suffix)
        right_numbered = re.fullmatch(r"([A-Z]{2,5})([1-9]\d*)", right_suffix)
        if (
            left_number_bleed is not None
            and right_numbered is not None
            and same_row
            and boxes_touch
            and left_too_wide
            and left.confidence < 0.50
        ):
            left_letters, left_digits = left_number_bleed.groups()
            right_letters, right_digits = right_numbered.groups()
            if left_letters == right_letters and left_digits[-1] == right_digits[0]:
                corrected_digits = left_digits[:-1]
                if corrected_digits:
                    left.clean_text = f"{left_prefix}{left_letters}{corrected_digits}"
                    left.raw_text = f"{left.raw_text} [trimmed neighbor suffix digit]"
                    left.parse_ok = parse_call_number(left.clean_text) is not None
                    left.confidence = min(left.confidence, 0.88)
                    boundary = int((left.center_x + right.center_x) / 2)
                    new_left_right = max(lx + 18, min(lx + lw, boundary))
                    if new_left_right < lx + lw:
                        left.crop_box = (lx, ly, new_left_right - lx, lh)
                    continue

        # Do not move a single terminal digit from the left crop to the right crop.
        # Hard-case review showed this rule could turn a correct pair such as
        # ".../DJW1, .../DLY" into ".../DJW, .../DLY1". Only the duplicate-digit
        # trimming branch above is kept.


def trim_horizontal_crop_x_overlaps(detections: list[Detection], image_width: int) -> None:
    if len(detections) < 2:
        return

    widths = [float(item.crop_box[2]) for item in detections if item.crop_box[2] > 0]
    typical_w = median(widths) or 28.0
    overlap_pad = max(2, min(6, int(typical_w * 0.16)))

    for left, right in zip(detections, detections[1:]):
        lx, ly, lw, lh = left.crop_box
        rx, ry, rw, rh = right.crop_box
        left_right = lx + lw
        if left_right <= rx + 1:
            continue

        boundary = int((left.center_x + right.center_x) / 2)
        min_left_w = min(lw, max(18, int(lw * 0.45)))
        min_right_w = min(rw, max(18, int(rw * 0.45)))

        new_left_right = max(lx + min_left_w, min(left_right, boundary - 1))
        new_right_x = min(rx + rw - min_right_w, max(rx, boundary + 1))

        if new_left_right < left_right:
            padded_right = min(image_width, new_left_right + overlap_pad)
            left.crop_box = (lx, ly, max(1, padded_right - lx), lh)
        if new_right_x > rx and new_right_x < rx + rw:
            padded_left = max(0, new_right_x - overlap_pad)
            right.crop_box = (padded_left, ry, max(1, rx + rw - padded_left), rh)


def add_horizontal_crop_x_context(detections: list[Detection], image_width: int) -> None:
    if len(detections) < 2:
        return

    widths = [float(item.crop_box[2]) for item in detections if item.crop_box[2] > 0]
    typical_w = median(widths) or 28.0
    pad = max(3, min(8, int(typical_w * 0.20)))
    for detection in detections:
        x, y, w, h = detection.crop_box
        x0 = max(0, x - pad)
        x1 = min(image_width, x + w + pad)
        if x1 > x0:
            detection.crop_box = (x0, y, x1 - x0, h)


# Call-number correction helpers. These rules handle common OCR slips such as
# missing final digits, duplicated suffix digits between adjacent crops, and
# impossible letter/digit order after the slash.
def suffix_stem_number(value: str) -> tuple[str, str, int] | None:
    parts = split_call_number_suffix(value)
    if parts is None:
        return None
    prefix, suffix = parts
    match = re.fullmatch(r"([A-Z]{1,5})([1-9]\d*)", suffix)
    if match is None:
        return None
    stem, number = match.groups()
    return prefix, stem, int(number)


def alpha_suffix_stem(value: str) -> tuple[str, str] | None:
    parts = split_call_number_suffix(value)
    if parts is None:
        return None
    prefix, suffix = parts
    if not re.fullmatch(r"[A-Z]{1,5}", suffix):
        return None
    return prefix, suffix


def suffix_has_embedded_digit(suffix: str) -> bool:
    return bool(re.search(r"\d(?=[A-Z])", suffix))


def trim_letters_after_suffix_number(value: str) -> str | None:
    parts = split_call_number_suffix(value)
    if parts is None:
        return None
    prefix, suffix = parts
    match = re.fullmatch(r"([A-Z]{2,5}\d+)[A-Z]+", suffix)
    if match is None:
        return None
    corrected = f"{prefix}{match.group(1)}"
    return corrected if parse_call_number(corrected) is not None else None


def trim_separated_trailing_digit_noise(raw_text: str, value: str) -> str | None:
    parts = split_call_number_suffix(value)
    if parts is None:
        return None
    prefix, suffix = parts
    match = re.fullmatch(r"([A-Z]{2,5}\d)\d+", suffix)
    if match is None:
        return None
    raw_pattern = re.escape(prefix.rstrip("/")) + r"\s*/\s*" + re.escape(match.group(1)) + r"\s+\d+\b"
    if re.search(raw_pattern, raw_text.upper()):
        corrected = f"{prefix}{match.group(1)}"
        return corrected if parse_call_number(corrected) is not None else None
    return None


def suffix_body_and_trailing_number(suffix: str) -> tuple[str, str]:
    match = re.fullmatch(r"(.+?)(\d*)", suffix)
    if not match:
        return suffix, ""
    return match.group(1), match.group(2)


def nearby_same_prefix_suffixes(
    detections: list[Detection],
    index: int,
    prefix: str,
    window: int = 3,
) -> list[str]:
    suffixes: list[str] = []
    start = max(0, index - window)
    end = min(len(detections), index + window + 1)
    for neighbor_index in range(start, end):
        if neighbor_index == index:
            continue
        parts = split_call_number_suffix(detections[neighbor_index].clean_text)
        if parts is None:
            continue
        neighbor_prefix, neighbor_suffix = parts
        if neighbor_prefix == prefix:
            suffixes.append(neighbor_suffix)
    return suffixes


def correct_contextual_suffix_ocr_errors(detections: list[Detection]) -> None:
    """Fix high-frequency Z/T/7/B OCR confusions only inside a local Z-suffix run."""
    if len(detections) < 3:
        return

    ordered = sorted(detections, key=lambda item: item.center_x)
    for index, detection in enumerate(ordered):
        parts = split_call_number_suffix(detection.clean_text)
        if parts is None:
            continue
        prefix, suffix = parts
        if not suffix:
            continue

        trimmed_text = trim_letters_after_suffix_number(detection.clean_text)
        if trimmed_text is not None:
            detection.raw_text = f"{detection.raw_text} [trimmed suffix letters after digit: {detection.clean_text}]"
            detection.clean_text = trimmed_text
            detection.parse_ok = True
            detection.confidence = min(detection.confidence, 0.92)
            parts = split_call_number_suffix(detection.clean_text)
            if parts is None:
                continue
            prefix, suffix = parts

        separated_digit_text = trim_separated_trailing_digit_noise(detection.raw_text, detection.clean_text)
        if separated_digit_text is not None and detection.confidence < 0.75:
            detection.raw_text = f"{detection.raw_text} [trimmed separated suffix digit noise: {detection.clean_text}]"
            detection.clean_text = separated_digit_text
            detection.parse_ok = True
            detection.confidence = min(detection.confidence, 0.88)
            parts = split_call_number_suffix(detection.clean_text)
            if parts is None:
                continue
            prefix, suffix = parts

        body, trailing_number = suffix_body_and_trailing_number(suffix)
        if not body:
            continue
        if "-" in prefix.rstrip("/"):
            continue
        ambiguous_leader = body[0] in {"T", "B"}
        ambiguous_seven = "7" in body
        if not ambiguous_leader and not ambiguous_seven:
            continue

        neighbor_suffixes = nearby_same_prefix_suffixes(ordered, index, prefix)
        if not neighbor_suffixes:
            continue
        z_neighbors = [item for item in neighbor_suffixes if item.startswith("Z")]
        if not z_neighbors:
            continue
        if body[0] == "B" and len(z_neighbors) < 2:
            continue

        corrected_body = body
        if corrected_body[0] in {"T", "B"}:
            corrected_body = "Z" + corrected_body[1:]
        corrected_body = corrected_body.replace("7", "Z")
        corrected_suffix = corrected_body + trailing_number
        if corrected_suffix == suffix:
            continue

        corrected_text = f"{prefix}{corrected_suffix}"
        if parse_call_number(corrected_text) is None:
            continue

        detection.raw_text = f"{detection.raw_text} [context suffix corrected: {detection.clean_text}]"
        detection.clean_text = corrected_text
        detection.parse_ok = True
        detection.confidence = min(detection.confidence, 0.93)

    for index in range(1, len(ordered) - 1):
        left = ordered[index - 1]
        detection = ordered[index]
        right = ordered[index + 1]
        if detection.confidence >= 0.94:
            continue
        left_alpha = alpha_suffix_stem(left.clean_text)
        current_numbered = suffix_stem_number(detection.clean_text)
        right_parts = split_call_number_suffix(right.clean_text)
        if left_alpha is None or current_numbered is None or right_parts is None:
            continue
        left_prefix, left_stem = left_alpha
        current_prefix, current_stem, current_number = current_numbered
        right_prefix, right_suffix = right_parts
        if left_prefix != current_prefix or left_prefix != right_prefix:
            continue
        if current_number != 1:
            continue
        if len(left_stem) != len(current_stem) or len(left_stem) < 2:
            continue
        if left_stem[0] != current_stem[0]:
            continue
        if text_edit_distance(left_stem, current_stem) != 1:
            continue
        corrected_suffix = f"{left_stem}{current_number}"
        if natural_parts(corrected_suffix) > natural_parts(right_suffix):
            continue
        corrected_text = f"{current_prefix}{corrected_suffix}"
        if parse_call_number(corrected_text) is None:
            continue
        detection.raw_text = f"{detection.raw_text} [context numbered suffix corrected: {detection.clean_text}]"
        detection.clean_text = corrected_text
        detection.parse_ok = True
        detection.confidence = min(detection.confidence, 0.90)

    for index in range(1, len(ordered) - 1):
        detection = ordered[index]
        left = nearest_parsed_detection(ordered, index, -1)
        right = nearest_parsed_detection(ordered, index, 1)
        if left is None or right is None:
            continue
        if detection.confidence >= 0.95:
            continue
        left_parts = split_call_number_suffix(left.clean_text)
        current_parts = split_call_number_suffix(detection.clean_text)
        right_parts = split_call_number_suffix(right.clean_text)
        if left_parts is None or current_parts is None or right_parts is None:
            continue
        left_prefix, left_suffix = left_parts
        current_prefix, current_suffix = current_parts
        right_prefix, right_suffix = right_parts
        if left_prefix != current_prefix or left_prefix != right_prefix:
            continue
        if not re.fullmatch(r"[A-Z]{2,5}", current_suffix):
            continue
        if natural_parts(left_suffix) <= natural_parts(current_suffix) <= natural_parts(right_suffix):
            continue
        if len(right_suffix) < len(current_suffix):
            continue
        candidate_suffix = right_suffix[: len(current_suffix)]
        if candidate_suffix[0] != current_suffix[0]:
            continue
        if text_edit_distance(candidate_suffix, current_suffix) != 1:
            continue
        if not (natural_parts(left_suffix) <= natural_parts(candidate_suffix) <= natural_parts(right_suffix)):
            continue
        corrected_text = f"{current_prefix}{candidate_suffix}"
        if parse_call_number(corrected_text) is None:
            continue
        detection.raw_text = f"{detection.raw_text} [context short suffix corrected: {detection.clean_text}]"
        detection.clean_text = corrected_text
        detection.parse_ok = True
        detection.confidence = min(detection.confidence, 0.90)


def set_inferred_numeric_suffix(detection: Detection, prefix: str, stem: str, number: int) -> None:
    inferred = f"{prefix}{stem}{number}"
    if parse_call_number(inferred) is None:
        return
    detection.raw_text = f"{detection.raw_text} [inferred numeric suffix {number}]"
    detection.clean_text = inferred
    detection.parse_ok = True
    detection.confidence = min(detection.confidence, 0.88)


def infer_horizontal_missing_numeric_suffixes(detections: list[Detection]) -> None:
    if len(detections) < 3:
        return

    for index, detection in enumerate(detections):
        alpha = alpha_suffix_stem(detection.clean_text)
        if alpha is None:
            continue
        prefix, stem = alpha

        if index + 2 < len(detections):
            right_1 = suffix_stem_number(detections[index + 1].clean_text)
            right_2 = suffix_stem_number(detections[index + 2].clean_text)
            if (
                right_1 is not None
                and right_2 is not None
                and right_1[0] == prefix
                and right_2[0] == prefix
                and right_1[1] == stem
                and right_2[1] == stem
                and right_2[2] == right_1[2] + 1
                and right_1[2] > 1
            ):
                set_inferred_numeric_suffix(detection, prefix, stem, right_1[2] - 1)
                continue

        if 0 < index < len(detections) - 1:
            left_1 = suffix_stem_number(detections[index - 1].clean_text)
            right_1 = suffix_stem_number(detections[index + 1].clean_text)
            if (
                left_1 is not None
                and right_1 is not None
                and left_1[0] == prefix
                and right_1[0] == prefix
                and left_1[1] == stem
                and right_1[1] == stem
                and right_1[2] == left_1[2] + 2
            ):
                set_inferred_numeric_suffix(detection, prefix, stem, left_1[2] + 1)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * max(0.0, min(1.0, q))
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def detection_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    inter_w = max(0, x1 - x0)
    inter_h = max(0, y1 - y0)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def horizontal_overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, _, aw, _ = a
    bx, _, bw, _ = b
    overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    return overlap / max(1, min(aw, bw))


def detection_text_quality_score(detection: Detection, typical_w: float) -> float:
    score = detection.confidence
    if detection.parse_ok:
        score += 0.35
    parts = split_call_number_suffix(detection.clean_text)
    if parts is not None:
        _, suffix = parts
        if suffix_has_embedded_digit(suffix):
            score -= 0.20
        if len(suffix) <= 1:
            score -= 0.18
    if typical_w > 0 and detection.crop_box[2] > typical_w * 1.45:
        score -= 0.08
    return score


def likely_same_physical_book(left: Detection, right: Detection, typical_w: float) -> bool:
    lx, ly, lw, lh = left.crop_box
    rx, ry, rw, rh = right.crop_box
    if lw <= 0 or rw <= 0 or lh <= 0 or rh <= 0:
        return False
    vertical_gap = abs((ly + lh / 2) - (ry + rh / 2))
    if vertical_gap > max(lh, rh) * 0.55:
        return False

    center_gap = abs(left.center_x - right.center_x)
    overlap_ratio = horizontal_overlap_ratio(left.crop_box, right.crop_box)
    if overlap_ratio < 0.62 and center_gap > max(10.0, min(lw, rw) * 0.58):
        return False

    left_parts = split_call_number_suffix(left.clean_text)
    right_parts = split_call_number_suffix(right.clean_text)
    if left_parts is None or right_parts is None:
        return left.clean_text == right.clean_text and bool(left.clean_text)
    left_prefix, left_suffix = left_parts
    right_prefix, right_suffix = right_parts
    if left_prefix != right_prefix:
        return False
    return text_edit_distance(left_suffix, right_suffix) <= 2


def prune_overlapping_duplicate_detections(detections: list[Detection]) -> list[Detection]:
    if len(detections) < 2:
        return detections

    widths = [float(item.crop_box[2]) for item in detections if item.crop_box[2] > 0]
    typical_w = median(widths) or 32.0
    ordered = sorted(detections, key=lambda item: item.center_x)
    keep = [True] * len(ordered)

    for index in range(len(ordered) - 1):
        if not keep[index]:
            continue
        left = ordered[index]
        right = ordered[index + 1]
        if not likely_same_physical_book(left, right, typical_w):
            continue
        left_score = detection_text_quality_score(left, typical_w)
        right_score = detection_text_quality_score(right, typical_w)
        if right_score >= left_score:
            keep[index] = False
        else:
            keep[index + 1] = False

    if all(keep):
        return detections

    pruned = [detection for detection, should_keep in zip(ordered, keep) if should_keep]
    for index, detection in enumerate(pruned, start=1):
        detection.index = index
    return pruned


def call_number_numeric_suffix(value: str) -> tuple[str, int] | None:
    match = re.match(r"^(.+?)(\d+)$", value)
    if not match:
        return None
    stem, number = match.groups()
    return stem, int(number)


def save_detection_crop(image: np.ndarray, detection: Detection, crops_dir: Path, force: bool = False) -> None:
    if detection.crop_path is not None and not force:
        return
    crops_dir.mkdir(parents=True, exist_ok=True)
    x, y, w, h = detection.crop_box
    crop = image[y : y + h, x : x + w]
    path = crops_dir / f"ocr_crop_{detection.index:03d}.jpg"
    write_image(path, crop)
    detection.crop_path = path


# Fine-mode crop helpers. They resize a candidate box within image bounds and
# retry OCR with slightly different margins when the first crop is suspicious.
def clamp_crop_box(
    image_shape: tuple[int, int, int],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> tuple[int, int, int, int] | None:
    image_h, image_w = image_shape[:2]
    ix0 = max(0, min(image_w - 1, int(round(x0))))
    iy0 = max(0, min(image_h - 1, int(round(y0))))
    ix1 = max(0, min(image_w, int(round(x1))))
    iy1 = max(0, min(image_h, int(round(y1))))
    if ix1 - ix0 < 12 or iy1 - iy0 < 8:
        return None
    return (ix0, iy0, ix1 - ix0, iy1 - iy0)


def is_horizontal_detection_layout(detections: list[Detection]) -> bool:
    meaningful = [item for item in detections if item.crop_box[2] > 0 and item.crop_box[3] > 0]
    if len(meaningful) < 4:
        return False
    xs = [item.center_x for item in meaningful]
    ys = [item.crop_box[1] + item.crop_box[3] / 2 for item in meaningful]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    return x_span > max(120.0, y_span * 1.8)


def add_crop_variant(
    variants: list[tuple[str, tuple[int, int, int, int]]],
    seen: set[tuple[int, int, int, int]],
    image_shape: tuple[int, int, int],
    name: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> None:
    box = clamp_crop_box(image_shape, x0, y0, x1, y1)
    if box is None or box in seen:
        return
    seen.add(box)
    variants.append((name, box))


def crop_retry_variant_boxes(
    image_shape: tuple[int, int, int],
    detections: list[Detection],
    detection: Detection,
) -> list[tuple[str, tuple[int, int, int, int]]]:
    x, y, w, h = detection.crop_box
    x0, y0, x1, y1 = float(x), float(y), float(x + w), float(y + h)
    variants: list[tuple[str, tuple[int, int, int, int]]] = []
    seen: set[tuple[int, int, int, int]] = set()

    add_crop_variant(variants, seen, image_shape, "crop_original", x0, y0, x1, y1)

    x_pad_small = max(3.0, w * 0.08)
    x_pad_med = max(5.0, w * 0.14)
    x_pad_big = max(7.0, w * 0.22)
    y_pad_med = max(5.0, h * 0.12)

    clean_text = normalize_ocr_text(detection.clean_text)
    parse_ok = parse_call_number(clean_text) is not None
    suffix_short = False
    parts = split_call_number_suffix(clean_text)
    if parts is not None:
        _, suffix = parts
        suffix_short = len(suffix) <= 2

    if not clean_text or not parse_ok or len(clean_text) > 16:
        add_crop_variant(variants, seen, image_shape, "x_tight_08", x0 + x_pad_small, y0, x1 - x_pad_small, y1)
        add_crop_variant(variants, seen, image_shape, "trim_left_14", x0 + x_pad_med, y0, x1, y1)
        add_crop_variant(variants, seen, image_shape, "trim_right_14", x0, y0, x1 - x_pad_med, y1)

    add_crop_variant(variants, seen, image_shape, "x_expand_10", x0 - x_pad_small, y0, x1 + x_pad_small, y1)

    if not clean_text or "/" not in clean_text or suffix_short:
        add_crop_variant(variants, seen, image_shape, "expand_right_22", x0, y0, x1 + x_pad_big, y1)
        add_crop_variant(variants, seen, image_shape, "expand_left_22", x0 - x_pad_big, y0, x1, y1)

    add_crop_variant(variants, seen, image_shape, "drop_top_red", x0, y0 + y_pad_med, x1, y1)

    if is_horizontal_detection_layout(detections):
        ordered = sorted(detections, key=lambda item: item.center_x)
        try:
            position = ordered.index(detection)
        except ValueError:
            position = -1
        if position >= 0:
            split_x0 = x0
            split_x1 = x1
            if position > 0:
                split_x0 = max(split_x0, (ordered[position - 1].center_x + detection.center_x) / 2)
            if position + 1 < len(ordered):
                split_x1 = min(split_x1, (detection.center_x + ordered[position + 1].center_x) / 2)
            if split_x1 - split_x0 >= max(12.0, w * 0.42):
                add_crop_variant(variants, seen, image_shape, "neighbor_midline", split_x0, y0, split_x1, y1)

    return variants[:7]


def crop_ocr_candidate_score(clean_text: str, confidence: float) -> float:
    clean_text = normalize_ocr_text(clean_text)
    if not clean_text:
        return -100.0 + confidence * 20.0

    score = confidence * 100.0 + min(len(clean_text), 22) * 0.45
    parse_ok = parse_call_number(clean_text) is not None
    has_slash = "/" in clean_text

    if parse_ok and has_slash:
        score += 95.0
    elif parse_ok:
        score += 22.0
    else:
        score -= 28.0

    if has_slash:
        left, right = clean_text.split("/", 1)
        if re.fullmatch(r"[A-Z]+\d+(?:\.\d+)?(?:-\d+)?", left):
            score += 12.0
        else:
            score -= 16.0
        if right and right[0].isalpha():
            score += 12.0
        else:
            score -= 24.0
        if re.fullmatch(r"[A-Z]{2,5}\d{0,4}(?::\d+)?", right):
            score += 12.0
        if len(right) <= 1:
            score -= 22.0
        if right.startswith("II"):
            score -= 30.0
    else:
        score -= 34.0

    if re.fullmatch(r"[A-Z]{1,3}", clean_text):
        score -= 65.0
    if re.fullmatch(r"[A-Z]\d{0,2}", clean_text):
        score -= 48.0
    if clean_text.startswith("II"):
        score -= 30.0
    if len(clean_text) > 24:
        score -= min(30.0, (len(clean_text) - 24) * 2.0)
    return score


def is_gapfill_detection(detection: Detection) -> bool:
    return detection.red_box[2] <= 1 and bool(detection.reason)


def detection_needs_crop_retry(
    image: np.ndarray,
    detection: Detection,
    confidence_threshold: float,
) -> bool:
    x, y, w, h = detection.crop_box
    if w <= 0 or h <= 0:
        return False

    clean_text = normalize_ocr_text(detection.clean_text)
    if not clean_text:
        return True
    if detection.confidence < confidence_threshold + 0.05:
        return True
    if parse_call_number(clean_text) is None:
        return True
    if "/" not in clean_text:
        return True
    parts = split_call_number_suffix(clean_text)
    if parts is not None:
        _, suffix = parts
        if len(suffix) <= 1 or suffix[0].isdigit():
            return True
        if len(suffix) <= 2:
            return True
        if re.fullmatch(r"[A-Z]{1,2}", suffix) and detection.confidence < 0.90:
            return True
    return False


def detection_needs_current_crop_refine(detection: Detection, confidence_threshold: float) -> bool:
    if detection.crop_path is None:
        return False

    clean_text = normalize_ocr_text(detection.clean_text)
    if is_gapfill_detection(detection) and parse_call_number(clean_text) is not None:
        return False
    if not clean_text:
        return True
    if "/" not in clean_text:
        return True
    if parse_call_number(clean_text) is None:
        return True
    if detection.confidence < max(0.40, confidence_threshold - 0.18):
        return True

    parts = split_call_number_suffix(clean_text)
    if parts is not None:
        _, suffix = parts
        if suffix_has_embedded_digit(suffix):
            return True
    return False


def write_variant_crop(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    path: Path,
) -> None:
    x, y, w, h = box
    crop = image[y : y + h, x : x + w]
    if crop.size > 0 and crop.shape[0] > crop.shape[1] * 1.35:
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if crop.size > 0:
        crop_h, crop_w = crop.shape[:2]
        max_ratio = 3.0
        if crop_w > crop_h * max_ratio:
            target_h = int(math.ceil(crop_w / max_ratio))
            pad_total = max(0, target_h - crop_h)
            top = pad_total // 2
            bottom = pad_total - top
            crop = cv2.copyMakeBorder(crop, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        elif crop_h > crop_w * max_ratio:
            target_w = int(math.ceil(crop_h / max_ratio))
            pad_total = max(0, target_w - crop_w)
            left = pad_total // 2
            right = pad_total - left
            crop = cv2.copyMakeBorder(crop, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    write_image(path, crop)


def crop_ocr_attempt_to_dict(attempt: CropOcrAttempt) -> dict[str, Any]:
    return {
        "variant": attempt.variant,
        "crop_box": list(attempt.crop_box),
        "crop_path": str(attempt.crop_path or ""),
        "raw_text": attempt.raw_text,
        "clean_text": attempt.clean_text,
        "confidence": round(attempt.confidence, 6),
        "parse_ok": attempt.parse_ok,
        "score": round(attempt.score, 3),
        "selected": attempt.selected,
    }


def parse_crop_ocr_attempts(value: str) -> list[CropOcrAttempt]:
    if not value:
        return []
    try:
        raw_items = json.loads(value)
    except Exception:
        return []
    if not isinstance(raw_items, list):
        return []

    attempts: list[CropOcrAttempt] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        attempts.append(
            CropOcrAttempt(
                variant=str(item.get("variant") or ""),
                crop_box=parse_json_box(json.dumps(item.get("crop_box") or [])),
                crop_path=Path(str(item.get("crop_path"))) if item.get("crop_path") else None,
                raw_text=str(item.get("raw_text") or ""),
                clean_text=str(item.get("clean_text") or ""),
                confidence=float(item.get("confidence") or 0.0),
                parse_ok=bool(item.get("parse_ok", False)),
                score=float(item.get("score") or 0.0),
                selected=bool(item.get("selected", False)),
            )
        )
    return attempts


def apply_crop_retry_result(detection: Detection, attempt: CropOcrAttempt, confidence_threshold: float) -> None:
    detection.raw_text = attempt.raw_text
    detection.clean_text = attempt.clean_text
    detection.confidence = attempt.confidence
    if is_gapfill_detection(detection):
        detection.confidence = min(detection.confidence, confidence_threshold - 0.01)
    detection.parse_ok = attempt.parse_ok
    detection.crop_box = attempt.crop_box
    detection.crop_path = attempt.crop_path


def should_accept_crop_retry(
    current: CropOcrAttempt,
    candidate: CropOcrAttempt,
    confidence_threshold: float,
) -> bool:
    if candidate.variant == current.variant:
        return False
    if candidate.clean_text == current.clean_text and candidate.confidence <= current.confidence + 0.02:
        return False

    current_complete = current.parse_ok and "/" in current.clean_text and current.confidence >= confidence_threshold
    candidate_complete = candidate.parse_ok and "/" in candidate.clean_text
    if candidate_complete and not current_complete:
        return candidate.score >= current.score - 8.0
    if candidate_complete and current_complete:
        current_alpha = alpha_suffix_stem(current.clean_text)
        candidate_numbered = suffix_stem_number(candidate.clean_text)
        if (
            current_alpha is not None
            and candidate_numbered is not None
            and current_alpha[0] == candidate_numbered[0]
            and current_alpha[1] == candidate_numbered[1]
            and candidate.confidence >= current.confidence - 0.12
        ):
            return True
        return candidate.score >= current.score + 14.0 and candidate.confidence >= current.confidence - 0.08
    return candidate.score >= current.score + 22.0


def refine_current_crop_ocr(
    image: np.ndarray,
    detections: list[Detection],
    crops_dir: Path,
    ocr: Any | None,
    confidence_threshold: float,
) -> None:
    if ocr is None:
        return

    candidates = [
        detection
        for detection in detections
        if detection_needs_current_crop_refine(detection, confidence_threshold)
    ]
    if not candidates:
        return

    retry_dir = crops_dir.parent / "crop_current_retries"
    retry_dir.mkdir(parents=True, exist_ok=True)

    def refine_priority(detection: Detection) -> tuple[int, float]:
        clean_text = normalize_ocr_text(detection.clean_text)
        parse_failed = clean_text and ("/" not in clean_text or parse_call_number(clean_text) is None)
        informative_failure = parse_failed and len(clean_text) >= 4
        return (0 if informative_failure else 1, crop_ocr_candidate_score(clean_text, detection.confidence))

    for detection in sorted(candidates, key=refine_priority)[:6]:
        current = CropOcrAttempt(
            variant="current",
            crop_box=detection.crop_box,
            crop_path=detection.crop_path,
            raw_text=detection.raw_text,
            clean_text=normalize_ocr_text(detection.clean_text),
            confidence=detection.confidence,
            parse_ok=parse_call_number(normalize_ocr_text(detection.clean_text)) is not None,
            score=crop_ocr_candidate_score(detection.clean_text, detection.confidence),
        )

        path = retry_dir / f"det_{detection.index:03d}_current_ocr.jpg"
        write_variant_crop(image, detection.crop_box, path)
        raw_text, confidence = run_ocr(ocr, path)
        clean_text = normalize_ocr_text(raw_text)
        candidate = CropOcrAttempt(
            variant="current_crop_ocr",
            crop_box=detection.crop_box,
            crop_path=path,
            raw_text=raw_text,
            clean_text=clean_text,
            confidence=confidence,
            parse_ok=parse_call_number(clean_text) is not None,
            score=crop_ocr_candidate_score(clean_text, confidence),
        )

        attempts = [current, candidate]
        if should_accept_crop_retry(current, candidate, confidence_threshold):
            apply_crop_retry_result(detection, candidate, confidence_threshold)
            candidate.selected = True
        else:
            current.selected = True
        detection.ocr_attempts = attempts


def refine_detection_ocr_with_crop_retries(
    image: np.ndarray,
    detections: list[Detection],
    crops_dir: Path,
    ocr: Any | None,
    confidence_threshold: float,
) -> None:
    if ocr is None:
        return

    retry_dir = crops_dir.parent / "crop_retries"
    retried_count = 0
    max_retry_detections = 4
    for detection in detections:
        needs_retry = detection_needs_crop_retry(image, detection, confidence_threshold)
        if not needs_retry or retried_count >= max_retry_detections:
            detection.ocr_attempts = [
                CropOcrAttempt(
                    variant="current",
                    crop_box=detection.crop_box,
                    crop_path=detection.crop_path,
                    raw_text=detection.raw_text,
                    clean_text=detection.clean_text,
                    confidence=detection.confidence,
                    parse_ok=detection.parse_ok,
                    score=crop_ocr_candidate_score(detection.clean_text, detection.confidence),
                    selected=True,
                )
            ]
            continue

        retried_count += 1
        retry_dir.mkdir(parents=True, exist_ok=True)
        current = CropOcrAttempt(
            variant="current",
            crop_box=detection.crop_box,
            crop_path=detection.crop_path,
            raw_text=detection.raw_text,
            clean_text=normalize_ocr_text(detection.clean_text),
            confidence=detection.confidence,
            parse_ok=parse_call_number(normalize_ocr_text(detection.clean_text)) is not None,
            score=crop_ocr_candidate_score(detection.clean_text, detection.confidence),
        )
        attempts = [current]

        for variant, box in crop_retry_variant_boxes(image.shape, detections, detection):
            path = retry_dir / f"det_{detection.index:03d}_{variant}.jpg"
            write_variant_crop(image, box, path)
            raw_text, confidence = run_ocr(ocr, path)
            clean_text = normalize_ocr_text(raw_text)
            if is_gapfill_detection(detection):
                confidence = min(confidence, confidence_threshold - 0.01)
            attempts.append(
                CropOcrAttempt(
                    variant=variant,
                    crop_box=box,
                    crop_path=path,
                    raw_text=raw_text,
                    clean_text=clean_text,
                    confidence=confidence,
                    parse_ok=parse_call_number(clean_text) is not None,
                    score=crop_ocr_candidate_score(clean_text, confidence),
                )
            )

        best = max(attempts, key=lambda item: (item.score, item.confidence))
        if should_accept_crop_retry(current, best, confidence_threshold):
            apply_crop_retry_result(detection, best, confidence_threshold)
            best.selected = True
        else:
            current.selected = True
        detection.ocr_attempts = attempts


def infer_thin_gap_segments(
    image: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    typical_w: float,
) -> list[tuple[int, int]]:
    image_h, image_w = image.shape[:2]
    x0 = max(0, min(image_w - 1, x0))
    x1 = max(0, min(image_w, x1))
    y0 = max(0, min(image_h - 1, y0))
    y1 = max(0, min(image_h, y1))
    if x1 - x0 < 18 or y1 - y0 < 40:
        return []

    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    roi_top = max(0, int(gray.shape[0] * 0.06))
    roi_bottom = max(roi_top + 20, int(gray.shape[0] * 0.98))
    roi = gray[roi_top:roi_bottom, :]
    if roi.size == 0:
        return []

    profile = roi.mean(axis=0)
    if len(profile) < 12:
        return []
    smooth = np.convolve(profile, np.ones(5, dtype=float) / 5, mode="same")
    darkness_limit = float(np.percentile(smooth, 30))
    min_spacing = max(9, int(typical_w * 0.28))

    minima: list[tuple[int, float]] = []
    for i in range(2, len(smooth) - 2):
        if smooth[i] > darkness_limit:
            continue
        if smooth[i] <= smooth[i - 1] and smooth[i] <= smooth[i + 1]:
            if not minima or i - minima[-1][0] > min_spacing:
                minima.append((i, float(smooth[i])))
            elif smooth[i] < minima[-1][1]:
                minima[-1] = (i, float(smooth[i]))

    if len(minima) < 2:
        return []

    boundaries = [x0 + item[0] for item in minima]
    segments: list[tuple[int, int]] = []
    min_w = max(8, int(typical_w * 0.25))
    max_w = max(22, int(typical_w * 1.20))
    pad = max(1, min(4, int(typical_w * 0.08)))
    for left, right in zip(boundaries, boundaries[1:]):
        width = right - left
        if width < min_w or width > max_w:
            continue
        segments.append((max(0, left - pad), min(image_w, right + pad)))
    return segments


def choose_visible_missing_numbers(missing_numbers: list[int], segment_count: int) -> list[int]:
    if segment_count <= 0 or not missing_numbers:
        return []
    if segment_count >= len(missing_numbers):
        return missing_numbers
    if segment_count == 1:
        return [missing_numbers[len(missing_numbers) // 2]]
    chosen: list[int] = []
    last_index = len(missing_numbers) - 1
    for i in range(segment_count):
        source_index = int(round(i * last_index / (segment_count - 1)))
        chosen.append(missing_numbers[source_index])
    return chosen


# Gap-filling detection. When two neighboring detections leave a book-width
# gap, the system adds a yellow suspected box so the user can review possible
# missed books without treating it as a confirmed wrong-shelf result.
def densify_horizontal_detections(
    image: np.ndarray,
    detections: list[Detection],
    band_box: tuple[int, int, int, int],
    crops_dir: Path,
) -> list[Detection]:
    """Add yellow placeholder crops in large horizontal gaps likely to contain missed call numbers."""
    if len(detections) < 4:
        return detections

    ordered = sorted(detections, key=lambda item: item.center_x)
    widths = [float(item.crop_box[2]) for item in ordered if item.crop_box[2] > 0]
    heights = [float(item.crop_box[3]) for item in ordered if item.crop_box[3] > 0]
    if not widths or not heights:
        return detections

    median_w = median(widths)
    median_h = median(heights)
    if median_w <= 0 or median_h <= 0:
        return detections

    centers = [item.center_x for item in ordered]
    gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1) if centers[i + 1] > centers[i]]
    if not gaps:
        return detections

    small_gap = percentile(gaps, 0.30)
    target_spacing = small_gap if small_gap > 0 else median_w * 1.45
    target_spacing = max(28.0, min(95.0, max(target_spacing, median_w * 1.18)))

    _, band_y, _, band_h = band_box
    lower_label_y = max(0, band_y - band_h * 0.18)
    lower_label_bottom_y = band_y + band_h * 0.42
    anchor_detections = [
        item
        for item in ordered
        if item.confidence >= 0.78
        and item.crop_box[1] >= lower_label_y
        and item.crop_box[1] + item.crop_box[3] >= lower_label_bottom_y
        and ("/" in item.clean_text or item.clean_text.startswith(("D66", "D669")))
    ]
    if len(anchor_detections) < 4:
        anchor_detections = [
            item
            for item in ordered
            if item.crop_box[1] >= lower_label_y
            and item.crop_box[1] + item.crop_box[3] >= lower_label_bottom_y
        ]
    if len(anchor_detections) < 4:
        return detections

    y_values = [float(item.crop_box[1]) for item in anchor_detections]
    bottom_values = [float(item.crop_box[1] + item.crop_box[3]) for item in anchor_detections]
    anchor_heights = [float(item.crop_box[3]) for item in anchor_detections if item.crop_box[3] > 0]
    typical_h = median(anchor_heights) or median_h
    crop_y0 = int(max(0, percentile(y_values, 0.15) - 6))
    crop_y1 = int(min(image.shape[0], percentile(bottom_values, 0.85) + 6))

    max_h = max(80, int(typical_h * 1.45))
    min_h = max(45, int(typical_h * 0.75))
    if crop_y1 - crop_y0 > max_h:
        center_y = int(median([item.crop_box[1] + item.crop_box[3] / 2 for item in anchor_detections]))
        crop_y0 = max(0, center_y - max_h // 2)
        crop_y1 = min(image.shape[0], crop_y0 + max_h)
    if crop_y1 - crop_y0 < min_h:
        center_y = (crop_y0 + crop_y1) // 2
        crop_y0 = max(0, center_y - min_h // 2)
        crop_y1 = min(image.shape[0], crop_y0 + min_h)

    estimated_w = int(max(28, min(90, median_w * 1.30)))
    inserted: list[Detection] = []

    red_mask = build_red_mask(image)

    def has_red_label_support(x0: int, x1: int) -> bool:
        rx0 = max(0, min(image.shape[1] - 1, x0))
        rx1 = max(0, min(image.shape[1], x1))
        if rx1 <= rx0:
            return False
        ry0 = max(0, min(image.shape[0] - 1, band_y))
        ry1 = max(0, min(image.shape[0], band_y + band_h))
        if ry1 <= ry0:
            return False
        label_region = red_mask[ry0:ry1, rx0:rx1]
        return np.count_nonzero(label_region) > max(8, int(label_region.size * 0.01))

    def append_gap_candidate(
        center_x: float,
        candidate_w: int | None = None,
        inferred_text: str = "",
        allow_without_red: bool = False,
    ) -> None:
        box_w = candidate_w or estimated_w
        x0 = int(max(0, center_x - estimated_w / 2))
        if candidate_w is not None:
            x0 = int(max(0, center_x - box_w / 2))
        x1 = int(min(image.shape[1], center_x + box_w / 2))
        if x1 - x0 < 18:
            return
        has_label = has_red_label_support(x0, x1)
        if not allow_without_red and not has_label:
            return
        crop_box = (x0, crop_y0, x1 - x0, crop_y1 - crop_y0)
        if allow_without_red and not has_label and crop_has_low_text_information(image, crop_box):
            return
        max_iou = 0.58 if inferred_text or allow_without_red else 0.42
        if any(detection_iou(crop_box, item.crop_box) > max_iou for item in ordered + inserted):
            return
        inserted.append(
            Detection(
                index=0,
                red_box=(x0, crop_y0, 1, crop_y1 - crop_y0),
                crop_box=crop_box,
                raw_text=inferred_text,
                clean_text=inferred_text,
                parse_ok=parse_call_number(inferred_text) is not None if inferred_text else False,
                reason="根据相邻薄书编号补充的疑似漏检位置" if inferred_text else "根据相邻书号间距补充的疑似漏检位置",
            )
        )

    first = ordered[0]
    leading_missing_count = int(round(first.center_x / target_spacing)) - 1
    leading_missing_count = max(0, min(leading_missing_count, 3))
    for offset in range(leading_missing_count, 0, -1):
        append_gap_candidate(first.center_x - target_spacing * offset)

    for left, right in zip(ordered, ordered[1:]):
        gap = right.center_x - left.center_x
        left_x, _, left_w, _ = left.crop_box
        right_x, _, _, _ = right.crop_box
        free_gap = right_x - (left_x + left_w)
        left_suffix = call_number_numeric_suffix(left.clean_text)
        right_suffix = call_number_numeric_suffix(right.clean_text)
        if (
            gap > median_w * 1.18
            and free_gap > max(10.0, median_w * 0.42)
            and left_suffix is not None
            and right_suffix is not None
            and left_suffix[0] == right_suffix[0]
            and 1 < right_suffix[1] - left_suffix[1] <= 6
        ):
            missing_numbers = list(range(left_suffix[1] + 1, right_suffix[1]))
            missing_count = len(missing_numbers)
            free_left = left_x + left_w
            free_right = right_x
            analysis_left = int(max(0, free_left - max(2, median_w * 0.08)))
            analysis_right = int(min(image.shape[1], free_right + max(10, median_w * 0.42)))
            thin_segments = infer_thin_gap_segments(
                image,
                crop_y0,
                crop_y1,
                analysis_left,
                analysis_right,
                median_w,
            )
            if thin_segments:
                visible_numbers = choose_visible_missing_numbers(missing_numbers, len(thin_segments))
                for number, (seg_x0, seg_x1) in zip(visible_numbers, thin_segments):
                    inferred_text = f"{left_suffix[0]}{number}"
                    append_gap_candidate(
                        (seg_x0 + seg_x1) / 2,
                        max(24, seg_x1 - seg_x0 + 10),
                        inferred_text,
                        allow_without_red=True,
                    )

        if gap <= target_spacing * 1.55 and free_gap <= median_w * 1.15:
            continue
        missing_count = int(round(gap / target_spacing)) - 1
        missing_count = max(0, min(missing_count, 8))
        allow_without_red = gap >= target_spacing * 1.95 or free_gap >= median_w * 1.45
        for offset in range(1, missing_count + 1):
            center_x = left.center_x + gap * offset / (missing_count + 1)
            append_gap_candidate(center_x, allow_without_red=allow_without_red)

    last = ordered[-1]
    trailing_missing_count = int(round((image.shape[1] - last.center_x) / target_spacing)) - 1
    trailing_missing_count = max(0, min(trailing_missing_count, 3))
    for offset in range(1, trailing_missing_count + 1):
        append_gap_candidate(last.center_x + target_spacing * offset)

    if not inserted:
        return detections

    combined = sorted(ordered + inserted, key=lambda item: item.center_x)
    for index, detection in enumerate(combined, start=1):
        detection.index = index
        save_detection_crop(image, detection, crops_dir, force=True)
    return combined


# OCR candidate construction. PaddleOCR can return several text fragments from
# one crop, so this stage joins plausible fragments and filters out noisy text.
def build_ocr_candidates(ocr_lines: list[OcrLine], allow_recovery: bool = True) -> list[OcrCandidate]:
    normal_lines = [line for line in ocr_lines if normalize_ocr_text(line.text)]
    prefix_samples: list[tuple[float, str]] = []
    for line in normal_lines:
        clean_text = normalize_ocr_text(line.text)
        if parse_call_number(clean_text) is not None:
            prefix_samples.append((line.box[1] + line.box[3] / 2, clean_text.split("/", 1)[0]))

    candidates: list[OcrCandidate] = []
    digit_lines: list[OcrLine] = []
    for line in sorted(normal_lines, key=lambda item: (item.box[1], item.box[0])):
        clean_text = normalize_ocr_text(line.text)
        if re.fullmatch(r"\d+", clean_text):
            digit_lines.append(line)
            continue

        recovered = clean_text
        if allow_recovery and parse_call_number(recovered) is None:
            recovered = recover_weak_call_number(clean_text, line, prefix_samples)

        if not recovered or ("/" not in recovered and parse_call_number(recovered) is None):
            continue

        candidates.append(
            OcrCandidate(
                raw_text=line.text,
                clean_text=recovered,
                confidence=line.confidence,
                box=line.box,
            )
        )

    merge_isolated_digit_lines(candidates, digit_lines)
    infer_adjacent_duplicate_suffixes(candidates)
    return candidates


def recover_weak_call_number(clean_text: str, line: OcrLine, prefix_samples: list[tuple[float, str]]) -> str:
    if not prefix_samples:
        return ""

    letters_digits = re.sub(r"[^A-Z0-9]", "", clean_text)
    if len(letters_digits) < 3:
        return ""

    y_center = line.box[1] + line.box[3] / 2
    _, prefix = min(prefix_samples, key=lambda item: abs(item[0] - y_center))
    suffix_match = re.search(r"([A-Z]{2,}\d*)$", letters_digits)
    if not suffix_match:
        return ""

    suffix = suffix_match.group(1)
    if len(suffix) > 3:
        suffix = suffix[-2:]
    recovered = f"{prefix}/{suffix}"
    return recovered if parse_call_number(recovered) is not None else ""


def merge_boxes(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x0 = min(ax, bx)
    y0 = min(ay, by)
    x1 = max(ax + aw, bx + bw)
    y1 = max(ay + ah, by + bh)
    return (x0, y0, x1 - x0, y1 - y0)


def merge_isolated_digit_lines(candidates: list[OcrCandidate], digit_lines: list[OcrLine]) -> None:
    merge_left_side_2026_digit_lines(candidates, digit_lines)

    for digit in digit_lines:
        digit_text = normalize_ocr_text(digit.text)
        if digit_text == "2026":
            continue
        dx, dy, dw, dh = digit.box
        digit_y = dy + dh / 2
        digit_x = dx + dw / 2

        possible: list[tuple[float, OcrCandidate]] = []
        for candidate in candidates:
            if candidate.clean_text.endswith(digit_text):
                continue
            cx, cy, cw, ch = candidate.box
            cand_y = cy + ch / 2
            cand_right = cx + cw
            y_gap = abs(digit_y - cand_y)
            if y_gap > max(ch, dh) * 0.9:
                continue
            if digit_x < cand_right - cw * 0.15:
                continue
            x_gap = max(0.0, dx - cand_right)
            possible.append((y_gap + x_gap * 0.02, candidate))

        if not possible:
            continue

        _, target = min(possible, key=lambda item: item[0])
        target.raw_text = f"{target.raw_text} {digit.text}"
        target.clean_text = normalize_ocr_text(f"{target.clean_text}{digit_text}")
        target.confidence = min(target.confidence, digit.confidence)
        target.box = merge_boxes(target.box, digit.box)


def merge_left_side_2026_digit_lines(candidates: list[OcrCandidate], digit_lines: list[OcrLine]) -> None:
    for digit in digit_lines:
        digit_text = normalize_ocr_text(digit.text)
        if digit_text != "2026":
            continue

        dx, dy, dw, dh = digit.box
        digit_y = dy + dh / 2
        digit_x = dx + dw / 2

        possible: list[tuple[float, OcrCandidate]] = []
        for candidate in candidates:
            if candidate.clean_text.endswith("2026"):
                continue
            if parse_call_number(candidate.clean_text) is None or "/" not in candidate.clean_text:
                continue

            cx, cy, cw, ch = candidate.box
            cand_y = cy + ch / 2
            if abs(digit_y - cand_y) > max(ch, dh) * 1.25:
                continue

            # The 2026 block belongs to the call number on its right; never attach
            # it to a candidate whose main text is already to the left of 2026.
            if digit_x > cx + cw * 0.35:
                continue

            merged = normalize_ocr_text(f"{candidate.clean_text}{digit_text}")
            if parse_call_number(merged) is None:
                continue
            x_gap = max(0.0, cx - (dx + dw))
            possible.append((abs(digit_y - cand_y) + x_gap * 0.03, candidate))

        if not possible:
            continue

        _, target = min(possible, key=lambda item: item[0])
        target.raw_text = f"{target.raw_text} {digit.text}"
        target.clean_text = normalize_ocr_text(f"{target.clean_text}{digit_text}")
        target.confidence = min(target.confidence, digit.confidence)
        target.box = merge_boxes(target.box, digit.box)


def infer_adjacent_duplicate_suffixes(candidates: list[OcrCandidate]) -> None:
    ordered = sorted(candidates, key=lambda item: item.box[1])
    for i in range(len(ordered) - 1):
        upper = ordered[i]
        lower = ordered[i + 1]
        if upper.clean_text != lower.clean_text:
            continue
        if re.search(r"\d$", upper.clean_text):
            continue

        suffix = upper.clean_text.split("/", 1)[1] if "/" in upper.clean_text else ""
        if not (1 <= len(suffix) <= 2 and suffix.isalpha()):
            continue

        inferred = f"{upper.clean_text}1"
        if parse_call_number(inferred) is None:
            continue
        upper.raw_text = f"{upper.raw_text} [inferred 1]"
        upper.clean_text = inferred
        upper.confidence = min(upper.confidence, 0.80)


def valid_detection_count(detections: list[Detection]) -> int:
    return sum(1 for detection in detections if detection.parse_ok)


def prune_unread_gapfill_detections(image: np.ndarray, detections: list[Detection], image_width: int) -> list[Detection]:
    if not detections:
        return detections

    pruned: list[Detection] = []
    for detection in detections:
        is_gapfill = detection.reason.startswith("根据相邻书号间距") or detection.reason.startswith("根据相邻薄书编号")
        if is_gapfill:
            if not detection.clean_text:
                continue
            if detection.confidence <= 0.0 and crop_has_low_text_information(image, detection.crop_box):
                continue
            pruned.append(detection)
            continue
        pruned.append(detection)

    if len(pruned) == len(detections):
        return detections

    for index, detection in enumerate(pruned, start=1):
        detection.index = index
    return pruned


def prune_blank_unread_detections(image: np.ndarray, detections: list[Detection]) -> list[Detection]:
    if len(detections) < 6:
        return detections

    valid_count = sum(1 for item in detections if item.parse_ok and item.clean_text)
    if valid_count < 6:
        return detections

    pruned: list[Detection] = []
    for detection in detections:
        if detection.clean_text or detection.raw_text or detection.confidence > 0:
            pruned.append(detection)
            continue
        if not crop_has_low_text_information(image, detection.crop_box):
            pruned.append(detection)

    if len(pruned) == len(detections):
        return detections

    for index, detection in enumerate(pruned, start=1):
        detection.index = index
    return pruned


def crop_has_low_text_information(image: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return True
    crop = image[max(0, y) : max(0, y) + h, max(0, x) : max(0, x) + w]
    if crop.size == 0:
        return True
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    roi = gray[max(0, int(gray.shape[0] * 0.15)) :, :]
    if roi.size == 0:
        return True
    std = float(roi.std())
    edges = cv2.Canny(roi, 50, 150)
    edge_ratio = float((edges > 0).mean())
    return std < 48.0 and edge_ratio < 0.08


def prune_contained_narrow_duplicate_detections(detections: list[Detection]) -> list[Detection]:
    if len(detections) < 2:
        return detections

    keep = [True] * len(detections)
    for i, narrow in enumerate(detections):
        if not narrow.clean_text or parse_call_number(narrow.clean_text) is None:
            continue
        nx, ny, nw, nh = narrow.crop_box
        if nw <= 0 or nh <= 0:
            continue
        for j, wide in enumerate(detections):
            if i == j or not keep[i]:
                continue
            wx, wy, ww, wh = wide.crop_box
            if ww <= 0 or wh <= 0:
                continue
            if ww < max(nw * 1.8, nw + 18):
                continue
            overlap_x = max(0, min(nx + nw, wx + ww) - max(nx, wx))
            overlap_y = max(0, min(ny + nh, wy + wh) - max(ny, wy))
            if overlap_x / max(1, nw) < 0.72:
                continue
            if overlap_y / max(1, min(nh, wh)) < 0.60:
                continue
            if narrow.clean_text == wide.clean_text:
                keep[i] = False
                continue
            if maybe_transfer_contained_narrow_text(narrow, wide):
                keep[i] = False

    if all(keep):
        return detections

    pruned = [detection for detection, should_keep in zip(detections, keep) if should_keep]
    for index, detection in enumerate(pruned, start=1):
        detection.index = index
    return pruned


def maybe_transfer_contained_narrow_text(narrow: Detection, wide: Detection) -> bool:
    narrow_parts = split_call_number_suffix(narrow.clean_text)
    wide_parts = split_call_number_suffix(wide.clean_text)
    if narrow_parts is None or wide_parts is None:
        return False
    narrow_prefix, narrow_suffix = narrow_parts
    wide_prefix, wide_suffix = wide_parts
    if narrow_prefix != wide_prefix:
        return False
    if not re.fullmatch(r"[A-Z]{1,4}\d?", narrow_suffix):
        return False
    if len(wide_suffix) < len(narrow_suffix) + 2:
        return False
    if narrow.confidence < max(0.70, wide.confidence - 0.08):
        return False

    wide.raw_text = f"{wide.raw_text} [contained narrow text: {narrow.clean_text}]"
    wide.clean_text = narrow.clean_text
    wide.confidence = min(wide.confidence, narrow.confidence, 0.90)
    wide.parse_ok = parse_call_number(wide.clean_text) is not None
    return wide.parse_ok


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def filter_spatial_outliers(
    detections: list[Detection],
    image_shape: tuple[int, int, int],
    order: str,
) -> list[Detection]:
    if len(detections) < 6:
        return detections

    image_h, image_w = image_shape[:2]
    widths = [d.crop_box[2] for d in detections if d.crop_box[2] > 0]
    heights = [d.crop_box[3] for d in detections if d.crop_box[3] > 0]
    median_w = median([float(value) for value in widths])
    median_h = median([float(value) for value in heights])
    if median_w <= 0 or median_h <= 0:
        return detections

    if order == "x_asc":
        aligned_values = [d.crop_box[1] + d.crop_box[3] / 2 for d in detections]
        aligned_median = median(aligned_values)
        aligned_limit = max(median_h * 3.2, image_h * 0.12)
    else:
        aligned_values = [d.crop_box[0] + d.crop_box[2] / 2 for d in detections]
        aligned_median = median(aligned_values)
        aligned_limit = max(median_w * 3.2, image_w * 0.10)

    filtered: list[Detection] = []
    for detection in detections:
        x, y, w, h = detection.crop_box
        center = y + h / 2 if order == "x_asc" else x + w / 2
        if abs(center - aligned_median) > aligned_limit:
            continue
        if h > median_h * 3.5 or h < median_h * 0.25:
            continue
        if w > median_w * 5.0 or w < median_w * 0.20:
            continue
        filtered.append(detection)

    min_remaining = max(4, int(len(detections) * 0.65))
    if len(filtered) < min_remaining:
        return detections

    for index, detection in enumerate(filtered, start=1):
        detection.index = index
    return filtered


# Result rendering and report helpers. These functions draw colored boxes,
# export JSON summaries, and build the crop-diagnostic HTML pages used during
# manual review.
def color_for_status(status: str) -> tuple[int, int, int]:
    if status == "green":
        return (0, 180, 0)
    if status == "red":
        return (0, 0, 255)
    return (0, 210, 255)


def annotate_image(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    annotated = image.copy()
    for position, detection in enumerate(detections, start=1):
        color = color_for_status(detection.status)
        x, y, w, h = detection.crop_box
        rx, ry, rw, rh = detection.red_box
        cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), color, 2)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        label = detection.clean_text or "UNREAD"
        if detection.status == "red" and detection.recommended_position is not None:
            label = f"{label} -> #{detection.recommended_position}"
        label = f"{position}: {label}"

        text_y = max(20, y - 8)
        cv2.putText(
            annotated,
            label[:40],
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def write_report_csv(path: Path, detections: list[Detection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "actual_position",
                "raw_text",
                "clean_text",
                "confidence",
                "parse_ok",
                "status",
                "reason",
                "recommended_position",
                "red_box",
                "crop_box",
                "crop_path",
                "ocr_attempts",
            ]
        )
        for i, detection in enumerate(detections, start=1):
            writer.writerow(
                [
                    i,
                    detection.raw_text,
                    detection.clean_text,
                    f"{detection.confidence:.4f}",
                    detection.parse_ok,
                    detection.status,
                    detection.reason,
                    detection.recommended_position or "",
                    json.dumps(detection.red_box, ensure_ascii=False),
                    json.dumps(detection.crop_box, ensure_ascii=False),
                    str(detection.crop_path or ""),
                    json.dumps(
                        [crop_ocr_attempt_to_dict(attempt) for attempt in detection.ocr_attempts],
                        ensure_ascii=False,
                    ),
                ]
            )


def write_summary(
    path: Path,
    image_path: Path,
    detections: list[Detection],
    ocr_enabled: bool,
    elapsed_seconds: float = 0.0,
    rotate_mode: str = "",
    warnings: list[str] | None = None,
) -> None:
    counts = {"green": 0, "yellow": 0, "red": 0}
    for detection in detections:
        counts[detection.status] = counts.get(detection.status, 0) + 1

    parsed_texts = [d.clean_text for d in detections if d.parse_ok]
    recommended = sorted(parsed_texts, key=lambda text: call_number_order_key(text))

    data = {
        "image": str(image_path),
        "ocr_enabled": ocr_enabled,
        "rotate_mode": rotate_mode,
        "total_detected": len(detections),
        "green": counts.get("green", 0),
        "yellow": counts.get("yellow", 0),
        "red": counts.get("red", 0),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "warnings": warnings or [],
        "actual_order": [d.clean_text for d in detections],
        "recommended_order": recommended,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def recommended_order(detections: list[Detection]) -> list[str]:
    parsed_texts = [d.clean_text for d in detections if d.parse_ok]
    return sorted(parsed_texts, key=lambda text: call_number_order_key(text))


def markdown_join(values: list[str]) -> str:
    return " -> ".join(value for value in values if value) if values else "无"


def estimate_vertical_spine_deviation(image: np.ndarray, max_side: int) -> tuple[float, float, int]:
    working, _ = resize_to_max_side(image, max_side)
    h, w = working.shape[:2]
    if w < h * 1.15:
        return 0.0, 0.0, 0

    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 170)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=80, maxLineGap=12)
    deviations: list[float] = []
    if lines is None:
        return 0.0, 0.0, 0

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(value) for value in line]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 80:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        deviation = abs(abs(angle) - 90)
        if deviation < 35:
            deviations.append(deviation)

    if len(deviations) < 60:
        return 0.0, 0.0, len(deviations)
    return float(np.median(deviations)), float(np.percentile(deviations, 90)), len(deviations)


def should_skip_bad_position_image(image: np.ndarray, max_side: int) -> bool:
    median_deviation, p90_deviation, line_count = estimate_vertical_spine_deviation(image, max_side)
    severe_tilt = line_count >= 160 and median_deviation >= 5.0 and p90_deviation >= 10.0
    strong_perspective_tail = line_count >= 800 and median_deviation >= 2.5 and p90_deviation >= 7.0
    return severe_tilt or strong_perspective_tail


def evaluate_pose_warnings(
    image: np.ndarray,
    mask: np.ndarray,
    detections: list[Detection],
    rotate_mode: str,
    max_side: int,
) -> list[str]:
    warnings: list[str] = []
    if rotate_mode != "none":
        return warnings

    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return warnings

    meaningful = [item for item in detections if item.clean_text or item.parse_ok]
    if len(meaningful) >= 8:
        right_edge = max(item.crop_box[0] + item.crop_box[2] for item in meaningful)
        band_box = make_horizontal_code_band_box(image.shape, mask)
        right_still_has_red = False
        if band_box is not None:
            _, by, _, bh = band_box
            x0 = min(w - 1, max(0, int(right_edge + w * 0.02)))
            x1 = min(w, int(w * 0.98))
            if x1 > x0:
                red_tail = mask[by : by + bh, x0:x1]
                right_still_has_red = np.count_nonzero(red_tail) > max(20, red_tail.size * 0.003)

        if right_edge < w * 0.82 and right_still_has_red:
            warnings.append(
                "右侧仍有红色标签但识别框提前结束，可能受拍摄角度或透视影响漏识别右侧书籍；建议正对书架重拍，必要时用精细模式复核。"
            )

    median_deviation, p90_deviation, line_count = estimate_vertical_spine_deviation(image, max_side)
    if not warnings and line_count >= 80 and (median_deviation >= 3.0 or p90_deviation >= 7.0):
        warnings.append(
            "检测到画面有轻微倾斜，当前结果可参考；若右侧或边缘书籍漏识别，建议让手机尽量平行书架后重拍。"
        )

    return warnings


def format_seconds(seconds: float) -> str:
    return f"{seconds:.1f}s"


def relative_markdown_path(path: Path, base: Path) -> str:
    resolved_path = path.resolve()
    resolved_base = base.resolve()
    try:
        rel = resolved_path.relative_to(resolved_base)
        return rel.as_posix()
    except ValueError:
        pass

    try:
        rel = Path(os.path.relpath(resolved_path, resolved_base))
        return rel.as_posix()
    except ValueError:
        # Windows cannot compute a relative path across different drives.
        # A file URI keeps Markdown reports valid when users choose any output folder.
        return resolved_path.as_uri()


def relative_html_path(path: Path, base: Path) -> str:
    return html.escape(relative_markdown_path(path, base), quote=True)


def write_crop_diagnostics_html(
    path: Path,
    image_path: Path,
    detections: list[Detection],
    warnings: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = path.parent

    def image_link(label: str, image_file: Path) -> str:
        if not image_file.exists():
            return ""
        src = relative_html_path(image_file, base)
        safe_label = html.escape(label)
        return (
            "<figure>"
            f"<a href=\"{src}\"><img src=\"{src}\" alt=\"{safe_label}\"></a>"
            f"<figcaption>{safe_label}</figcaption>"
            "</figure>"
        )

    overview = "".join(
        [
            image_link("annotated", base / "annotated.jpg"),
            image_link("rotated", base / "rotated.jpg"),
            image_link("bottom code band", base / "bottom_code_band.jpg"),
            image_link("code strip", base / "code_strip.jpg"),
        ]
    )

    warning_items = "".join(f"<li>{html.escape(item)}</li>" for item in (warnings or []))
    if not warning_items:
        warning_items = "<li>none</li>"

    cards: list[str] = []
    for detection in detections:
        attempts = detection.ocr_attempts or [
            CropOcrAttempt(
                variant="final",
                crop_box=detection.crop_box,
                crop_path=detection.crop_path,
                raw_text=detection.raw_text,
                clean_text=detection.clean_text,
                confidence=detection.confidence,
                parse_ok=detection.parse_ok,
                score=crop_ocr_candidate_score(detection.clean_text, detection.confidence),
                selected=True,
            )
        ]
        selected = next((attempt for attempt in attempts if attempt.selected), attempts[0])
        selected_src = relative_html_path(selected.crop_path, base) if selected.crop_path else ""
        selected_img = (
            f"<a href=\"{selected_src}\"><img class=\"selected-img\" src=\"{selected_src}\" alt=\"selected crop\"></a>"
            if selected_src
            else "<div class=\"missing-img\">no crop</div>"
        )

        box_text = html.escape(json.dumps(detection.crop_box, ensure_ascii=False))
        reason = html.escape(detection.reason or "")
        raw_text = html.escape(detection.raw_text or "")
        clean_text = html.escape(detection.clean_text or "UNREAD")
        status = html.escape(detection.status or "yellow")
        cards.append(
            f"<article class=\"det {status}\">"
            "<div class=\"det-head\">"
            f"<h2>#{detection.index} {clean_text}</h2>"
            f"<span class=\"status\">{status}</span>"
            "</div>"
            "<div class=\"det-main\">"
            f"<div>{selected_img}</div>"
            "<dl>"
            f"<dt>raw</dt><dd>{raw_text or 'UNREAD'}</dd>"
            f"<dt>confidence</dt><dd>{detection.confidence:.4f}</dd>"
            f"<dt>parse</dt><dd>{str(detection.parse_ok).lower()}</dd>"
            f"<dt>crop box</dt><dd class=\"mono\">{box_text}</dd>"
            f"<dt>reason</dt><dd>{reason}</dd>"
            "</dl>"
            "</div>"
            "</article>"
        )

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Crop diagnostics - {html.escape(image_path.name)}</title>
  <style>
    :root {{ --ink:#17201b; --muted:#68736c; --line:#d9e0dc; --paper:#f7faf8; --panel:#fff; --green:#157348; --yellow:#9b6500; --red:#b73535; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif; color:var(--ink); background:var(--paper); }}
    header {{ position:sticky; top:0; z-index:2; padding:14px 18px; border-bottom:1px solid var(--line); background:rgba(247,250,248,.96); }}
    h1 {{ margin:0 0 4px; font-size:20px; }}
    .meta, figcaption, .box {{ color:var(--muted); font-size:12px; }}
    .overview {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; padding:16px 18px; }}
    figure {{ margin:0; }}
    figure img {{ width:100%; max-height:420px; object-fit:contain; border:1px solid var(--line); background:white; }}
    .warnings {{ margin:0 18px 14px; padding:10px 18px; border:1px solid var(--line); background:white; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:14px; padding:0 18px 22px; }}
    .det {{ border:1px solid var(--line); border-radius:8px; background:var(--panel); overflow:hidden; }}
    .det.green {{ border-color:rgba(21,115,72,.45); }}
    .det.yellow {{ border-color:rgba(155,101,0,.65); }}
    .det.red {{ border-color:rgba(183,53,53,.72); }}
    .det-head {{ display:flex; align-items:center; justify-content:space-between; gap:8px; padding:10px 12px; border-bottom:1px solid var(--line); }}
    .det h2 {{ margin:0; font-size:16px; word-break:break-all; }}
    .status {{ font-weight:800; }}
    .green .status {{ color:var(--green); }} .yellow .status {{ color:var(--yellow); }} .red .status {{ color:var(--red); }}
    .det-main {{ display:grid; grid-template-columns:150px 1fr; gap:10px; padding:10px 12px; }}
    .selected-img {{ display:block; width:150px; height:190px; object-fit:contain; border:1px solid var(--line); background:white; }}
    dl {{ display:grid; grid-template-columns:86px 1fr; gap:4px 8px; margin:0; font-size:13px; }}
    dt {{ color:var(--muted); }} dd {{ margin:0; word-break:break-all; }}
    .mono {{ font-family:Consolas,"SFMono-Regular",monospace; }}
    .missing-img {{ display:grid; place-items:center; width:150px; height:190px; border:1px dashed var(--line); color:var(--muted); background:#fafafa; }}
    @media (max-width:640px) {{ .grid {{ grid-template-columns:1fr; padding:0 10px 18px; }} .overview {{ padding:12px 10px; }} .det-main {{ grid-template-columns:1fr; }} .selected-img,.missing-img {{ width:100%; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Crop diagnostics</h1>
    <div class="meta">{html.escape(image_path.name)} | detections {len(detections)} | generated {time.strftime('%Y-%m-%d %H:%M:%S')}</div>
  </header>
  <section class="overview">{overview}</section>
  <ul class="warnings">{warning_items}</ul>
  <main class="grid">{''.join(cards)}</main>
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def write_crop_diagnostics_index(path: Path, results: list[ImageRunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for result in results:
        result_dir = result.result_dir or result.output_dir / result.image_path.stem
        diagnostics_path = result_dir / "crop_diagnostics.html"
        if not diagnostics_path.exists():
            continue
        green = sum(1 for item in result.detections if item.status == "green")
        yellow = sum(1 for item in result.detections if item.status == "yellow")
        red = sum(1 for item in result.detections if item.status == "red")
        link = relative_html_path(diagnostics_path, path.parent)
        rows.append(
            "<tr>"
            f"<td><a href=\"{link}\">{html.escape(result.image_path.name)}</a></td>"
            f"<td>{html.escape(result.rotate_mode or '')}</td>"
            f"<td>{len(result.detections)}</td>"
            f"<td>{green}</td>"
            f"<td>{yellow}</td>"
            f"<td>{red}</td>"
            f"<td>{result.elapsed_seconds:.1f}s</td>"
            "</tr>"
        )

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Crop diagnostics index</title>
  <style>
    body {{ margin:22px; font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif; color:#17201b; }}
    table {{ width:100%; border-collapse:collapse; }}
    th,td {{ border:1px solid #d9e0dc; padding:7px 9px; font-size:13px; text-align:left; }}
    th {{ background:#f2f6f3; }}
    a {{ color:#126a4a; font-weight:700; }}
  </style>
</head>
<body>
  <h1>Crop diagnostics index</h1>
  <table>
    <tr><th>image</th><th>rotate</th><th>total</th><th>green</th><th>yellow</th><th>red</th><th>time</th></tr>
    {''.join(rows)}
  </table>
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "default"


def result_dir_for_image(output_dir: Path, image_path: Path, output_variant: str | None = None) -> Path:
    if not output_variant:
        return output_dir / image_path.stem
    return output_dir / f"{image_path.stem}__{safe_name(output_variant)}"


def parse_json_box(value: str) -> tuple[int, int, int, int]:
    try:
        items = json.loads(value)
        if isinstance(items, list) and len(items) == 4:
            return tuple(int(item) for item in items)  # type: ignore[return-value]
    except Exception:
        pass
    return (0, 0, 0, 0)


def load_cached_result(image_path: Path, output_dir: Path, output_variant: str | None = None) -> ImageRunResult | None:
    result_dir = result_dir_for_image(output_dir, image_path, output_variant)
    summary_path = result_dir / "summary.json"
    report_path = result_dir / "report.csv"
    annotated_path = result_dir / "annotated.jpg"
    if not (summary_path.exists() and report_path.exists() and annotated_path.exists()):
        return None

    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    detections: list[Detection] = []
    try:
        with report_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                index = int(row.get("actual_position") or len(detections) + 1)
                crop_path_text = row.get("crop_path") or ""
                crop_path = Path(crop_path_text) if crop_path_text else None
                detections.append(
                    Detection(
                        index=index,
                        red_box=parse_json_box(row.get("red_box") or ""),
                        crop_box=parse_json_box(row.get("crop_box") or ""),
                        crop_path=crop_path,
                        raw_text=row.get("raw_text") or "",
                        clean_text=row.get("clean_text") or "",
                        confidence=float(row.get("confidence") or 0.0),
                        parse_ok=(row.get("parse_ok") or "").lower() == "true",
                        status=row.get("status") or "yellow",
                        reason=row.get("reason") or "",
                        recommended_position=int(row["recommended_position"]) if row.get("recommended_position") else None,
                        ocr_attempts=parse_crop_ocr_attempts(row.get("ocr_attempts") or ""),
                    )
                )
    except Exception:
        return None

    return ImageRunResult(
        image_path=image_path,
        output_dir=output_dir,
        detections=detections,
        ocr_enabled=bool(data.get("ocr_enabled", False)),
        rotate_mode=str(data.get("rotate_mode") or ""),
        elapsed_seconds=float(data.get("elapsed_seconds") or 0.0),
        result_dir=result_dir,
        from_cache=True,
        warnings=[str(item) for item in data.get("warnings") or [] if item],
    )


def evaluate_quality(detections: list[Detection], ocr_enabled: bool) -> tuple[str, float, str]:
    if not detections:
        return "bad", 0.0, "未检测到红色标签或索书号。"

    confidences = [d.confidence for d in detections if d.confidence > 0]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    yellow_count = sum(1 for d in detections if d.status == "yellow")
    yellow_ratio = yellow_count / len(detections)

    if not ocr_enabled:
        return "unknown", 0.0, "OCR 未启用，需要人工复核。"

    score = avg_confidence * 100 - yellow_ratio * 45
    if len(detections) < 4:
        score -= 20
    score = max(0.0, min(100.0, score))

    if score >= 85 and yellow_ratio <= 0.15:
        return "good", score, "图像质量较好。"
    if score >= 60:
        return "fair", score, "图像可用，但部分位置建议人工复核。"
    return "bad", score, "图像质量较差，建议从更正的角度重新拍摄。"


def write_markdown_report(path: Path, results: list[ImageRunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# 图书馆书架智能巡检原型演示报告",
        "",
        "| 图片 | 方向 | 质量 | 检测书本数 | 错序/可疑数 | 耗时 | 结果图 |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]

    total_elapsed = sum(result.elapsed_seconds for result in results)
    quality_data: dict[Path, tuple[str, float, str]] = {}
    for result in results:
        quality, score, prompt = evaluate_quality(result.detections, result.ocr_enabled)
        quality_data[result.image_path] = (quality, score, prompt)
        suspect_count = sum(1 for d in result.detections if d.status in {"red", "yellow"})
        result_link = relative_markdown_path(result.annotated_path, path.parent)
        lines.append(
            f"| {result.image_path.name} | {result.rotate_mode or 'unknown'} | {quality} | "
            f"{len(result.detections)} | {suspect_count} | {format_seconds(result.elapsed_seconds)} | {result_link} |"
        )

    lines.extend(["", f"- 本批图片总耗时：{format_seconds(total_elapsed)}", "", "## 识别详情", ""])

    for result in results:
        quality, score, prompt = quality_data[result.image_path]
        suspect_positions = [
            str(i)
            for i, detection in enumerate(result.detections, start=1)
            if detection.status in {"red", "yellow"}
        ]
        actual = [d.clean_text or "UNREAD" for d in result.detections]
        recommended = recommended_order(result.detections)
        original_link = relative_markdown_path(result.image_path, path.parent)
        annotated_link = relative_markdown_path(result.annotated_path, path.parent)

        lines.extend(
            [
                f"### {result.image_path.name}",
                "",
                f"![{result.image_path.name} 原图]({original_link})",
                "",
                f"![{result.image_path.name} 识别结果]({annotated_link})",
                "",
                f"- 自动选择方向：{result.rotate_mode or 'unknown'}",
                f"- 图像质量：{quality}，评分：{score:.1f}",
                f"- 检测书本数：{len(result.detections)}",
                f"- 本图耗时：{format_seconds(result.elapsed_seconds)}",
                f"- 错序/可疑位置：{', '.join(suspect_positions) if suspect_positions else '无'}",
                f"- 系统提示：{prompt}",
                f"- 当前顺序：{markdown_join(actual)}",
                f"- 推荐顺序：{markdown_join(recommended)}",
                "",
            ]
        )

        if suspect_positions:
            lines.extend(["| 位置 | 识别结果 | 状态 | 原因 | 推荐位置 |", "| ---: | --- | --- | --- | ---: |"])
            for i, detection in enumerate(result.detections, start=1):
                if detection.status not in {"red", "yellow"}:
                    continue
                recommended_position = detection.recommended_position or ""
                lines.append(
                    f"| {i} | {detection.clean_text or 'UNREAD'} | {detection.status} | {detection.reason} | {recommended_position} |"
                )
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# Single-image pipeline. This is the core workflow used by both the command
# line and Flask Web server: load image, detect bottom call-number boxes,
# crop/OCR each box, correct obvious OCR errors, sort, then export results.
def inspect_image(
    image_path: Path,
    output_dir: Path,
    result_dir: Path | None,
    rotate: str,
    max_side: int,
    crop_right_ratio: float,
    y_padding_ratio: float,
    ocr_mode: str,
    confidence_threshold: float,
    ocr: Any | None = None,
    crop_retry: bool = False,
) -> tuple[list[Detection], str, float, Path, list[str]]:
    start_time = time.perf_counter()
    original = read_image(image_path)
    base_dir = result_dir or output_dir / image_path.stem
    if should_skip_bad_position_image(original, max_side):
        working, _ = resize_to_max_side(original, max_side)
        mask = build_red_mask(working)
        detections: list[Detection] = []
        elapsed_seconds = time.perf_counter() - start_time
        write_image(base_dir / "rotated.jpg", working)
        write_image(base_dir / "red_mask.jpg", mask)
        write_image(base_dir / "annotated.jpg", working)
        write_report_csv(base_dir / "report.csv", detections)
        warnings = ["图像距离、倾斜或清晰度不适合可靠识别，本次未识别书号。建议靠近书架并让手机尽量平行书架后重新拍摄。"]
        write_summary(
            base_dir / "summary.json",
            image_path,
            detections,
            ocr is not None,
            elapsed_seconds,
            "bad_position",
            warnings,
        )
        write_crop_diagnostics_html(base_dir / "crop_diagnostics.html", image_path, detections, warnings)
        return detections, "bad_position", elapsed_seconds, base_dir, warnings

    used_rotate = choose_rotation_with_ocr(original, max_side, crop_right_ratio, ocr) if rotate == "auto" else rotate
    rotated = rotate_image(original, used_rotate)
    working, _ = resize_to_max_side(rotated, max_side)

    mask = build_red_mask(working)
    red_boxes = detect_red_label_boxes(mask)
    red_column = find_main_red_column(mask)

    fallback_detections: list[Detection] = []
    for box in red_boxes:
        crop_box = make_crop_box(working.shape, box, crop_right_ratio, y_padding_ratio)
        if crop_box is None:
            continue
        fallback_detections.append(Detection(index=len(fallback_detections) + 1, red_box=box, crop_box=crop_box))

    # After left rotation, real shelf order from left to right equals y descending.
    fallback_detections.sort(key=lambda d: d.center_y, reverse=True)
    for i, detection in enumerate(fallback_detections, start=1):
        detection.index = i

    crops_dir = base_dir / "crops"
    crop_boxes = [d.crop_box for d in fallback_detections]
    crop_paths = save_crops(working, crop_boxes, crops_dir)
    for detection, crop_path in zip(fallback_detections, crop_paths):
        detection.crop_path = crop_path

    detections = fallback_detections
    horizontal_band_box = make_horizontal_code_band_box(working.shape, mask) if used_rotate == "none" else None
    strip_box = make_code_strip_box(working.shape, red_column, crop_right_ratio)
    if strip_box is not None and horizontal_band_box is None:
        sx, sy, sw, sh = strip_box
        strip = working[sy : sy + sh, sx : sx + sw]
        strip_path = base_dir / "code_strip.jpg"
        write_image(strip_path, strip)
        if ocr is not None:
            ocr_lines = run_ocr_lines(ocr, strip_path)
            strip_detections = detections_from_ocr_strip(working, strip_box, ocr_lines, crops_dir)
            strip_detections = filter_spatial_outliers(strip_detections, working.shape, "y_desc")
            if strip_detections:
                detections = strip_detections

    if used_rotate == "none" and ocr is not None:
        band_box = horizontal_band_box or make_bottom_code_band_box(working.shape)
        if band_box is not None:
            bx, by, bw, bh = band_box
            band = working[by : by + bh, bx : bx + bw]
            band_path = base_dir / "bottom_code_band.jpg"
            write_image(band_path, band)
            band_lines = run_ocr_lines(ocr, band_path)
            band_detections = detections_from_ocr_strip(
                working,
                band_box,
                band_lines,
                crops_dir,
                order="x_asc",
                allow_recovery=False,
                keep_invalid=False,
                min_confidence=0.18,
            )
            band_detections = filter_spatial_outliers(band_detections, working.shape, "x_asc")
            band_detections = densify_horizontal_detections(working, band_detections, band_box, crops_dir)
            band_valid_count = valid_detection_count(band_detections)
            current_valid_count = valid_detection_count(detections)
            if band_valid_count > current_valid_count or (
                band_valid_count == current_valid_count and len(band_detections) > len(detections)
            ):
                detections = band_detections

    if used_rotate == "none" and horizontal_band_box is not None:
        bottom_text_detections = detections_from_bottom_text_band(working, mask, horizontal_band_box, crops_dir)
        if should_use_bottom_text_band_detections(
            current=detections,
            bottom_text=bottom_text_detections,
            image_shape=working.shape,
        ):
            detections = bottom_text_detections

    pending_ocr: list[tuple[Detection, Path]] = []
    for detection in detections:
        if detection.raw_text:
            continue
        if detection.crop_path is None:
            continue
        ocr_input_path = detection.crop_path
        crop_x, crop_y, crop_w, crop_h = detection.crop_box
        needs_rotated_input = crop_h > crop_w * 1.35
        if (crop_retry or needs_rotated_input) and ocr is not None:
            ocr_input_dir = crops_dir.parent / "crop_ocr_inputs"
            ocr_input_dir.mkdir(parents=True, exist_ok=True)
            ocr_input_path = ocr_input_dir / f"det_{detection.index:03d}_initial.jpg"
            write_variant_crop(working, detection.crop_box, ocr_input_path)
        pending_ocr.append((detection, ocr_input_path))

    batch_results = run_ocr_batch(ocr, [path for _, path in pending_ocr])
    for (detection, _ocr_input_path), (raw_text, confidence) in zip(pending_ocr, batch_results):
        clean_text = normalize_ocr_text(raw_text)
        detection.raw_text = raw_text
        detection.clean_text = clean_text
        if is_gapfill_detection(detection):
            confidence = min(confidence, confidence_threshold - 0.01)
        detection.confidence = confidence
        detection.parse_ok = parse_call_number(clean_text) is not None

    detections = prune_blank_unread_detections(working, detections)
    detections = prune_unread_gapfill_detections(working, detections, working.shape[1])
    detections = prune_overlapping_duplicate_detections(detections)
    detections = prune_contained_narrow_duplicate_detections(detections)

    if not crop_retry:
        refine_current_crop_ocr(working, detections, crops_dir, ocr, confidence_threshold)
        detections = prune_blank_unread_detections(working, detections)
        detections = prune_unread_gapfill_detections(working, detections, working.shape[1])
        detections = prune_overlapping_duplicate_detections(detections)
        detections = prune_contained_narrow_duplicate_detections(detections)

    if crop_retry:
        refine_detection_ocr_with_crop_retries(working, detections, crops_dir, ocr, confidence_threshold)

        detections = prune_blank_unread_detections(working, detections)
        detections = prune_unread_gapfill_detections(working, detections, working.shape[1])
        detections = prune_overlapping_duplicate_detections(detections)
        detections = prune_contained_narrow_duplicate_detections(detections)
    correct_contextual_suffix_ocr_errors(detections)
    correct_isolated_narrow_prefix_ocr(detections)
    apply_sort_status(detections, confidence_threshold)
    if remove_boundary_order_outliers(detections):
        apply_sort_status(detections, confidence_threshold)
    downgrade_isolated_prefix_outlier_status(detections)
    downgrade_uncertain_sort_status(detections)
    mark_uncertain_horizontal_detections(detections, working.shape[1])

    warnings = evaluate_pose_warnings(working, mask, detections, used_rotate, max_side)
    annotated = annotate_image(working, detections)
    elapsed_seconds = time.perf_counter() - start_time
    write_image(base_dir / "rotated.jpg", working)
    write_image(base_dir / "red_mask.jpg", mask)
    write_image(base_dir / "annotated.jpg", annotated)
    write_report_csv(base_dir / "report.csv", detections)
    write_summary(base_dir / "summary.json", image_path, detections, ocr is not None, elapsed_seconds, used_rotate, warnings)
    write_crop_diagnostics_html(base_dir / "crop_diagnostics.html", image_path, detections, warnings)

    return detections, used_rotate, elapsed_seconds, base_dir, warnings


# Batch and command-line helpers.
def collect_images(input_paths: list[Path]) -> list[Path]:
    images: list[Path] = []
    for input_path in input_paths:
        images.extend(collect_images_from_path(input_path))

    seen: set[Path] = set()
    unique: list[Path] = []
    for image in images:
        resolved = image.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(image)
    return unique


def collect_images_from_path(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise ValueError(f"Input path does not exist: {input_path}")


ProgressCallback = Callable[[str, Any], None]


def run_inspection(
    input_paths: list[Path],
    output_dir: Path,
    rotate: str = "auto",
    max_side: int = 2400,
    crop_right_ratio: float = 0.32,
    y_padding_ratio: float = 0.18,
    ocr_mode: str = "auto",
    confidence_threshold: float = 0.65,
    report_name: str = "demo_report.md",
    markdown_report: bool = True,
    ocr: Any | None = None,
    load_ocr_if_needed: bool = True,
    output_variant: str | None = None,
    use_cache: bool = False,
    crop_retry: bool = False,
    progress: ProgressCallback | None = None,
) -> list[ImageRunResult]:
    images = collect_images(input_paths)
    if not images:
        raise ValueError("No images found.")

    if progress is not None:
        progress("batch_start", {"total": len(images)})

    ocr_load_elapsed = 0.0
    if ocr_mode != "off" and ocr is None and load_ocr_if_needed:
        if progress is not None:
            progress("ocr_load_start", None)
        ocr_load_start = time.perf_counter()
        ocr = load_paddle_ocr()
        ocr_load_elapsed = time.perf_counter() - ocr_load_start
        if progress is not None:
            progress("ocr_load_done", {"elapsed_seconds": ocr_load_elapsed, "loaded": ocr is not None})
        if ocr_mode == "on" and ocr is None:
            raise RuntimeError("PaddleOCR is not installed. Install it or run with --ocr off.")
    elif ocr_mode == "on" and ocr is None:
        raise RuntimeError("PaddleOCR is not loaded. Please check the OCR environment or run with --ocr off.")
    elif progress is not None and ocr_mode != "off":
        progress("ocr_reused", None)

    batch_start = time.perf_counter()
    results: list[ImageRunResult] = []
    for index, image_path in enumerate(images, start=1):
        if progress is not None:
            progress("image_start", {"index": index, "total": len(images), "image_path": image_path})

        result_dir = result_dir_for_image(output_dir, image_path, output_variant)
        if use_cache:
            cached = load_cached_result(image_path, output_dir, output_variant)
            if cached is not None:
                results.append(cached)
                if progress is not None:
                    progress("image_cached", cached)
                continue

        detections, used_rotate, elapsed_seconds, actual_result_dir, warnings = inspect_image(
            image_path=image_path,
            output_dir=output_dir,
            result_dir=result_dir,
            rotate=rotate,
            max_side=max_side,
            crop_right_ratio=crop_right_ratio,
            y_padding_ratio=y_padding_ratio,
            ocr_mode=ocr_mode,
            confidence_threshold=confidence_threshold,
            ocr=ocr,
            crop_retry=crop_retry,
        )
        result = ImageRunResult(
            image_path=image_path,
            output_dir=output_dir,
            detections=detections,
            ocr_enabled=ocr is not None,
            rotate_mode=used_rotate,
            elapsed_seconds=elapsed_seconds,
            result_dir=actual_result_dir,
            warnings=warnings,
        )
        results.append(result)
        if progress is not None:
            progress("image_done", result)

    if markdown_report:
        report_path = output_dir / report_name
        write_markdown_report(report_path, results)
        if progress is not None:
            progress("report_done", {"report_path": report_path})

    write_crop_diagnostics_index(output_dir / "crop_diagnostics_index.html", results)

    if progress is not None:
        progress(
            "batch_done",
            {
                "elapsed_seconds": time.perf_counter() - batch_start,
                "ocr_load_elapsed_seconds": ocr_load_elapsed,
                "total": len(images),
            },
        )

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MVP library shelf inspector: red-label detection, call-number crop, OCR, and order check."
    )
    parser.add_argument("input", type=Path, nargs="+", help="Image file(s) or folder(s).")
    parser.add_argument("--output", type=Path, default=Path("runs"), help="Output folder.")
    parser.add_argument("--rotate", choices=["auto", "left", "right", "none"], default="auto")
    parser.add_argument("--max-side", type=int, default=2400, help="Resize longest side before processing.")
    parser.add_argument("--crop-right-ratio", type=float, default=0.32)
    parser.add_argument("--y-padding-ratio", type=float, default=0.18)
    parser.add_argument("--ocr", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument("--crop-retry", action="store_true", help="Retry suspicious crops with narrow crop variants.")
    parser.add_argument("--report-name", default="demo_report.md", help="Markdown report filename in output folder.")
    parser.add_argument("--no-markdown-report", action="store_true", help="Skip writing the Markdown summary report.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    def print_progress(event: str, payload: Any) -> None:
        result = payload if isinstance(payload, ImageRunResult) else None
        if event == "image_done" and result is not None:
            green = sum(1 for d in result.detections if d.status == "green")
            yellow = sum(1 for d in result.detections if d.status == "yellow")
            red = sum(1 for d in result.detections if d.status == "red")
            print(
                f"{result.image_path.name}: rotate={result.rotate_mode} "
                f"detected={len(result.detections)} green={green} yellow={yellow} red={red} "
                f"time={format_seconds(result.elapsed_seconds)}"
            )
        elif event == "report_done":
            print(f"Markdown report: {args.output / args.report_name}")
        elif event == "batch_done" and isinstance(payload, dict):
            print(f"Total time: {format_seconds(float(payload.get('elapsed_seconds', 0.0)))}")

    run_inspection(
        input_paths=args.input,
        output_dir=args.output,
        rotate=args.rotate,
        max_side=args.max_side,
        crop_right_ratio=args.crop_right_ratio,
        y_padding_ratio=args.y_padding_ratio,
        ocr_mode=args.ocr,
        confidence_threshold=args.confidence_threshold,
        report_name=args.report_name,
        markdown_report=not args.no_markdown_report,
        crop_retry=args.crop_retry,
        progress=print_progress,
    )


if __name__ == "__main__":
    main()
