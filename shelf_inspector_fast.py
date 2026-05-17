from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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

    @property
    def annotated_path(self) -> Path:
        return (self.result_dir or self.output_dir / self.image_path.stem) / "annotated.jpg"

    @property
    def summary_path(self) -> Path:
        return (self.result_dir or self.output_dir / self.image_path.stem) / "summary.json"


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


def load_paddle_ocr() -> Any | None:
    try:
        from paddleocr import PaddleOCR
    except Exception:
        return None

    local_model_base = Path.home() / ".paddlex" / "official_models"
    local_mobile_det = local_model_base / "PP-OCRv5_mobile_det"
    local_mobile_rec = local_model_base / "PP-OCRv5_mobile_rec"

    candidates = [
        {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "PP-OCRv5_mobile_rec",
            "text_detection_model_dir": str(local_mobile_det),
            "text_recognition_model_dir": str(local_mobile_rec),
            "text_det_limit_side_len": 960,
            "enable_mkldnn": False,
            "enable_hpi": False,
            "device": "cpu",
        },
        {"use_textline_orientation": True, "lang": "ch"},
        {"use_angle_cls": True, "lang": "ch"},
        {"lang": "ch"},
    ]
    last_error: Exception | None = None
    for kwargs in candidates:
        try:
            return PaddleOCR(**kwargs)
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    return None


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
        ":": "",
        "：": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^A-Z0-9./-]", "", text)
    match = re.search(r"[A-Z][A-Z0-9.-]*/[A-Z0-9-]+", text)
    if match:
        text = match.group(0)
    text = re.sub(r"^R(?=[A-Z][0-9])", "", text)

    # Common OCR fixes in the classification-number part before '/'.
    if "/" in text:
        left, right = text.split("/", 1)
        fixed_left = []
        for i, ch in enumerate(left):
            if i > 0 and ch == "O":
                fixed_left.append("0")
            else:
                fixed_left.append(ch)
        left = "".join(fixed_left)
        if len(right) >= 2 and right[0] == "0" and right[1].isalpha():
            right = "O" + right[1:]
        right = re.sub(r"(?<=[A-Z])0(?=[A-Z])", "O", right)
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
    for item in re.findall(r"\d+|[A-Z]+|-", value):
        if item.isdigit():
            parts.append((0, int(item)))
        else:
            parts.append((1, item))
    return tuple(parts)


def parse_call_number(value: str) -> tuple[Any, ...] | None:
    match = re.match(r"^([A-Z]+)([0-9]+(?:\.[0-9]+)?)(?:-([0-9]+))?(?:/([A-Z0-9-]+))?$", value)
    if not match:
        return None

    letter, number, aux, suffix = match.groups()
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


def call_number_order_key(value: str) -> tuple[Any, ...] | None:
    parsed = parse_call_number(value)
    if parsed is None:
        return None

    letter, number_parts, _has_aux, _aux_parts, _suffix_parts = parsed
    # Prototype sorting focuses on CLC classification order. Real shelves may not
    # keep the suffix/cutter code in strict alphabetical order, as shown by the
    # newly supplied correct samples. Auxiliary-number groups are narrower, so
    # their suffix order remains useful for catching obvious local inversions.
    number_parts = number_parts[:1]
    return (letter, number_parts)


def local_aux_order_key(value: str) -> tuple[tuple[Any, ...], tuple[Any, ...]] | None:
    parsed = parse_call_number(value)
    if parsed is None:
        return None
    letter, number_parts, has_aux, aux_parts, suffix_parts = parsed
    if not has_aux:
        return None
    return (letter, number_parts, aux_parts), suffix_parts


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
        if not 0.28 <= center_ratio <= 0.78:
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
        parse_ok = parse_call_number(clean_text) is not None
        if not keep_invalid and not parse_ok:
            continue

        lx, ly, lw, lh = candidate.box
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

        crop = image[y0:y1, x0:x1]
        crop_path = crops_dir / f"ocr_crop_{detection.index:03d}.jpg"
        write_image(crop_path, crop)
        detection.crop_path = crop_path
        detections.append(detection)

    if order == "x_asc":
        detections.sort(key=lambda d: d.center_x)
    else:
        detections.sort(key=lambda d: d.center_y, reverse=True)
    for i, detection in enumerate(detections, start=1):
        detection.index = i
    return detections


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


def save_detection_crop(image: np.ndarray, detection: Detection, crops_dir: Path) -> None:
    if detection.crop_path is not None:
        return
    crops_dir.mkdir(parents=True, exist_ok=True)
    x, y, w, h = detection.crop_box
    crop = image[y : y + h, x : x + w]
    path = crops_dir / f"ocr_crop_{detection.index:03d}.jpg"
    write_image(path, crop)
    detection.crop_path = path


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
    # OCR often misses faint vertical call numbers. For filled gaps, crop the
    # full lower book-label region instead of a tiny OCR line-height box.
    crop_y0 = int(max(0, band_y + band_h * 0.42))
    crop_y1 = int(min(image.shape[0], band_y + band_h * 0.96))
    if crop_y1 - crop_y0 < max(36, median_h * 1.8):
        y_values = [float(item.crop_box[1]) for item in ordered]
        bottom_values = [float(item.crop_box[1] + item.crop_box[3]) for item in ordered]
        crop_y0 = int(max(band_y, percentile(y_values, 0.20)))
        crop_y1 = int(min(image.shape[0], percentile(bottom_values, 0.85)))

    estimated_w = int(max(22, min(85, median_w * 1.15)))
    inserted: list[Detection] = []
    for left, right in zip(ordered, ordered[1:]):
        gap = right.center_x - left.center_x
        if gap <= target_spacing * 1.75:
            continue
        missing_count = int(round(gap / target_spacing)) - 1
        missing_count = max(0, min(missing_count, 8))
        for offset in range(1, missing_count + 1):
            center_x = left.center_x + gap * offset / (missing_count + 1)
            x0 = int(max(0, center_x - estimated_w / 2))
            x1 = int(min(image.shape[1], center_x + estimated_w / 2))
            if x1 - x0 < 18:
                continue
            crop_box = (x0, crop_y0, x1 - x0, crop_y1 - crop_y0)
            if any(detection_iou(crop_box, item.crop_box) > 0.35 for item in ordered + inserted):
                continue
            inserted.append(
                Detection(
                    index=0,
                    red_box=(x0, max(0, band_y - 4), x1 - x0, 8),
                    crop_box=crop_box,
                    reason="根据相邻书号间距补充的疑似漏检位置",
                )
            )

    if not inserted:
        return detections

    combined = sorted(ordered + inserted, key=lambda item: item.center_x)
    for index, detection in enumerate(combined, start=1):
        detection.index = index
        if detection in inserted:
            detection.crop_path = None
        save_detection_crop(image, detection, crops_dir)
    return combined


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
    for digit in digit_lines:
        digit_text = normalize_ocr_text(digit.text)
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
                ]
            )


def write_summary(
    path: Path,
    image_path: Path,
    detections: list[Detection],
    ocr_enabled: bool,
    elapsed_seconds: float = 0.0,
    rotate_mode: str = "",
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
    return line_count >= 80 and median_deviation >= 8.0 and p90_deviation >= 13.0


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
) -> tuple[list[Detection], str, float, Path]:
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
        write_summary(base_dir / "summary.json", image_path, detections, ocr is not None, elapsed_seconds, "bad_position")
        return detections, "bad_position", elapsed_seconds, base_dir

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
    strip_box = make_code_strip_box(working.shape, red_column, crop_right_ratio)
    if strip_box is not None:
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
        band_box = make_horizontal_code_band_box(working.shape, mask) or make_bottom_code_band_box(working.shape)
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
            if valid_detection_count(band_detections) > valid_detection_count(detections):
                detections = band_detections

    for detection in detections:
        if detection.raw_text:
            continue
        if detection.crop_path is None:
            continue
        raw_text, confidence = run_ocr(ocr, detection.crop_path)
        clean_text = normalize_ocr_text(raw_text)
        detection.raw_text = raw_text
        detection.clean_text = clean_text
        if detection.reason.startswith("根据相邻书号间距"):
            confidence = min(confidence, confidence_threshold - 0.01)
        detection.confidence = confidence
        detection.parse_ok = parse_call_number(clean_text) is not None

    apply_sort_status(detections, confidence_threshold)
    if remove_boundary_order_outliers(detections):
        apply_sort_status(detections, confidence_threshold)
    downgrade_uncertain_sort_status(detections)

    annotated = annotate_image(working, detections)
    elapsed_seconds = time.perf_counter() - start_time
    write_image(base_dir / "rotated.jpg", working)
    write_image(base_dir / "red_mask.jpg", mask)
    write_image(base_dir / "annotated.jpg", annotated)
    write_report_csv(base_dir / "report.csv", detections)
    write_summary(base_dir / "summary.json", image_path, detections, ocr is not None, elapsed_seconds, used_rotate)

    return detections, used_rotate, elapsed_seconds, base_dir


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

        detections, used_rotate, elapsed_seconds, actual_result_dir = inspect_image(
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
        )
        result = ImageRunResult(
            image_path=image_path,
            output_dir=output_dir,
            detections=detections,
            ocr_enabled=ocr is not None,
            rotate_mode=used_rotate,
            elapsed_seconds=elapsed_seconds,
            result_dir=actual_result_dir,
        )
        results.append(result)
        if progress is not None:
            progress("image_done", result)

    if markdown_report:
        report_path = output_dir / report_name
        write_markdown_report(report_path, results)
        if progress is not None:
            progress("report_done", {"report_path": report_path})

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
        progress=print_progress,
    )


if __name__ == "__main__":
    main()
