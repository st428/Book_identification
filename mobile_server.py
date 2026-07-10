"""Flask Web/mobile entry point for the final official-OCR shelf inspector.

The server receives one or more uploaded shelf images, reuses a resident
PaddleOCR instance, runs the selected inspection mode, and returns JSON data
for the browser UI. Runtime uploads and cached results are kept under
stage5_mobile_results so the project code stays separate from user output.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import socket
import threading
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for

from shelf_inspector_fast import (
    ImageRunResult,
    call_number_order_key,
    load_paddle_ocr,
    ocr_model_cache_tag,
    run_inspection,
)


# Project paths and Web runtime configuration.
WORKSPACE = Path(__file__).resolve().parent
UPLOAD_DIR = WORKSPACE / "stage5_mobile_results" / "uploads"
RESULT_DIR = WORKSPACE / "stage5_mobile_results" / "results"
REVIEW_DIR = WORKSPACE / "stage5_mobile_results" / "reviews"
REVIEW_JSONL_PATH = REVIEW_DIR / "review_records.jsonl"
REVIEW_CSV_PATH = REVIEW_DIR / "review_records.csv"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REVIEW_FIELDS = [
    "review_id",
    "result_id",
    "image_name",
    "crop_id",
    "ocr_text",
    "corrected_text",
    "status",
    "note",
    "created_at",
]
REVIEW_STATUSES = {"confirmed", "corrected", "unreadable", "ignored"}

# Web stable line: official PaddleOCR only. Bump this when Web behavior,
# post-processing, crop rules, or cache semantics change.
WEB_CACHE_VERSION = "web_official_ocr_stable_20260608_v5"
WEB_OCR_POLICY = "official_paddleocr_v5_mobile"
WEB_MODES = {
    "standard": {
        "label": "普通模式",
        "max_side": 1200,
        "crop_retry": False,
    },
    "fine": {
        "label": "精细模式",
        "max_side": 1600,
        "crop_retry": True,
    },
}

# Flask app and OCR singleton state. PaddleOCR is expensive to construct, so
# the Web server keeps one instance alive and shares it across requests.
app = Flask(__name__, template_folder="mobile_web", static_folder="mobile_web", static_url_path="/static")

ocr_lock = threading.Lock()
ocr_ready = threading.Event()
ocr_ready.set()
ocr_cache: object | None = None
ocr_loading = False
ocr_error: str | None = None
review_lock = threading.Lock()


def local_ip() -> str:
    """Return the LAN IP shown to phone users after the server starts."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def ensure_ocr_loaded() -> object | None:
    """Load official PaddleOCR once and let concurrent requests wait for it."""
    global ocr_cache, ocr_loading, ocr_error
    should_load = False
    with ocr_lock:
        if ocr_cache is not None:
            return ocr_cache
        if ocr_loading:
            should_load = False
        else:
            ocr_loading = True
            ocr_error = None
            ocr_ready.clear()
            should_load = True

    if not should_load:
        ocr_ready.wait()
        with ocr_lock:
            return ocr_cache

    try:
        loaded = load_paddle_ocr(use_env=False)
    except Exception as exc:
        with ocr_lock:
            ocr_loading = False
            ocr_error = str(exc)
            ocr_ready.set()
        return None

    with ocr_lock:
        ocr_cache = loaded
        ocr_loading = False
        ocr_ready.set()
    return ocr_cache


def preload_ocr() -> None:
    ensure_ocr_loaded()


def result_counts(result: ImageRunResult) -> dict[str, int]:
    """Count green/yellow/red detections for front-end summary cards."""
    return {
        "green": sum(1 for item in result.detections if item.status == "green"),
        "yellow": sum(1 for item in result.detections if item.status == "yellow"),
        "red": sum(1 for item in result.detections if item.status == "red"),
        "total": len(result.detections),
    }


def safe_stem(value: str) -> str:
    """Make a short filesystem-safe stem while preserving a readable name."""
    stem = Path(value).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return cleaned[:40] or "image"


def save_uploaded_image(uploaded: object) -> Path:
    """Store uploaded image content by hash so repeated uploads can use cache."""
    filename = getattr(uploaded, "filename", "") or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"{filename or '图片'} 的格式不支持。")

    data = uploaded.read()
    if not data:
        raise ValueError(f"{filename or '图片'} 内容为空。")

    digest = hashlib.sha256(data).hexdigest()[:16]
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    image_path = UPLOAD_DIR / f"{digest}_{safe_stem(filename)}{suffix}"
    if not image_path.exists():
        image_path.write_bytes(data)
    return image_path


def result_url(path: Path) -> str:
    return url_for("result_file", filename=path.relative_to(RESULT_DIR).as_posix())


def optional_result_url(path: Path | None) -> str:
    """Return a browser URL for result files that are safely under RESULT_DIR."""
    if path is None:
        return ""
    try:
        resolved = path.resolve()
        result_root = RESULT_DIR.resolve()
        relative = resolved.relative_to(result_root)
    except Exception:
        return ""
    if not resolved.exists():
        return ""
    return url_for("result_file", filename=relative.as_posix())


def resolve_result_dir(result_id: str) -> Path | None:
    """Resolve a result_id without allowing path traversal outside RESULT_DIR."""
    if not result_id:
        return None
    try:
        root = RESULT_DIR.resolve()
        candidate = (RESULT_DIR / result_id).resolve()
        candidate.relative_to(root)
    except Exception:
        return None
    if not candidate.is_dir():
        return None
    return candidate


def export_file_for(result_id: str, export_format: str) -> tuple[Path, str] | None:
    result_dir = resolve_result_dir(result_id)
    if result_dir is None:
        return None
    normalized = export_format.lower()
    if normalized == "csv":
        return result_dir / "report.csv", "text/csv; charset=utf-8"
    if normalized == "json":
        return result_dir / "summary.json", "application/json; charset=utf-8"
    return None


def crop_id_for(item: object) -> str:
    crop_path = getattr(item, "crop_path", None)
    if crop_path:
        return Path(crop_path).stem
    index = int(getattr(item, "index", 0) or 0)
    return f"crop_{index:03d}" if index > 0 else f"crop_{uuid4().hex[:8]}"


def review_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def normalize_review_item(payload: dict[str, object], result_id: str, image_name: str) -> dict[str, str]:
    status = str(payload.get("status") or "").strip()
    if status not in REVIEW_STATUSES:
        raise ValueError(f"Unsupported review status: {status or 'empty'}")

    return {
        "review_id": f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}",
        "result_id": result_id,
        "image_name": image_name,
        "crop_id": str(payload.get("crop_id") or "").strip(),
        "ocr_text": str(payload.get("ocr_text") or "").strip(),
        "corrected_text": str(payload.get("corrected_text") or "").strip(),
        "status": status,
        "note": str(payload.get("note") or "").strip(),
        "created_at": review_timestamp(),
    }


def append_review_records(records: list[dict[str, str]]) -> None:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    with review_lock:
        with REVIEW_JSONL_PATH.open("a", encoding="utf-8", newline="\n") as jsonl_file:
            for record in records:
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        write_header = not REVIEW_CSV_PATH.exists() or REVIEW_CSV_PATH.stat().st_size == 0
        with REVIEW_CSV_PATH.open("a", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=REVIEW_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(records)


def load_review_records(result_id: str | None = None) -> list[dict[str, str]]:
    if not REVIEW_JSONL_PATH.exists():
        return []

    records: list[dict[str, str]] = []
    with review_lock:
        with REVIEW_JSONL_PATH.open("r", encoding="utf-8") as jsonl_file:
            for line in jsonl_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                if result_id and record.get("result_id") != result_id:
                    continue
                records.append({field: str(record.get(field) or "") for field in REVIEW_FIELDS})
    return records


def review_file_label() -> str:
    try:
        return REVIEW_JSONL_PATH.relative_to(WORKSPACE).as_posix()
    except ValueError:
        return str(REVIEW_JSONL_PATH)


def current_ocr_model_tag() -> str:
    try:
        return ocr_model_cache_tag(use_env=False)
    except Exception:
        return "rec_invalid"


def output_variant_for_mode(mode: str) -> str:
    """Build a cache namespace from Web version, OCR policy, and selected mode."""
    return f"{WEB_CACHE_VERSION}_{current_ocr_model_tag()}_{mode}"


def status_label(status: str) -> str:
    return {"green": "绿", "yellow": "黄", "red": "红"}.get(status, status)


def clean_reorder_suggestions(result: ImageRunResult) -> list[str]:
    """Convert red sorting results into short user-facing shelf suggestions."""
    counts = result_counts(result)
    if counts["total"] == 0:
        return ["图像距离、角度或清晰度不适合可靠识别，本次未识别书号。建议靠近书架并让手机尽量平行书架后重拍。"]
    if counts["yellow"] > 0:
        return ["存在黄色不确定项，请先人工确认黄色书号；确认后再参考调架建议。"]
    if counts["red"] == 0:
        return ["当前未检测到明显错架书籍。"]

    red_items = [(index, item) for index, item in enumerate(result.detections, start=1) if item.status == "red"]
    if len(red_items) == 2:
        (a_pos, a_item), (b_pos, b_item) = red_items
        if a_item.recommended_position == b_pos and b_item.recommended_position == a_pos:
            return [f"交换第 {a_pos} 本《{a_item.clean_text}》和第 {b_pos} 本《{b_item.clean_text}》的位置。"]

    parsed = [
        item
        for item in result.detections
        if item.parse_ok and item.confidence > 0 and call_number_order_key(item.clean_text) is not None
    ]
    target_order = sorted(parsed, key=lambda item: call_number_order_key(item.clean_text))
    suggestions: list[str] = []
    for actual_pos, item in red_items[:5]:
        target_pos = item.recommended_position
        if target_pos is None:
            continue
        if target_pos <= 1 and len(target_order) > 1:
            neighbor = target_order[1].clean_text
            suggestions.append(f"把第 {actual_pos} 本《{item.clean_text}》移动到最前面，放在《{neighbor}》前面。")
        elif target_pos >= len(target_order) and len(target_order) > 1:
            neighbor = target_order[-2].clean_text
            suggestions.append(f"把第 {actual_pos} 本《{item.clean_text}》移动到最后面，放在《{neighbor}》后面。")
        elif 1 < target_pos < len(target_order):
            before = target_order[target_pos].clean_text
            after = target_order[target_pos - 2].clean_text
            if target_pos < actual_pos:
                suggestions.append(f"把第 {actual_pos} 本《{item.clean_text}》移动到第 {target_pos} 位，放在《{before}》前面。")
            else:
                suggestions.append(f"把第 {actual_pos} 本《{item.clean_text}》移动到第 {target_pos} 位，放在《{after}》后面。")

    return suggestions or ["检测到红色错序项，但当前结果不足以生成可靠的最小移动建议，请结合标注图人工确认。"]


def detection_review_info(item: object) -> dict[str, str]:
    """Explain whether a detection is suitable for manual review or training."""
    reason = getattr(item, "reason", "") or ""
    status = getattr(item, "status", "") or ""
    confidence = float(getattr(item, "confidence", 0.0) or 0.0)
    clean_text = getattr(item, "clean_text", "") or ""
    parse_ok = bool(getattr(item, "parse_ok", False))

    if (
        reason.startswith("根据相邻薄书编号")
        or reason.startswith("根据相邻书号间距")
        or (status == "yellow" and confidence == 0.0 and clean_text and parse_ok)
    ):
        return {
            "review_label": "疑似漏检补框",
            "train_use": "不训练",
            "review_note": "由相邻书号推测生成，需人工确认真实书号和 crop 是否只包含一本书。",
        }
    if not clean_text:
        return {
            "review_label": "OCR 未识别",
            "train_use": "不训练",
            "review_note": "无有效识别文本；当前稳定版不再把这类 crop 加入训练。",
        }
    if reason.startswith("分类号与相邻书号不一致"):
        return {
            "review_label": "前缀/排架差异",
            "train_use": "人工确认",
            "review_note": "可能是 OCR 前缀错误，也可能是馆内真实排架差异。",
        }
    if confidence < 0.65 or reason.startswith("OCR 置信度较低"):
        return {
            "review_label": "低置信 OCR",
            "train_use": "人工复核",
            "review_note": "优先检查 crop 是否完整；必要时改用精细模式。",
        }
    if status == "red":
        return {
            "review_label": "疑似错架",
            "train_use": "人工确认",
            "review_note": "先确认 OCR 文本无误，再判断是否真的需要调架。",
        }
    if status == "green":
        return {
            "review_label": "自动通过",
            "train_use": "无需处理",
            "review_note": "书号识别和排序判断均未触发明显异常。",
        }
    return {
        "review_label": "人工复核",
        "train_use": "人工确认",
        "review_note": reason or "需要人工复核。",
    }


def serialize_result(result: ImageRunResult, original_name: str) -> dict[str, object]:
    """Convert one backend result into the JSON shape consumed by the UI."""
    result_dir = result.result_dir or result.output_dir / result.image_path.stem
    result_id = result_dir.name
    diagnostics_path = result_dir / "crop_diagnostics.html"
    return {
        "result_id": result_id,
        "image": original_name,
        "stored_image": result.image_path.name,
        "rotate": result.rotate_mode,
        "elapsed_seconds": round(result.elapsed_seconds, 1),
        "from_cache": result.from_cache,
        "counts": result_counts(result),
        "warnings": result.warnings,
        "annotated_url": result_url(result.annotated_path),
        "diagnostics_url": result_url(diagnostics_path) if diagnostics_path.exists() else "",
        "export_csv_url": url_for("export_result", result_id=result_id, format="csv"),
        "export_json_url": url_for("export_result", result_id=result_id, format="json"),
        "actual_order": [item.clean_text or "UNREAD" for item in result.detections],
        "order_items": [
            {
                "crop_id": crop_id_for(item),
                "crop_url": optional_result_url(item.crop_path),
                "text": item.clean_text or "UNREAD",
                "ocr_text": item.clean_text or "",
                "raw_text": item.raw_text,
                "status": item.status,
                "status_label": status_label(item.status),
                "reason": item.reason,
                "confidence": round(item.confidence, 4),
                **detection_review_info(item),
            }
            for item in result.detections
        ],
        "suggestions": clean_reorder_suggestions(result),
    }


@app.route("/")
def index() -> str:
    """Serve the single-page mobile Web interface."""
    return render_template("index.html")


@app.route("/api/status")
def status() -> object:
    """Expose OCR/cache/mode status for the browser."""
    with ocr_lock:
        state = "loaded" if ocr_cache is not None else "loading" if ocr_loading else "not_loaded"
        error = ocr_error
    return jsonify(
        {
            "ocr_status": state,
            "ocr_error": error,
            "ocr_model": current_ocr_model_tag(),
            "ocr_policy": WEB_OCR_POLICY,
            "cache_version": WEB_CACHE_VERSION,
            "modes": {key: {"label": value["label"]} for key, value in WEB_MODES.items()},
        }
    )


@app.route("/api/preload", methods=["POST"])
def preload() -> object:
    threading.Thread(target=preload_ocr, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/inspect", methods=["POST"])
def inspect() -> object:
    """Handle multi-image upload and run the chosen inspection mode."""
    uploaded_files = request.files.getlist("images") or request.files.getlist("image")
    uploaded_files = [item for item in uploaded_files if item and item.filename]
    if not uploaded_files:
        return jsonify({"ok": False, "error": "请先选择或拍摄图片。"}), 400

    mode = request.form.get("mode", "standard")
    if mode not in WEB_MODES:
        mode = "standard"
    mode_config = WEB_MODES[mode]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    job_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"

    try:
        image_paths = [save_uploaded_image(item) for item in uploaded_files]
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    original_names = [item.filename for item in uploaded_files]
    ocr = ensure_ocr_loaded()
    if ocr is None:
        with ocr_lock:
            error = ocr_error
        if error:
            return jsonify({"ok": False, "error": f"OCR 加载失败：{error}"}), 500
        return jsonify({"ok": False, "error": "OCR 正在加载，请稍后再试。"}), 503

    try:
        results = run_inspection(
            input_paths=image_paths,
            output_dir=RESULT_DIR,
            rotate="auto",
            max_side=int(mode_config["max_side"]),
            ocr_mode="auto",
            report_name=f"{job_id}_report.md",
            markdown_report=False,
            ocr=ocr,
            load_ocr_if_needed=False,
            output_variant=output_variant_for_mode(mode),
            use_cache=True,
            crop_retry=bool(mode_config["crop_retry"]),
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    items = [serialize_result(result, original_names[index]) for index, result in enumerate(results)]
    return jsonify(
        {
            "ok": True,
            "mode": mode,
            "mode_label": mode_config["label"],
            "ocr_policy": WEB_OCR_POLICY,
            "ocr_model": current_ocr_model_tag(),
            "cache_version": WEB_CACHE_VERSION,
            "total": len(items),
            "cached": sum(1 for item in items if item["from_cache"]),
            "results": items,
        }
    )


@app.route("/api/export")
def export_result() -> object:
    """Download report.csv or summary.json for a completed result directory."""
    result_id = request.args.get("result_id", "").strip()
    export_format = request.args.get("format", "csv").strip().lower()
    export_target = export_file_for(result_id, export_format)
    if export_target is None:
        return jsonify({"ok": False, "error": "结果不存在或导出格式不支持。"}), 404

    export_path, mimetype = export_target
    if not export_path.exists():
        return jsonify({"ok": False, "error": f"{export_format.upper()} 结果文件不存在。"}), 404

    return send_from_directory(
        export_path.parent,
        export_path.name,
        mimetype=mimetype,
        as_attachment=True,
        download_name=f"{result_id}.{export_format}",
    )


@app.route("/api/review", methods=["POST"])
def save_review() -> object:
    """Persist manual OCR review records as JSONL and CSV."""
    payload = request.get_json(silent=True) or {}
    result_id = str(payload.get("result_id") or "").strip()
    image_name = str(payload.get("image_name") or "").strip()
    items = payload.get("items") or []

    if resolve_result_dir(result_id) is None:
        return jsonify({"ok": False, "error": "结果不存在，无法保存复核记录。"}), 404
    if not isinstance(items, list) or not items:
        return jsonify({"ok": False, "error": "没有可保存的复核记录。"}), 400

    records: list[dict[str, str]] = []
    try:
        for item in items:
            if not isinstance(item, dict):
                continue
            records.append(normalize_review_item(item, result_id, image_name))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    if not records:
        return jsonify({"ok": False, "error": "没有可保存的复核记录。"}), 400

    append_review_records(records)
    return jsonify(
        {
            "ok": True,
            "saved_count": len(records),
            "review_file": review_file_label(),
        }
    )


@app.route("/api/reviews")
def reviews() -> object:
    """Return saved manual review records, optionally filtered by result_id."""
    result_id = request.args.get("result_id", "").strip()
    if result_id and resolve_result_dir(result_id) is None:
        return jsonify({"ok": False, "error": "结果不存在。"}), 404
    return jsonify({"ok": True, "items": load_review_records(result_id or None)})


@app.route("/results/<path:filename>")
def result_file(filename: str) -> object:
    return send_from_directory(RESULT_DIR, filename)


def main() -> None:
    """Start the Web server and warm OCR in a background thread."""
    ip = local_ip()
    print("书架智能巡检 Web 稳定版已启动")
    print(f"OCR 策略：{WEB_OCR_POLICY}（忽略 BOOK_OCR_REC_MODEL_DIR）")
    print(f"缓存版本：{WEB_CACHE_VERSION}")
    print("本机访问：http://127.0.0.1:5000")
    print(f"手机访问：http://{ip}:5000")
    print("请确保手机和电脑连接同一个网络。")
    threading.Thread(target=preload_ocr, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
