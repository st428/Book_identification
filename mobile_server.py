from __future__ import annotations

import hashlib
import re
import socket
import threading
import time
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for

from shelf_inspector_fast import ImageRunResult, call_number_order_key, load_paddle_ocr, run_inspection


WORKSPACE = Path(__file__).resolve().parent
UPLOAD_DIR = WORKSPACE / "stage5_mobile_results" / "uploads"
RESULT_DIR = WORKSPACE / "stage5_mobile_results" / "results"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WEB_CACHE_VERSION = "web_calib20260513"

app = Flask(__name__, template_folder="mobile_web", static_folder="mobile_web", static_url_path="/static")

ocr_lock = threading.Lock()
ocr_ready = threading.Event()
ocr_ready.set()
ocr_cache: object | None = None
ocr_loading = False
ocr_error: str | None = None


def local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def ensure_ocr_loaded() -> object | None:
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
        loaded = load_paddle_ocr()
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
    return {
        "green": sum(1 for item in result.detections if item.status == "green"),
        "yellow": sum(1 for item in result.detections if item.status == "yellow"),
        "red": sum(1 for item in result.detections if item.status == "red"),
        "total": len(result.detections),
    }


def safe_stem(value: str) -> str:
    stem = Path(value).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return cleaned[:40] or "image"


def save_uploaded_image(uploaded: object) -> Path:
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


def reorder_suggestions(result: ImageRunResult) -> list[str]:
    counts = result_counts(result)
    if counts["yellow"] > 0:
        return ["存在黄色不确定识别项，建议先人工确认黄色书号，再生成调架建议。"]
    if counts["red"] == 0:
        return ["当前未检测到错序书籍，不需要调整顺序。"]

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

    return suggestions or ["检测到红色错序项，但当前结果不足以生成可靠的最简移动建议，请结合标注图人工确认。"]


def status_label(status: str) -> str:
    return {"green": "绿", "yellow": "黄", "red": "红"}.get(status, status)


def serialize_result(result: ImageRunResult, original_name: str) -> dict[str, object]:
    return {
        "image": original_name,
        "stored_image": result.image_path.name,
        "rotate": result.rotate_mode,
        "elapsed_seconds": round(result.elapsed_seconds, 1),
        "from_cache": result.from_cache,
        "counts": result_counts(result),
        "annotated_url": result_url(result.annotated_path),
        "actual_order": [item.clean_text or "UNREAD" for item in result.detections],
        "order_items": [
            {
                "text": item.clean_text or "UNREAD",
                "status": item.status,
                "status_label": status_label(item.status),
            }
            for item in result.detections
        ],
        "suggestions": reorder_suggestions(result),
    }


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/api/status")
def status() -> object:
    with ocr_lock:
        state = "loaded" if ocr_cache is not None else "loading" if ocr_loading else "not_loaded"
        error = ocr_error
    return jsonify({"ocr_status": state, "ocr_error": error})


@app.route("/api/preload", methods=["POST"])
def preload() -> object:
    threading.Thread(target=preload_ocr, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/inspect", methods=["POST"])
def inspect() -> object:
    uploaded_files = request.files.getlist("images") or request.files.getlist("image")
    uploaded_files = [item for item in uploaded_files if item and item.filename]
    if not uploaded_files:
        return jsonify({"ok": False, "error": "请先选择或拍摄图片。"}), 400

    mode = request.form.get("mode", "standard")
    max_side_by_mode = {"fast": 1200, "standard": 1600, "fine": 2000}
    max_side = max_side_by_mode.get(mode, 1600)
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
            max_side=max_side,
            ocr_mode="auto",
            report_name=f"{job_id}_report.md",
            markdown_report=True,
            ocr=ocr,
            load_ocr_if_needed=False,
            output_variant=f"{WEB_CACHE_VERSION}_{mode}",
            use_cache=True,
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    items = [serialize_result(result, original_names[index]) for index, result in enumerate(results)]
    return jsonify(
        {
            "ok": True,
            "mode": mode,
            "total": len(items),
            "cached": sum(1 for item in items if item["from_cache"]),
            "results": items,
        }
    )


@app.route("/results/<path:filename>")
def result_file(filename: str) -> object:
    return send_from_directory(RESULT_DIR, filename)


def main() -> None:
    ip = local_ip()
    print("移动端 Web 原型已启动")
    print("本机访问：http://127.0.0.1:5000")
    print(f"手机访问：http://{ip}:5000")
    print("请确保手机和电脑连接同一个 Wi-Fi。")
    threading.Thread(target=preload_ocr, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
