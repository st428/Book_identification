from __future__ import annotations

import argparse
import csv
import hashlib
import html
import random
import shutil
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_DIR = Path("stage5_mobile_results") / "results"
DEFAULT_OUTPUT_DIR = Path("stage5_mobile_results") / "rec_label_export"
DEFAULT_VERSION_KEYWORD = "web_calib20260519_posewarn_lowband_edgefill_thin_tight_prune_fast"


@dataclass
class CropRecord:
    crop_path: Path
    source_report: Path
    source_image: str
    position: str
    label: str
    confidence: float
    status: str
    reason: str
    include: bool = True
    review_note: str = ""
    crop_ocr_text: str = ""
    crop_ocr_confidence: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PaddleOCR recognition crops and draft labels.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--version-keyword",
        default=DEFAULT_VERSION_KEYWORD,
        help="Only read result folders whose path contains this keyword. Use empty string to include all.",
    )
    parser.add_argument("--include-yellow", action="store_true", help="Include yellow uncertain rows in draft labels.")
    parser.add_argument("--include-red", action="store_true", help="Include red rows in draft labels.")
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument(
        "--no-quality-review",
        action="store_true",
        help="Disable image-quality review. By default, low-information crops stay in HTML but include=0.",
    )
    parser.add_argument(
        "--verify-crop-ocr",
        action="store_true",
        help="Run OCR on each exported crop again and set include=0 when the crop OCR does not match the label.",
    )
    parser.add_argument("--ocr-match-threshold", type=float, default=0.78)
    parser.add_argument("--ocr-mismatch-min-confidence", type=float, default=0.55)
    parser.add_argument(
        "--from-draft",
        type=Path,
        help="Regenerate rec_gt_train/test from a corrected labels_draft.tsv instead of scanning reports.",
    )
    return parser.parse_args()


def read_summary_image(report_path: Path) -> str:
    summary_path = report_path.with_name("summary.json")
    if not summary_path.exists():
        return report_path.parent.name
    try:
        import json

        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return report_path.parent.name
    image = str(data.get("image") or "")
    return Path(image).name if image else report_path.parent.name


def iter_report_paths(results_dir: Path, version_keyword: str) -> list[Path]:
    reports = sorted(results_dir.rglob("report.csv"))
    if version_keyword:
        reports = [path for path in reports if version_keyword in str(path.parent)]
    return reports


def read_crop_records(report_path: Path) -> list[CropRecord]:
    source_image = read_summary_image(report_path)
    records: list[CropRecord] = []
    try:
        with report_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                crop_text = row.get("crop_path") or ""
                if not crop_text:
                    continue
                crop_path = Path(crop_text)
                if not crop_path.exists():
                    crop_path = report_path.parent / crop_text
                if not crop_path.exists():
                    continue
                label = (row.get("clean_text") or "").strip()
                confidence = float(row.get("confidence") or 0)
                records.append(
                    CropRecord(
                        crop_path=crop_path,
                        source_report=report_path,
                        source_image=source_image,
                        position=row.get("actual_position") or "",
                        label=label,
                        confidence=confidence,
                        status=(row.get("status") or "").strip(),
                        reason=(row.get("reason") or "").strip(),
                    )
                )
    except Exception as exc:
        print(f"Skip unreadable report: {report_path} ({exc})")
    return records


def should_include(record: CropRecord, args: argparse.Namespace) -> bool:
    if not record.label:
        return False
    if record.status == "green" and record.confidence >= args.min_confidence:
        return True
    if record.status == "yellow" and args.include_yellow:
        return True
    if record.status == "red" and args.include_red:
        return True
    return False


def append_review_note(record: CropRecord, note: str) -> None:
    if not note:
        return
    record.review_note = f"{record.review_note}; {note}" if record.review_note else note


def read_image_gray(path: Path) -> Any | None:
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None
    return image


def crop_quality_notes(path: Path) -> list[str]:
    gray = read_image_gray(path)
    if gray is None:
        return []

    try:
        import cv2
        import numpy as np
    except Exception:
        return []

    h, w = gray.shape[:2]
    notes: list[str] = []
    if w < 18 or h < 55:
        notes.append(f"尺寸过小 {w}x{h}")

    # Ignore a little of the top border/red tape edge; the useful call-number
    # glyphs are normally in the lower part of these vertical crops.
    roi = gray[max(0, int(h * 0.15)) :, :]
    if roi.size == 0:
        notes.append("有效区域为空")
        return notes

    mean = float(roi.mean())
    std = float(roi.std())
    edges = cv2.Canny(roi, 50, 150)
    edge_ratio = float((edges > 0).mean())
    very_dark = mean < 35 and std < 18 and edge_ratio < 0.025
    very_bright = mean > 225 and std < 18 and edge_ratio < 0.025
    low_information = std < 13 and edge_ratio < 0.02

    if very_dark:
        notes.append(f"近乎全黑 mean={mean:.1f} std={std:.1f} edge={edge_ratio:.3f}")
    elif very_bright:
        notes.append(f"近乎全白 mean={mean:.1f} std={std:.1f} edge={edge_ratio:.3f}")
    elif low_information:
        notes.append(f"纹理信息过少 std={std:.1f} edge={edge_ratio:.3f}")

    border = max(2, min(6, w // 5))
    if border > 0 and w >= 12:
        left_dark = float((roi[:, :border] < 80).mean())
        right_dark = float((roi[:, -border:] < 80).mean())
        center_dark = float((roi[:, border:-border] < 80).mean()) if w > border * 2 else 0.0
        if max(left_dark, right_dark) > 0.42 and center_dark < 0.08:
            notes.append("疑似只裁到书脊边缘/残缺")

    _ = np  # Keep linters quiet when cv2 imports numpy internally in some envs.
    return notes


def normalize_for_match(value: str) -> str:
    try:
        from shelf_inspector_fast import normalize_ocr_text

        return normalize_ocr_text(value)
    except Exception:
        return "".join(ch for ch in value.upper() if ch.isalnum() or ch in ".-/:-")


def label_match_score(expected: str, actual_raw: str) -> float:
    expected_norm = normalize_for_match(expected)
    actual_norm = normalize_for_match(actual_raw)
    if not expected_norm or not actual_norm:
        return 0.0
    if expected_norm == actual_norm:
        return 1.0
    if expected_norm in actual_norm:
        return 0.98
    if actual_norm in expected_norm and len(actual_norm) >= max(4, int(len(expected_norm) * 0.65)):
        return 0.86
    return SequenceMatcher(None, expected_norm, actual_norm).ratio()


def apply_training_review_filters(records: list[CropRecord], args: argparse.Namespace) -> None:
    if not args.no_quality_review:
        for record in records:
            notes = crop_quality_notes(record.crop_path)
            if notes:
                record.include = False
                append_review_note(record, "低质量crop: " + "；".join(notes))

    if not args.verify_crop_ocr:
        return

    try:
        from shelf_inspector_fast import load_paddle_ocr, parse_call_number, run_ocr
    except Exception as exc:
        print(f"Skip crop OCR verification: cannot import OCR helpers ({exc})")
        return

    ocr = load_paddle_ocr()
    if ocr is None:
        print("Skip crop OCR verification: PaddleOCR is unavailable.")
        return

    for index, record in enumerate(records, start=1):
        raw_text, confidence = run_ocr(ocr, record.crop_path)
        record.crop_ocr_text = raw_text
        record.crop_ocr_confidence = confidence
        score = label_match_score(record.label, raw_text)
        crop_clean = normalize_for_match(raw_text)
        crop_has_call_number = parse_call_number(crop_clean) is not None if crop_clean else False
        strong_mismatch = (
            crop_clean
            and score < args.ocr_match_threshold
            and (crop_has_call_number or confidence >= args.ocr_mismatch_min_confidence)
        )
        weak_mismatch = crop_clean and score < args.ocr_match_threshold
        if strong_mismatch:
            record.include = False
            append_review_note(
                record,
                f"crop二次OCR不匹配 score={score:.2f} text={raw_text or '<empty>'}",
            )
        elif weak_mismatch:
            append_review_note(
                record,
                f"crop二次OCR弱匹配 score={score:.2f} text={raw_text}",
            )
        elif not crop_clean:
            append_review_note(record, "crop二次OCR为空")
        if index % 50 == 0:
            print(f"Crop OCR verified: {index}/{len(records)}")


def stable_crop_name(record: CropRecord, index: int) -> str:
    digest_source = f"{record.crop_path}|{record.source_report}|{record.position}".encode("utf-8", errors="ignore")
    digest = hashlib.sha1(digest_source).hexdigest()[:10]
    source_stem = Path(record.source_image).stem[:28]
    return f"{index:06d}_{source_stem}_{record.position}_{digest}{record.crop_path.suffix.lower() or '.jpg'}"


def export_records(records: list[CropRecord], output_dir: Path, val_ratio: float, seed: int) -> None:
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    shuffled_indices = list(range(len(records)))
    rng.shuffle(shuffled_indices)
    val_count = int(round(len(records) * max(0.0, min(0.8, val_ratio))))
    val_indices = set(shuffled_indices[:val_count])

    draft_path = output_dir / "labels_draft.tsv"
    train_path = output_dir / "rec_gt_train.txt"
    test_path = output_dir / "rec_gt_test.txt"
    review_path = output_dir / "review_low_confidence.tsv"
    html_path = output_dir / "labels_review.html"

    train_rows: list[tuple[str, str]] = []
    test_rows: list[tuple[str, str]] = []
    review_rows: list[list[str]] = []

    with draft_path.open("w", encoding="utf-8", newline="") as draft_file:
        writer = csv.writer(draft_file, delimiter="\t")
        writer.writerow(
            [
                "include",
                "split",
                "image",
                "label",
                "confidence",
                "status",
                "reason",
                "review_note",
                "crop_ocr_text",
                "crop_ocr_confidence",
                "source_image",
                "source_report",
            ]
        )
        for index, record in enumerate(records, start=1):
            filename = stable_crop_name(record, index)
            dest = image_dir / filename
            if not dest.exists():
                shutil.copy2(record.crop_path, dest)
            rel_image = dest.relative_to(output_dir).as_posix()
            split = "test" if index - 1 in val_indices else "train"
            writer.writerow(
                [
                    "1" if record.include else "0",
                    split,
                    rel_image,
                    record.label,
                    f"{record.confidence:.4f}",
                    record.status,
                    record.reason,
                    record.review_note,
                    record.crop_ocr_text,
                    f"{record.crop_ocr_confidence:.4f}" if record.crop_ocr_text else "",
                    record.source_image,
                    str(record.source_report),
                ]
            )
            if record.include:
                if split == "test":
                    test_rows.append((rel_image, record.label))
                else:
                    train_rows.append((rel_image, record.label))
            if not record.include or record.status != "green" or record.confidence < 0.85:
                review_rows.append(
                    [
                        rel_image,
                        record.label,
                        f"{record.confidence:.4f}",
                        record.status,
                        record.reason,
                        record.review_note,
                        record.crop_ocr_text,
                        f"{record.crop_ocr_confidence:.4f}" if record.crop_ocr_text else "",
                        record.source_image,
                        str(record.source_report),
                    ]
                )

    write_rec_gt(train_path, train_rows)
    write_rec_gt(test_path, test_rows)
    with review_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "image",
                "label",
                "confidence",
                "status",
                "reason",
                "review_note",
                "crop_ocr_text",
                "crop_ocr_confidence",
                "source_image",
                "source_report",
            ]
        )
        writer.writerows(review_rows)
    generate_review_html(draft_path, html_path)


def write_rec_gt(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        for image_path, label in rows:
            f.write(f"{image_path}\t{label}\n")


def read_draft_rows(draft_path: Path) -> list[dict[str, str]]:
    with draft_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def generate_review_html(draft_path: Path, html_path: Path) -> None:
    rows = read_draft_rows(draft_path)
    body_rows: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        include = (row.get("include") or "1").strip()
        checked = " checked" if include in {"1", "true", "True", "yes", "Y", "y"} else ""
        image = row.get("image") or ""
        label = row.get("label") or ""
        split = row.get("split") or "train"
        confidence = row.get("confidence") or ""
        status = row.get("status") or ""
        reason = row.get("reason") or ""
        review_note = row.get("review_note") or ""
        crop_ocr_text = row.get("crop_ocr_text") or ""
        crop_ocr_confidence = row.get("crop_ocr_confidence") or ""
        source_image = row.get("source_image") or ""
        source_report = row.get("source_report") or ""
        body_rows.append(
            "        <tr>"
            f"<td class=\"row-no\">{row_index}</td>"
            f"<td><input class=\"include\" type=\"checkbox\"{checked}></td>"
            f"<td><select class=\"split\"><option value=\"train\"{' selected' if split != 'test' else ''}>train</option>"
            f"<option value=\"test\"{' selected' if split == 'test' else ''}>test</option></select></td>"
            f"<td class=\"crop\"><img src=\"{html.escape(image, quote=True)}\" alt=\"crop {row_index}\"></td>"
            f"<td><input class=\"label\" value=\"{html.escape(label, quote=True)}\" spellcheck=\"false\"></td>"
            f"<td class=\"mono confidence\">{html.escape(confidence)}</td>"
            f"<td class=\"status {html.escape(status)}\">{html.escape(status)}</td>"
            f"<td class=\"reason\">{html.escape(reason)}</td>"
            f"<td class=\"review-note\">{html.escape(review_note)}</td>"
            f"<td class=\"ocr-text\">{html.escape(crop_ocr_text)}</td>"
            f"<td class=\"ocr-conf mono\">{html.escape(crop_ocr_confidence)}</td>"
            f"<td class=\"source\">{html.escape(source_image)}</td>"
            f"<td class=\"path\">{html.escape(source_report)}</td>"
            f"<td class=\"image-path\">{html.escape(image)}</td>"
            "</tr>"
        )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>书号识别训练标注表</title>
  <style>
    :root {{
      --ink: #17201b;
      --muted: #6a746d;
      --line: #d7ddd8;
      --paper: #f8faf8;
      --green: #16823f;
      --yellow: #a36b00;
      --red: #b93a32;
    }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
      background: var(--paper);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(248, 250, 248, 0.96);
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
    }}
    button {{
      min-height: 36px;
      padding: 0 14px;
      border: 0;
      border-radius: 6px;
      color: #fff;
      font-weight: 700;
      background: #126a4a;
      cursor: pointer;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      background: #fff;
    }}
    th,
    td {{
      border-bottom: 1px solid var(--line);
      padding: 6px 8px;
      vertical-align: middle;
      font-size: 13px;
    }}
    th {{
      position: sticky;
      top: 61px;
      z-index: 1;
      text-align: left;
      background: #eef4ef;
    }}
    .row-no {{
      width: 44px;
      color: var(--muted);
      text-align: right;
    }}
    .crop {{
      width: 120px;
      text-align: center;
      background: #f3f5f3;
    }}
    .crop img {{
      max-width: 104px;
      max-height: 180px;
      object-fit: contain;
      image-rendering: auto;
      border: 1px solid var(--line);
      background: #fff;
    }}
    input.label {{
      width: 145px;
      min-height: 32px;
      padding: 4px 7px;
      border: 1px solid #cbd4ce;
      border-radius: 5px;
      font: 15px Consolas, "Microsoft YaHei", monospace;
    }}
    select {{
      min-height: 30px;
    }}
    .mono {{
      font-family: Consolas, monospace;
    }}
    .confidence {{
      width: 72px;
    }}
    .status {{
      width: 64px;
      font-weight: 700;
    }}
    .status.green {{
      color: var(--green);
    }}
    .status.yellow {{
      color: var(--yellow);
    }}
    .status.red {{
      color: var(--red);
    }}
    .reason {{
      width: 190px;
      color: var(--muted);
    }}
    .review-note {{
      width: 250px;
      color: #8a4c13;
    }}
    .ocr-text {{
      width: 180px;
      color: var(--muted);
      font-family: Consolas, "Microsoft YaHei", monospace;
    }}
    .ocr-conf {{
      width: 72px;
    }}
    .source {{
      width: 230px;
    }}
    .path,
    .image-path {{
      display: none;
    }}
    tr:has(input.include:not(:checked)) {{
      opacity: 0.42;
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>书号识别训练标注表</h1>
      <div class="meta">直接修改 label 列；取消 include 可排除样本；完成后下载 TSV，再用 export_rec_dataset.py --from-draft 生成训练清单。</div>
    </div>
    <button type="button" onclick="downloadDraft()">下载修正后的 labels_draft.tsv</button>
  </header>
  <main>
    <table id="labelTable">
      <thead>
        <tr>
          <th class="row-no">#</th>
          <th>include</th>
          <th>split</th>
          <th class="crop">crop</th>
          <th>label</th>
          <th>conf</th>
          <th>status</th>
          <th>reason</th>
          <th>review note</th>
          <th>crop OCR</th>
          <th>OCR conf</th>
          <th>source image</th>
          <th class="path">source report</th>
          <th class="image-path">image</th>
        </tr>
      </thead>
      <tbody>
{chr(10).join(body_rows)}
      </tbody>
    </table>
  </main>
  <script>
    function tsvEscape(value) {{
      return String(value || "").replace(/\\t/g, " ").replace(/\\r?\\n/g, " ");
    }}

    function downloadDraft() {{
      const headers = ["include", "split", "image", "label", "confidence", "status", "reason", "review_note", "crop_ocr_text", "crop_ocr_confidence", "source_image", "source_report"];
      const lines = [headers.join("\\t")];
      document.querySelectorAll("#labelTable tbody tr").forEach((row) => {{
        const include = row.querySelector(".include").checked ? "1" : "0";
        const split = row.querySelector(".split").value;
        const image = row.querySelector(".image-path").textContent;
        const label = row.querySelector(".label").value.trim();
        const confidence = row.querySelector(".confidence").textContent;
        const status = row.querySelector(".status").textContent;
        const reason = row.querySelector(".reason").textContent;
        const reviewNote = row.querySelector(".review-note").textContent;
        const cropOcrText = row.querySelector(".ocr-text").textContent;
        const cropOcrConfidence = row.querySelector(".ocr-conf").textContent;
        const sourceImage = row.querySelector(".source").textContent;
        const sourceReport = row.querySelector(".path").textContent;
        lines.push([include, split, image, label, confidence, status, reason, reviewNote, cropOcrText, cropOcrConfidence, sourceImage, sourceReport].map(tsvEscape).join("\\t"));
      }});
      const blob = new Blob([lines.join("\\n") + "\\n"], {{ type: "text/tab-separated-values;charset=utf-8" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "labels_draft_corrected.tsv";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }}
  </script>
</body>
</html>
"""
    html_path.write_text(html_text, encoding="utf-8")


def regenerate_from_draft(draft_path: Path, output_dir: Path) -> None:
    train_rows: list[tuple[str, str]] = []
    test_rows: list[tuple[str, str]] = []
    with draft_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if (row.get("include") or "").strip() not in {"1", "true", "True", "yes", "Y", "y"}:
                continue
            image = (row.get("image") or "").strip()
            label = (row.get("label") or "").strip()
            split = (row.get("split") or "train").strip().lower()
            if not image or not label:
                continue
            if split == "test":
                test_rows.append((image, label))
            else:
                train_rows.append((image, label))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_rec_gt(output_dir / "rec_gt_train.txt", train_rows)
    write_rec_gt(output_dir / "rec_gt_test.txt", test_rows)
    generate_review_html(draft_path, output_dir / "labels_review.html")
    print(f"Corrected draft: {draft_path.resolve()}")
    print(f"Train labels: {len(train_rows)}")
    print(f"Test labels: {len(test_rows)}")
    print(f"PaddleOCR train labels: {(output_dir / 'rec_gt_train.txt').resolve()}")
    print(f"PaddleOCR test labels: {(output_dir / 'rec_gt_test.txt').resolve()}")


def main() -> None:
    args = parse_args()
    if args.from_draft is not None:
        regenerate_from_draft(args.from_draft, args.output_dir)
        return

    reports = iter_report_paths(args.results_dir, args.version_keyword)
    records: list[CropRecord] = []
    for report in reports:
        for record in read_crop_records(report):
            if should_include(record, args):
                records.append(record)

    apply_training_review_filters(records, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_records(records, args.output_dir, args.val_ratio, args.seed)

    print(f"Reports scanned: {len(reports)}")
    print(f"Crops exported: {len(records)}")
    print(f"Output: {args.output_dir.resolve()}")
    print(f"Draft labels: {(args.output_dir / 'labels_draft.tsv').resolve()}")
    print(f"PaddleOCR train labels: {(args.output_dir / 'rec_gt_train.txt').resolve()}")
    print(f"PaddleOCR test labels: {(args.output_dir / 'rec_gt_test.txt').resolve()}")
    print("Next: open labels_draft.tsv, correct wrong labels, then use rec_gt_train/test for recognition fine-tuning.")


if __name__ == "__main__":
    main()
