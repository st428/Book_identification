from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from datetime import datetime
from pathlib import Path

from organize_project_data import (
    STAGE_DIR,
    TRAINING_DATA_DIR,
    classify_export_dir,
    hardlink_or_copy,
    known_manual_batches,
    sha1_file,
)


STANDARD_COLUMNS = [
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
EXTRA_COLUMNS = ["batch", "source_export"]
STATUS_ORDER = {"yellow": 0, "red": 1, "": 2, "green": 3}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def corrected_sibling_exists(export_dir: Path) -> bool:
    name = export_dir.name
    if name.endswith("_green"):
        name = name[: -len("_green")]
    return (export_dir.parent / f"{name}_corrected").exists()


def full_export_exists_for_green(export_dir: Path) -> bool:
    if not export_dir.name.endswith("_green"):
        return False
    return (export_dir.parent / export_dir.name[: -len("_green")]).exists()


def select_review_sources(stage_dir: Path) -> tuple[list[Path], list[dict[str, str]]]:
    manual_names = {b.export_dir.name for b in known_manual_batches(stage_dir)}
    sources: list[Path] = []
    skipped: list[dict[str, str]] = []
    for export_dir in sorted(stage_dir.glob("rec_label_export*")):
        if not export_dir.is_dir():
            continue
        category, note = classify_export_dir(export_dir, manual_names)
        if category not in {"needs_review", "auto_candidate"}:
            continue
        if corrected_sibling_exists(export_dir):
            skipped.append(
                {
                    "name": export_dir.name,
                    "category": category,
                    "reason": "has_corrected_export",
                    "path": str(export_dir),
                }
            )
            continue
        if full_export_exists_for_green(export_dir):
            skipped.append(
                {
                    "name": export_dir.name,
                    "category": category,
                    "reason": "covered_by_full_export",
                    "path": str(export_dir),
                }
            )
            continue
        if not (export_dir / "labels_draft.tsv").exists():
            skipped.append(
                {
                    "name": export_dir.name,
                    "category": category,
                    "reason": "missing_labels_draft",
                    "path": str(export_dir),
                }
            )
            continue
        sources.append(export_dir)
    return sources, skipped


def source_image_path(export_dir: Path, image_ref: str) -> Path | None:
    candidates = [
        export_dir / image_ref,
        export_dir / "images" / Path(image_ref).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def normalize_reason(status: str, confidence_text: str) -> str:
    try:
        confidence = float(confidence_text or 0)
    except ValueError:
        confidence = 0.0
    if status == "yellow":
        if confidence <= 0.001:
            return "补框或邻近推测结果，需人工重点核对"
        if confidence < 0.65:
            return "OCR 置信度较低，需人工重点核对"
        return "黄色疑似结果，需人工核对"
    if status == "red":
        return "红色错架/异常候选，需人工核对"
    if status == "green":
        return "自动识别候选，训练前需人工确认"
    return "待人工核对"


def build_rows(sources: list[Path], output_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    image_dir = output_dir / "images"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    duplicates: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for source in sources:
        draft = source / "labels_draft.tsv"
        for index, row in enumerate(read_tsv(draft), start=1):
            image_ref = (row.get("image") or "").strip()
            label = (row.get("label") or "").strip()
            src = source_image_path(source, image_ref)
            if not src:
                missing.append(
                    {
                        "source_export": source.name,
                        "row": str(index),
                        "image": image_ref,
                        "label": label,
                    }
                )
                continue
            digest = sha1_file(src)
            dedupe_key = (digest, label)
            if dedupe_key in seen:
                duplicates.append(
                    {
                        "source_export": source.name,
                        "row": str(index),
                        "image": image_ref,
                        "label": label,
                        "sha1": digest,
                    }
                )
                continue
            seen.add(dedupe_key)

            dest_name = f"{source.name}__{Path(image_ref).name}"
            dest = image_dir / dest_name
            if dest.exists():
                dest = image_dir / f"{source.name}__{digest[:10]}__{Path(image_ref).name}"
            hardlink_or_copy(src, dest)

            out = {key: (row.get(key) or "").strip() for key in STANDARD_COLUMNS}
            out["image"] = dest.relative_to(output_dir).as_posix()
            out.setdefault("include", "1")
            out.setdefault("split", "train")
            out["reason"] = normalize_reason(out.get("status", ""), out.get("confidence", ""))
            out["review_note"] = out.get("review_note") or "needs_human_review"
            out["batch"] = source.name.replace("rec_label_export_", "")
            out["source_export"] = str(source)
            rows.append(out)

    rows.sort(
        key=lambda r: (
            STATUS_ORDER.get(r.get("status", ""), 9),
            float(r.get("confidence") or 0),
            r.get("source_image", ""),
            r.get("image", ""),
        )
    )
    return rows, duplicates, missing


def generate_html(rows: list[dict[str, str]], html_path: Path, title: str) -> None:
    body: list[str] = []
    for idx, row in enumerate(rows, start=1):
        include = (row.get("include") or "1").strip()
        checked = " checked" if include in {"1", "true", "True", "yes", "Y", "y"} else ""
        split = row.get("split") or "train"
        body.append(
            "        <tr>"
            f"<td class=\"row-no\">{idx}</td>"
            f"<td><input class=\"include\" type=\"checkbox\"{checked}></td>"
            f"<td><select class=\"split\"><option value=\"train\"{' selected' if split != 'test' else ''}>train</option>"
            f"<option value=\"test\"{' selected' if split == 'test' else ''}>test</option></select></td>"
            f"<td class=\"crop\"><img src=\"{html.escape(row.get('image', ''), quote=True)}\" alt=\"crop {idx}\"></td>"
            f"<td><input class=\"label\" value=\"{html.escape(row.get('label', ''), quote=True)}\" spellcheck=\"false\"></td>"
            f"<td class=\"mono confidence\">{html.escape(row.get('confidence', ''))}</td>"
            f"<td class=\"status {html.escape(row.get('status', ''))}\">{html.escape(row.get('status', ''))}</td>"
            f"<td class=\"reason\">{html.escape(row.get('reason', ''))}</td>"
            f"<td class=\"source\">{html.escape(row.get('source_image', ''))}</td>"
            f"<td class=\"batch\">{html.escape(row.get('batch', ''))}</td>"
            f"<td class=\"note\">{html.escape(row.get('review_note', ''))}</td>"
            f"<td class=\"hidden crop-ocr-text\">{html.escape(row.get('crop_ocr_text', ''))}</td>"
            f"<td class=\"hidden crop-ocr-confidence\">{html.escape(row.get('crop_ocr_confidence', ''))}</td>"
            f"<td class=\"hidden source-report\">{html.escape(row.get('source_report', ''))}</td>"
            f"<td class=\"hidden image-path\">{html.escape(row.get('image', ''))}</td>"
            f"<td class=\"hidden source-export\">{html.escape(row.get('source_export', ''))}</td>"
            "</tr>"
        )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --ink: #17201b;
      --muted: #68736c;
      --line: #d8ded9;
      --paper: #f7faf8;
      --green: #147a3d;
      --yellow: #a26200;
      --red: #b3352e;
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
      z-index: 3;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(247, 250, 248, 0.97);
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 4px;
    }}
    button {{
      min-height: 36px;
      padding: 0 14px;
      border: 0;
      border-radius: 6px;
      color: white;
      font-weight: 700;
      background: #126a4a;
      cursor: pointer;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      background: white;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 6px 8px;
      vertical-align: middle;
      font-size: 13px;
    }}
    th {{
      position: sticky;
      top: 65px;
      z-index: 2;
      text-align: left;
      background: #eef4ef;
    }}
    .row-no {{
      width: 42px;
      color: var(--muted);
      text-align: right;
    }}
    .crop {{
      width: 128px;
      text-align: center;
      background: #f2f5f3;
    }}
    .crop img {{
      max-width: 112px;
      max-height: 190px;
      object-fit: contain;
      border: 1px solid var(--line);
      background: white;
    }}
    input.label {{
      width: 145px;
      min-height: 32px;
      padding: 4px 7px;
      border: 1px solid #c9d3cd;
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
    .status.green {{ color: var(--green); }}
    .status.yellow {{ color: var(--yellow); }}
    .status.red {{ color: var(--red); }}
    .reason {{
      width: 170px;
      color: var(--muted);
    }}
    .source {{
      width: 230px;
    }}
    .batch {{
      width: 150px;
      color: var(--muted);
    }}
    .note {{
      width: 160px;
      color: #8a4c13;
    }}
    .hidden {{
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
      <h1>{html.escape(title)}</h1>
      <div class="meta">核对 crop 和 label；错误的直接改 label；不适合作训练的取消 include；完成后点右侧按钮下载 TSV。</div>
    </div>
    <button type="button" onclick="downloadDraft()">下载修正后的 TSV</button>
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
          <th>source image</th>
          <th>batch</th>
          <th>review note</th>
          <th class="hidden">crop OCR</th>
          <th class="hidden">OCR conf</th>
          <th class="hidden">source report</th>
          <th class="hidden">image</th>
          <th class="hidden">source export</th>
        </tr>
      </thead>
      <tbody>
{chr(10).join(body)}
      </tbody>
    </table>
  </main>
  <script>
    function tsvEscape(value) {{
      return String(value || "").replace(/\\t/g, " ").replace(/\\r?\\n/g, " ");
    }}

    function downloadDraft() {{
      const headers = {json.dumps(STANDARD_COLUMNS + EXTRA_COLUMNS, ensure_ascii=False)};
      const lines = [headers.join("\\t")];
      document.querySelectorAll("#labelTable tbody tr").forEach((row) => {{
        const values = [
          row.querySelector(".include").checked ? "1" : "0",
          row.querySelector(".split").value,
          row.querySelector(".image-path").textContent,
          row.querySelector(".label").value.trim(),
          row.querySelector(".confidence").textContent,
          row.querySelector(".status").textContent,
          row.querySelector(".reason").textContent,
          row.querySelector(".note").textContent,
          row.querySelector(".crop-ocr-text").textContent,
          row.querySelector(".crop-ocr-confidence").textContent,
          row.querySelector(".source").textContent,
          row.querySelector(".source-report").textContent,
          row.querySelector(".batch").textContent,
          row.querySelector(".source-export").textContent,
        ];
        lines.push(values.map(tsvEscape).join("\\t"));
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deduped human-review queue from unverified OCR crops.")
    parser.add_argument("--stage-dir", type=Path, default=STAGE_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TRAINING_DATA_DIR / "recognition" / "review_needed" / f"review_needed_{datetime.now().strftime('%Y%m%d')}",
    )
    args = parser.parse_args()

    stage_dir = args.stage_dir.resolve()
    output_dir = args.output_dir.resolve()
    sources, skipped_sources = select_review_sources(stage_dir)
    rows, duplicates, missing = build_rows(sources, output_dir)
    write_tsv(output_dir / "labels_draft.tsv", STANDARD_COLUMNS + EXTRA_COLUMNS, rows)
    write_tsv(
        output_dir / "duplicates_skipped.tsv",
        ["source_export", "row", "image", "label", "sha1"],
        duplicates,
    )
    write_tsv(
        output_dir / "missing_images.tsv",
        ["source_export", "row", "image", "label"],
        missing,
    )
    write_tsv(
        output_dir / "skipped_sources.tsv",
        ["name", "category", "reason", "path"],
        skipped_sources,
    )

    summary_rows: list[dict[str, str]] = []
    for source in sources:
        source_rows = [row for row in rows if row.get("source_export") == str(source)]
        summary_rows.append(
            {
                "source_export": source.name,
                "rows": str(len(source_rows)),
                "green": str(sum(1 for row in source_rows if row.get("status") == "green")),
                "yellow": str(sum(1 for row in source_rows if row.get("status") == "yellow")),
                "red": str(sum(1 for row in source_rows if row.get("status") == "red")),
                "path": str(source),
            }
        )
    write_tsv(output_dir / "source_summary.tsv", ["source_export", "rows", "green", "yellow", "red", "path"], summary_rows)
    generate_html(rows, output_dir / "labels_review.html", "待人工校验的索书号 crop")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "sources": [str(s) for s in sources],
        "rows": len(rows),
        "duplicates_skipped": len(duplicates),
        "missing_images": len(missing),
        "skipped_sources": skipped_sources,
        "html": str(output_dir / "labels_review.html"),
        "draft": str(output_dir / "labels_draft.tsv"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
