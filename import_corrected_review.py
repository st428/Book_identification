from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

from organize_project_data import ROOT, STAGE_DIR, hardlink_or_copy


TRUE_VALUES = {"1", "true", "True", "yes", "Y", "y"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_gt(path: Path, rows: list[tuple[str, str]]) -> None:
    text = "\n".join(f"{image}\t{label}" for image, label in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def infer_batch_name(rows: list[dict[str, str]], fallback: str) -> str:
    names = sorted({(row.get("batch") or "").strip() for row in rows if (row.get("batch") or "").strip()})
    if len(names) == 1:
        return names[0]
    if names:
        return "mixed_" + "_".join(name.replace("/", "_") for name in names[:3])
    return fallback


def find_review_image(review_dir: Path, image_ref: str) -> Path | None:
    rel = Path(image_ref)
    candidates = [
        review_dir / image_ref,
        review_dir / "images" / rel.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a user-corrected review TSV as a verified OCR dataset batch.")
    parser.add_argument("corrected_tsv", type=Path)
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=STAGE_DIR / "training_data" / "recognition" / "review_needed" / "review_needed_20260527",
        help="Directory containing the review images referenced by the TSV.",
    )
    parser.add_argument("--batch-name", default="")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    corrected_tsv = args.corrected_tsv.resolve()
    review_dir = args.review_dir.resolve()
    rows = read_rows(corrected_tsv)
    batch_name = args.batch_name.strip() or infer_batch_name(rows, corrected_tsv.stem)
    output_dir = (args.output_dir or (STAGE_DIR / f"rec_label_export_{batch_name}_corrected")).resolve()
    image_dir = output_dir / "images"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    included_rows: list[dict[str, str]] = []
    excluded_rows: list[dict[str, str]] = []
    missing_rows: list[dict[str, str]] = []
    train_rows: list[tuple[str, str]] = []
    test_rows: list[tuple[str, str]] = []
    bad_label_rows: list[dict[str, str]] = []
    allowed = set("-./0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    out_rows: list[dict[str, str]] = []
    fieldnames = list(rows[0].keys()) if rows else []
    if "verified_image" not in fieldnames:
        fieldnames.append("verified_image")

    for idx, row in enumerate(rows, start=2):
        image_ref = (row.get("image") or "").strip()
        src = find_review_image(review_dir, image_ref)
        out = dict(row)
        if src is None:
            missing_rows.append({"row": str(idx), "image": image_ref, "label": (row.get("label") or "").strip()})
            out["verified_image"] = ""
            out_rows.append(out)
            continue

        dest_name = Path(image_ref).name
        dest = image_dir / dest_name
        if dest.exists():
            dest = image_dir / f"{idx:06d}_{dest_name}"
        hardlink_or_copy(src, dest)
        rel_image = dest.relative_to(output_dir).as_posix()
        out["image"] = rel_image
        out["verified_image"] = rel_image
        out_rows.append(out)

        included = (row.get("include") or "").strip() in TRUE_VALUES
        label = (row.get("label") or "").strip()
        if not included:
            excluded_rows.append(out)
            continue
        included_rows.append(out)
        invalid = sorted(set(label) - allowed)
        if not label or invalid:
            bad_label_rows.append(
                {
                    "row": str(idx),
                    "image": rel_image,
                    "label": label,
                    "invalid_chars": "".join(invalid),
                }
            )
            continue
        split = (row.get("split") or "train").strip().lower()
        if split == "test":
            test_rows.append((rel_image, label))
        else:
            train_rows.append((rel_image, label))

    write_tsv(output_dir / "labels_draft_corrected.tsv", fieldnames, out_rows)
    write_tsv(output_dir / "excluded_rows.tsv", fieldnames, excluded_rows)
    write_tsv(output_dir / "missing_images.tsv", ["row", "image", "label"], missing_rows)
    write_tsv(output_dir / "bad_label_rows.tsv", ["row", "image", "label", "invalid_chars"], bad_label_rows)
    write_gt(output_dir / "rec_gt_train.txt", train_rows)
    write_gt(output_dir / "rec_gt_test.txt", test_rows)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_tsv": str(corrected_tsv),
        "review_dir": str(review_dir),
        "output_dir": str(output_dir),
        "batch_name": batch_name,
        "rows": len(rows),
        "included": len(included_rows),
        "excluded": len(excluded_rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "missing_images": len(missing_rows),
        "bad_label_rows": len(bad_label_rows),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
