from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STAGE_DIR = ROOT / "stage5_mobile_results"
TRAINING_DATA_DIR = STAGE_DIR / "training_data"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class BatchSpec:
    name: str
    export_dir: Path
    image_dir: Path
    category: str
    note: str


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_gt(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "\t" not in line:
            continue
        image, label = line.split("\t", 1)
        rows.append((image.replace("\\", "/"), label.strip()))
    return rows


def write_gt(path: Path, rows: list[tuple[str, str]]) -> None:
    text = "\n".join(f"{image}\t{label}" for image, label in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if sha1_file(src) == sha1_file(dst):
            return
        raise FileExistsError(f"Destination exists with different content: {dst}")
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def resolve_image(export_dir: Path, image_dir: Path, image_ref: str) -> Path | None:
    ref = Path(image_ref)
    candidates = [
        export_dir / image_ref,
        image_dir / ref.name,
        export_dir / "images" / ref.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def known_manual_batches(stage_dir: Path) -> list[BatchSpec]:
    batches = [
        BatchSpec(
            name="initial_20260520",
            export_dir=stage_dir / "rec_label_export",
            image_dir=stage_dir / "rec_label_export" / "images",
            category="manual_verified",
            note="Initial recognition crops with labels_draft_corrected.tsv.",
        ),
        BatchSpec(
            name="newdata_20260520",
            export_dir=stage_dir / "rec_label_export_newdata",
            image_dir=stage_dir / "rec_label_export_newdata" / "images",
            category="manual_verified",
            note="newdata recognition crops with labels_draft_corrected.tsv.",
        ),
        BatchSpec(
            name="double_data_20260524",
            export_dir=stage_dir / "rec_label_export_double_data_20260524_corrected",
            image_dir=stage_dir / "rec_label_export_double_data_20260524" / "images",
            category="manual_verified",
            note="double_data corrected from user-reviewed TSV.",
        ),
    ]
    known_export_dirs = {batch.export_dir.resolve() for batch in batches}
    for export_dir in sorted(stage_dir.glob("rec_label_export_*_corrected")):
        if export_dir.resolve() in known_export_dirs:
            continue
        if not (export_dir / "rec_gt_train.txt").exists() and not (export_dir / "rec_gt_test.txt").exists():
            continue
        image_dir = export_dir / "images"
        if not image_dir.exists():
            continue
        batches.append(
            BatchSpec(
                name=export_dir.name.removeprefix("rec_label_export_").removesuffix("_corrected"),
                export_dir=export_dir,
                image_dir=image_dir,
                category="manual_verified",
                note="User-corrected recognition crops imported from review queue.",
            )
        )
    return batches


def classify_export_dir(path: Path, manual_names: set[str]) -> tuple[str, str]:
    name = path.name
    if name in manual_names:
        return "manual_verified", "Known user-reviewed source."
    if "combined" in name and "with_qq_auto" in name:
        return "legacy_mixed", "Mixed with unreviewed QQ auto labels; do not train by default."
    if "combined" in name and "with_double_corrected" in name:
        return "legacy_mixed", "Derived from a mixed auto-label set; do not train by default."
    if name == "rec_label_export_combined_20260520":
        return "derived_manual", "Derived from older manual batches; redundant with manual sources."
    if name.endswith("_corrected") or (path / "labels_draft_corrected.tsv").exists():
        return "manual_verified", "Has corrected labels."
    if "reviewed" in name or "strict" in name or "ocr_probe" in name:
        return "obsolete_review", "Superseded review experiment."
    if "green" in name or "qq" in name:
        return "auto_candidate", "Auto-exported candidate; needs human review before training."
    if (path / "labels_draft.tsv").exists():
        return "needs_review", "Draft labels only; needs human review before training."
    return "legacy_other", "Unclassified legacy export."


def mirror_batch_metadata(batch: BatchSpec, out_dir: Path) -> dict[str, object]:
    batch_dir = out_dir / "recognition" / batch.category / "source_exports" / batch.name
    batch_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for filename in [
        "labels_draft_corrected.tsv",
        "labels_draft.tsv",
        "review_low_confidence.tsv",
        "rec_gt_train.txt",
        "rec_gt_test.txt",
        "labels_review.html",
    ]:
        src = batch.export_dir / filename
        if src.exists():
            shutil.copy2(src, batch_dir / filename)
            copied.append(filename)

    train_count = len(read_gt(batch.export_dir / "rec_gt_train.txt"))
    test_count = len(read_gt(batch.export_dir / "rec_gt_test.txt"))
    meta = {
        "name": batch.name,
        "category": batch.category,
        "export_dir": str(batch.export_dir),
        "image_dir": str(batch.image_dir),
        "train_rows": train_count,
        "test_rows": test_count,
        "total_rows": train_count + test_count,
        "copied_files": copied,
        "note": batch.note,
    }
    (batch_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta


def build_manual_train_ready(stage_dir: Path, out_dir: Path, date_tag: str) -> dict[str, object]:
    batches = [b for b in known_manual_batches(stage_dir) if b.export_dir.exists()]
    ready_dir = out_dir / "recognition" / "train_ready" / f"manual_only_{date_tag}"
    images_dir = ready_dir / "images"
    if ready_dir.exists():
        shutil.rmtree(ready_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    train_rows: list[tuple[str, str]] = []
    test_rows: list[tuple[str, str]] = []
    master_rows: list[dict[str, object]] = []
    duplicate_rows: list[dict[str, object]] = []
    conflict_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    seen_hash_label: dict[tuple[str, str], str] = {}
    seen_hash: dict[str, tuple[str, str]] = {}

    def add_rows(batch: BatchSpec, split: str, rows: list[tuple[str, str]]) -> None:
        target_rows = train_rows if split == "train" else test_rows
        for image_ref, label in rows:
            src = resolve_image(batch.export_dir, batch.image_dir, image_ref)
            if src is None:
                missing_rows.append(
                    {
                        "batch": batch.name,
                        "split": split,
                        "image": image_ref,
                        "label": label,
                    }
                )
                continue
            digest = sha1_file(src)
            key = (digest, label)
            original = seen_hash_label.get(key)
            if original:
                duplicate_rows.append(
                    {
                        "batch": batch.name,
                        "split": split,
                        "image": image_ref,
                        "label": label,
                        "kept_image": original,
                        "reason": "same_image_same_label",
                    }
                )
                continue
            if digest in seen_hash and seen_hash[digest][1] != label:
                conflict_rows.append(
                    {
                        "batch": batch.name,
                        "split": split,
                        "image": image_ref,
                        "label": label,
                        "existing_image": seen_hash[digest][0],
                        "existing_label": seen_hash[digest][1],
                        "reason": "same_image_different_label",
                    }
                )

            dest_name = f"{batch.name}__{Path(image_ref).name}"
            dest_rel = f"images/{dest_name}"
            dest = ready_dir / dest_rel
            if dest.exists():
                dest_name = f"{batch.name}__{digest[:10]}__{Path(image_ref).name}"
                dest_rel = f"images/{dest_name}"
                dest = ready_dir / dest_rel
            hardlink_or_copy(src, dest)
            seen_hash_label[key] = dest_rel
            seen_hash.setdefault(digest, (dest_rel, label))
            target_rows.append((dest_rel, label))
            master_rows.append(
                {
                    "split": split,
                    "image": dest_rel,
                    "label": label,
                    "batch": batch.name,
                    "source_image": str(src),
                    "sha1": digest,
                }
            )

    for batch in batches:
        add_rows(batch, "train", read_gt(batch.export_dir / "rec_gt_train.txt"))
        add_rows(batch, "test", read_gt(batch.export_dir / "rec_gt_test.txt"))

    write_gt(ready_dir / "rec_gt_train.txt", train_rows)
    write_gt(ready_dir / "rec_gt_test.txt", test_rows)
    write_tsv(
        ready_dir / "labels_master.tsv",
        ["split", "image", "label", "batch", "source_image", "sha1"],
        master_rows,
    )
    write_tsv(
        ready_dir / "duplicates_skipped.tsv",
        ["batch", "split", "image", "label", "kept_image", "reason"],
        duplicate_rows,
    )
    write_tsv(
        ready_dir / "label_conflicts.tsv",
        ["batch", "split", "image", "label", "existing_image", "existing_label", "reason"],
        conflict_rows,
    )
    write_tsv(
        ready_dir / "missing_images.tsv",
        ["batch", "split", "image", "label"],
        missing_rows,
    )

    manifest = {
        "dataset": "manual_only",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ready_dir": str(ready_dir),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "total_rows": len(train_rows) + len(test_rows),
        "duplicates_skipped": len(duplicate_rows),
        "label_conflicts": len(conflict_rows),
        "missing_images": len(missing_rows),
        "batches": [b.name for b in batches],
        "policy": "Only human-reviewed or corrected recognition crops are included.",
    }
    (ready_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def catalog_existing_exports(stage_dir: Path, out_dir: Path) -> list[dict[str, object]]:
    manual_names = {b.export_dir.name for b in known_manual_batches(stage_dir)}
    rows: list[dict[str, object]] = []
    for export_dir in sorted(stage_dir.glob("rec_label_export*")):
        if not export_dir.is_dir():
            continue
        category, note = classify_export_dir(export_dir, manual_names)
        train_count = len(read_gt(export_dir / "rec_gt_train.txt"))
        test_count = len(read_gt(export_dir / "rec_gt_test.txt"))
        rows.append(
            {
                "name": export_dir.name,
                "category": category,
                "train_rows": train_count,
                "test_rows": test_count,
                "total_rows": train_count + test_count,
                "path": str(export_dir),
                "note": note,
            }
        )
    write_tsv(
        out_dir / "recognition" / "export_catalog.tsv",
        ["name", "category", "train_rows", "test_rows", "total_rows", "path", "note"],
        rows,
    )
    return rows


def dedupe_image_files(stage_dir: Path, out_dir: Path) -> dict[str, object]:
    roots = [
        *stage_dir.glob("rec_label_export*/images"),
        *stage_dir.glob("rec_train_paddle*/images"),
        *stage_dir.glob("rec_train_paddlex*/images"),
        *(out_dir / "recognition").glob("**/images"),
    ]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(
                f for f in root.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES
            )

    by_size: dict[int, list[Path]] = {}
    for path in files:
        try:
            by_size.setdefault(path.stat().st_size, []).append(path)
        except OSError:
            continue

    by_hash: dict[str, list[Path]] = {}
    for same_size in by_size.values():
        if len(same_size) < 2:
            continue
        for path in same_size:
            try:
                by_hash.setdefault(sha1_file(path), []).append(path)
            except OSError:
                continue

    report_rows: list[dict[str, object]] = []
    replaced = 0
    saved_bytes = 0
    for digest, group in sorted(by_hash.items()):
        unique_paths = sorted({p.resolve() for p in group}, key=lambda p: str(p).lower())
        if len(unique_paths) < 2:
            continue
        canonical = unique_paths[0]
        for duplicate in unique_paths[1:]:
            try:
                if os.path.samefile(canonical, duplicate):
                    continue
            except OSError:
                pass
            size = duplicate.stat().st_size
            tmp = duplicate.with_suffix(duplicate.suffix + ".dedupe_tmp")
            if tmp.exists():
                tmp.unlink()
            duplicate.rename(tmp)
            try:
                os.link(canonical, duplicate)
                tmp.unlink()
                replaced += 1
                saved_bytes += size
                report_rows.append(
                    {
                        "sha1": digest,
                        "canonical": str(canonical),
                        "duplicate": str(duplicate),
                        "bytes": size,
                        "action": "replaced_with_hardlink",
                    }
                )
            except OSError as exc:
                tmp.rename(duplicate)
                report_rows.append(
                    {
                        "sha1": digest,
                        "canonical": str(canonical),
                        "duplicate": str(duplicate),
                        "bytes": size,
                        "action": f"kept_copy: {exc}",
                    }
                )

    write_tsv(
        out_dir / "dedupe_report.tsv",
        ["sha1", "canonical", "duplicate", "bytes", "action"],
        report_rows,
    )
    return {
        "scanned_images": len(files),
        "duplicate_files_relinked": replaced,
        "logical_bytes_saved": saved_bytes,
        "report": str(out_dir / "dedupe_report.tsv"),
    }


def remove_obsolete_dirs(stage_dir: Path, out_dir: Path) -> dict[str, object]:
    obsolete_names = [
        "rec_label_export_combined_20260523_with_qq_auto",
        "rec_label_export_combined_20260524_with_double_corrected",
        "rec_label_export_double_data_20260524_reviewed",
        "rec_label_export_double_data_20260524_reviewed_ocr_probe",
        "rec_label_export_double_data_20260524_reviewed_ocr_probe2",
        "rec_label_export_double_data_20260524_reviewed_strict",
        "rec_train_paddlex_combined_20260523_with_qq_auto",
        "rec_train_paddlex_combined_20260524_with_double_corrected",
        "rec_train_paddlex_combined_20260524_with_double_corrected_quick1",
        "rec_training_output_combined_20260523_with_qq_auto",
        "rec_training_output_combined_20260524_with_double_corrected_quick1",
        "rec_training_output_newdata",
    ]
    rows: list[dict[str, object]] = []
    removed = 0
    freed = 0
    for name in obsolete_names:
        path = stage_dir / name
        if not path.exists():
            continue
        size = 0
        file_count = 0
        for f in path.rglob("*"):
            if f.is_file():
                file_count += 1
                try:
                    size += f.stat().st_size
                except OSError:
                    pass
        shutil.rmtree(path)
        removed += 1
        freed += size
        rows.append(
            {
                "name": name,
                "path": str(path),
                "files": file_count,
                "bytes": size,
                "reason": "obsolete duplicate or mixed auto-label derivative",
            }
        )
    write_tsv(
        out_dir / "removed_obsolete_dirs.tsv",
        ["name", "path", "files", "bytes", "reason"],
        rows,
    )
    return {"removed_dirs": removed, "freed_bytes": freed, "report": str(out_dir / "removed_obsolete_dirs.tsv")}


def write_readme(out_dir: Path, manifest: dict[str, object]) -> None:
    readme = f"""# Training Data Layout

This directory is the canonical data entry point for OCR recognition training.

## Categories

- `recognition/manual_verified/source_exports/`: human-reviewed or corrected label exports.
- `recognition/train_ready/manual_only_*`: training-ready OCR recognition dataset.
- `recognition/export_catalog.tsv`: automatic classification of all legacy export directories.
- `dedupe_report.tsv`: duplicate image files replaced with hardlinks.

## Active Dataset

Use this dataset for the next recognition training run:

`{manifest["ready_dir"]}`

Rows:

- train: {manifest["train_rows"]}
- test: {manifest["test_rows"]}
- duplicate crop rows skipped: {manifest["duplicates_skipped"]}
- label conflicts: {manifest["label_conflicts"]}
- missing images: {manifest["missing_images"]}

## Policy

Newly exported crops are not used for training by default unless they have a corrected TSV
or are explicitly placed in a `*_corrected` export. Draft/auto/green-only exports are treated
as review candidates.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize project training data and dedupe generated crops.")
    parser.add_argument("--stage-dir", type=Path, default=STAGE_DIR)
    parser.add_argument("--out-dir", type=Path, default=TRAINING_DATA_DIR)
    parser.add_argument("--date-tag", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--skip-dedupe", action="store_true")
    parser.add_argument("--remove-obsolete", action="store_true")
    args = parser.parse_args()

    stage_dir = args.stage_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_meta = [
        mirror_batch_metadata(batch, out_dir)
        for batch in known_manual_batches(stage_dir)
        if batch.export_dir.exists()
    ]
    export_catalog = catalog_existing_exports(stage_dir, out_dir)
    manual_manifest = build_manual_train_ready(stage_dir, out_dir, args.date_tag)
    dedupe_summary = {"skipped": True}
    if not args.skip_dedupe:
        dedupe_summary = dedupe_image_files(stage_dir, out_dir)
    cleanup_summary = {"skipped": True}
    if args.remove_obsolete:
        cleanup_summary = remove_obsolete_dirs(stage_dir, out_dir)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage_dir": str(stage_dir),
        "out_dir": str(out_dir),
        "manual_batches": batch_meta,
        "export_catalog_rows": len(export_catalog),
        "manual_train_ready": manual_manifest,
        "dedupe": dedupe_summary,
        "cleanup": cleanup_summary,
    }
    (out_dir / "organize_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_readme(out_dir, manual_manifest)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
