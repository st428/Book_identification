from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PADDLEOCR_REPO = (
    Path(r"D:\CPPTicketManager-main\CPPTicketManager-main\Lib\site-packages")
    / "paddlex"
    / "repo_manager"
    / "repos"
    / "PaddleOCR"
)
DEFAULT_DATASET_DIR = ROOT / "stage5_mobile_results" / "rec_train_paddlex"
DEFAULT_TRAIN_OUTPUT = ROOT / "stage5_mobile_results" / "rec_training_output_bookcall_v1"
DEFAULT_EVAL_DIR = ROOT / "stage5_mobile_results" / "rec_eval_bookcall_v1"


def read_gt(path: Path) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image, label = line.split("\t", 1)
        rows.append((Path(image.replace("\\", "/")), label.strip()))
    return rows


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def parse_predictions(path: Path) -> dict[str, tuple[str, float]]:
    preds: dict[str, tuple[str, float]] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="mbcs")
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        image = Path(parts[0]).as_posix()
        try:
            score = float(parts[2])
        except ValueError:
            score = 0.0
        preds[image] = (parts[1].strip(), score)
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained book-call OCR checkpoint.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    train_output = args.train_output.resolve()
    config_path = (args.config.resolve() if args.config else train_output / "config.yml")
    checkpoint_path = (
        args.checkpoint.resolve()
        if args.checkpoint
        else train_output / "best_accuracy" / "best_accuracy.pdparams"
    )
    eval_dir = args.eval_dir.resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)

    gt_path = dataset_dir / ("train.txt" if args.split == "train" else "val.txt")
    rows = read_gt(gt_path)
    infer_list = eval_dir / f"{args.split}_images.txt"
    infer_list.write_text(
        "\n".join(str((dataset_dir / image).resolve()) for image, _ in rows) + "\n",
        encoding="mbcs",
    )

    result_path = eval_dir / f"{args.split}_predictions_raw.txt"
    cmd = [
        sys.executable,
        "tools/infer_rec.py",
        "-c",
        str(config_path),
        "-o",
        f"Global.checkpoints={checkpoint_path}",
        "Global.pretrained_model=",
        f"Global.infer_list={infer_list}",
        f"Global.infer_img={dataset_dir / 'images'}",
        f"Global.save_res_path={result_path}",
        "Global.use_gpu=False",
    ]
    subprocess.run(cmd, cwd=PADDLEOCR_REPO, check=True)

    preds = parse_predictions(result_path)
    out_rows = ["image\tlabel\tprediction\tscore\texact\tedit_distance\tnorm_similarity"]
    exact = 0
    norm_sum = 0.0
    missing = 0
    for image, label in rows:
        abs_image = (dataset_dir / image).resolve().as_posix()
        pred, score = preds.get(abs_image, ("", 0.0))
        if not pred:
            missing += 1
        dist = edit_distance(label, pred)
        norm = 1.0 - dist / max(len(label), len(pred), 1)
        norm_sum += norm
        ok = int(label == pred)
        exact += ok
        out_rows.append(f"{image.as_posix()}\t{label}\t{pred}\t{score:.6f}\t{ok}\t{dist}\t{norm:.6f}")

    predictions_tsv = eval_dir / f"{args.split}_predictions.tsv"
    predictions_tsv.write_text("\n".join(out_rows) + "\n", encoding="utf-8")
    total = len(rows)
    summary = {
        "split": args.split,
        "total": total,
        "exact": exact,
        "exact_accuracy": exact / total if total else 0.0,
        "mean_norm_similarity": norm_sum / total if total else 0.0,
        "missing_predictions": missing,
        "predictions_tsv": str(predictions_tsv),
    }
    summary_path = eval_dir / f"{args.split}_summary.txt"
    summary_path.write_text("\n".join(f"{k}={v}" for k, v in summary.items()) + "\n", encoding="utf-8")
    for k, v in summary.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
