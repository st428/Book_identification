from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_EXPORT_DIR = ROOT / "stage5_mobile_results" / "rec_label_export"
DEFAULT_DATASET_DIR = ROOT / "stage5_mobile_results" / "rec_train_paddlex"
DEFAULT_OUTPUT_DIR = ROOT / "stage5_mobile_results" / "rec_training_output"
DEFAULT_PADDLEOCR_CONFIG = (
    Path(r"D:\CPPTicketManager-main\CPPTicketManager-main\Lib\site-packages")
    / "paddlex"
    / "repo_manager"
    / "repos"
    / "PaddleOCR"
    / "configs"
    / "rec"
    / "PP-OCRv5"
    / "PP-OCRv5_mobile_rec.yml"
)
DEFAULT_LOCAL_BASIC_CONFIG = DEFAULT_DATASET_DIR / "ppocrv5_mobile_rec_bookcall_base.yml"


def read_gt(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image, label = line.split("\t", 1)
        rows.append((image.replace("\\", "/"), label.strip()))
    return rows


def write_gt(path: Path, rows: list[tuple[str, str]]) -> None:
    text = "\n".join(f"{image}\t{label}" for image, label in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def link_or_copy(src: str, dst: str) -> str:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def yaml_path(path: Path) -> str:
    return path.resolve().as_posix()


def build_config(
    dataset_dir: Path,
    output_dir: Path,
    basic_config_path: Path,
    epochs: int,
    batch_size: int,
) -> str:
    return f"""Global:
  model: PP-OCRv5_mobile_rec
  mode: check_dataset
  dataset_dir: "{yaml_path(dataset_dir)}"
  device: cpu
  output: "{yaml_path(output_dir)}"

CheckDataset:
  convert:
    enable: False
    src_dataset_type: null
  split:
    enable: False
    train_percent: null
    val_percent: null

Train:
  basic_config_path: "{yaml_path(basic_config_path)}"
  epochs_iters: {epochs}
  batch_size: {batch_size}
  learning_rate: 0.0005
  pretrain_weight_path: https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams
  resume_path: null
  log_interval: 10
  eval_interval: 1
  save_interval: 1

Evaluate:
  basic_config_path: "{yaml_path(basic_config_path)}"
  weight_path: "{yaml_path(output_dir / "best_accuracy" / "best_accuracy.pdparams")}"
  log_interval: 1

Export:
  basic_config_path: "{yaml_path(basic_config_path)}"
  weight_path: "{yaml_path(output_dir / "best_accuracy" / "best_accuracy.pdparams")}"
"""


def make_local_basic_config(
    src_path: Path,
    dst_path: Path,
    dict_path: Path,
    warmup_epochs: int,
) -> Path:
    data = yaml.safe_load(src_path.read_text(encoding="utf-8"))
    data.setdefault("Optimizer", {}).setdefault("lr", {})["warmup_epoch"] = warmup_epochs
    global_cfg = data.setdefault("Global", {})
    global_cfg["character_dict_path"] = yaml_path(dict_path)
    global_cfg["use_space_char"] = True
    global_cfg["use_visualdl"] = False
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return dst_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PaddleX OCR recognition training data.")
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basic-config-path", type=Path, default=DEFAULT_PADDLEOCR_CONFIG)
    parser.add_argument("--local-basic-config", type=Path, default=DEFAULT_LOCAL_BASIC_CONFIG)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--link-images",
        action="store_true",
        help="Use hardlinks for images when possible instead of copying duplicate crop files.",
    )
    args = parser.parse_args()

    export_dir = args.export_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    basic_config_path = args.basic_config_path.resolve()
    if not basic_config_path.exists():
        raise FileNotFoundError(f"PaddleOCR config not found: {basic_config_path}")

    train_rows = read_gt(export_dir / "rec_gt_train.txt")
    val_rows = read_gt(export_dir / "rec_gt_test.txt")
    chars = sorted({char for _, label in train_rows + val_rows for char in label})

    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_src = export_dir / "images"
    images_dst = dataset_dir / "images"
    if images_dst.exists():
        shutil.rmtree(images_dst)
    if args.link_images:
        shutil.copytree(images_src, images_dst, copy_function=link_or_copy)
    else:
        shutil.copytree(images_src, images_dst)

    write_gt(dataset_dir / "train.txt", train_rows)
    write_gt(dataset_dir / "val.txt", val_rows)
    dict_path = dataset_dir / "dict.txt"
    dict_path.write_text("\n".join(chars) + "\n", encoding="utf-8")
    local_basic_config = make_local_basic_config(
        basic_config_path,
        args.local_basic_config.resolve(),
        dict_path,
        args.warmup_epochs,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = dataset_dir / "ppocrv5_mobile_rec_bookcall_train.yaml"
    config_path.write_text(
        build_config(dataset_dir, output_dir, local_basic_config, args.epochs, args.batch_size),
        encoding="utf-8",
    )

    print(f"dataset_dir={dataset_dir}")
    print(f"output_dir={output_dir}")
    print(f"config={config_path}")
    print(f"train={len(train_rows)} val={len(val_rows)} chars={len(chars)}")
    print("chars=" + "".join(chars))


if __name__ == "__main__":
    main()
