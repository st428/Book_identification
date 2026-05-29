from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
PADDLEOCR_REPO = (
    Path(r"D:\CPPTicketManager-main\CPPTicketManager-main\Lib\site-packages")
    / "paddlex"
    / "repo_manager"
    / "repos"
    / "PaddleOCR"
)
DEFAULT_EXPORT_DIR = ROOT / "stage5_mobile_results" / "rec_label_export"
DEFAULT_DATASET_DIR = ROOT / "stage5_mobile_results" / "rec_train_paddleocr_full_dict"
DEFAULT_OUTPUT_DIR = ROOT / "stage5_mobile_results" / "rec_training_output_full_dict_v1"
DEFAULT_BASE_CONFIG = PADDLEOCR_REPO / "configs" / "rec" / "PP-OCRv5" / "PP-OCRv5_mobile_rec.yml"
DEFAULT_PRETRAIN = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/"
    "official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams"
)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare direct PaddleOCR full-dict recognition training.")
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.00005)
    args = parser.parse_args()

    export_dir = args.export_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    base_config = args.base_config.resolve()
    if not base_config.exists():
        raise FileNotFoundError(base_config)

    train_rows = read_gt(export_dir / "rec_gt_train.txt")
    val_rows = read_gt(export_dir / "rec_gt_test.txt")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_dst = dataset_dir / "images"
    if images_dst.exists():
        shutil.rmtree(images_dst)
    shutil.copytree(export_dir / "images", images_dst)
    write_gt(dataset_dir / "train.txt", train_rows)
    write_gt(dataset_dir / "val.txt", val_rows)

    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    cfg["Global"]["use_gpu"] = False
    cfg["Global"]["epoch_num"] = args.epochs
    cfg["Global"]["save_model_dir"] = str(output_dir)
    cfg["Global"]["save_epoch_step"] = 1
    cfg["Global"]["eval_batch_step"] = [0, 2000]
    cfg["Global"]["pretrained_model"] = DEFAULT_PRETRAIN
    cfg["Global"]["checkpoints"] = ""
    cfg["Global"]["use_visualdl"] = False
    cfg["Global"]["infer_img"] = str(dataset_dir / "images")
    cfg["Global"]["save_res_path"] = str(output_dir / "predicts.txt")
    cfg["Optimizer"]["lr"]["learning_rate"] = args.learning_rate
    cfg["Optimizer"]["lr"]["warmup_epoch"] = 0

    cfg["Train"]["dataset"]["data_dir"] = str(dataset_dir)
    cfg["Train"]["dataset"]["label_file_list"] = [str(dataset_dir / "train.txt")]
    cfg["Train"]["loader"]["batch_size_per_card"] = args.batch_size
    cfg["Train"]["loader"]["num_workers"] = 4
    if "sampler" in cfg["Train"]:
        cfg["Train"]["sampler"]["first_bs"] = args.batch_size

    cfg["Eval"]["dataset"]["data_dir"] = str(dataset_dir)
    cfg["Eval"]["dataset"]["label_file_list"] = [str(dataset_dir / "val.txt")]
    cfg["Eval"]["loader"]["batch_size_per_card"] = args.batch_size
    cfg["Eval"]["loader"]["num_workers"] = 2

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = dataset_dir / "ppocrv5_mobile_rec_full_dict_train.yml"
    config_path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"dataset_dir={dataset_dir}")
    print(f"output_dir={output_dir}")
    print(f"config={config_path}")
    print(f"train={len(train_rows)} val={len(val_rows)}")


if __name__ == "__main__":
    main()
