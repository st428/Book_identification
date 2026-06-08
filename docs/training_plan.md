# 后续训练方案

## 当前结论

最近一轮 `manual_only_20260530` 已经完成，但不能部署到 Web 端。整图对比中，自训练 `best_accuracy` 和 `iter_epoch_4` 比当前官方 Web OCR 更慢，且几乎不能产出可用绿色书号。

关键问题不是继续堆轮数，而是训练、导出和推理链路没有按“书号识别”这个窄任务收敛：

- 使用 PP-OCRv5 全量中文大字典时，输出类别太多，训练 4 轮后 `acc=0`，书号结构没有学稳。
- 旧导出模型虽然能被 Web 端加载，但推理表现明显劣于官方模型，不能直接接入。
- 后续必须先用 checkpoint 级评估和整图 Web 流水线评估双重通过，再考虑替换 Web 默认模型。

## 已调整的默认训练配置

推荐入口仍然是：

```powershell
python prepare_rec_training.py --export-dir <人工校验导出目录> --dataset-dir <训练数据目录> --output-dir <训练输出目录> --link-images
python run_rec_training.py -c <训练数据目录>\ppocrv5_mobile_rec_bookcall_train.yaml -o Global.mode=train
python evaluate_rec_checkpoint.py --dataset-dir <训练数据目录> --train-output <训练输出目录> --eval-dir <评估输出目录> --split val --min-exact-accuracy 0.8 --min-mean-norm-similarity 0.95
```

`prepare_rec_training.py` 的默认值现在改为：

- `dict_mode=call-number`：默认使用书号紧凑字典。
- 字典基础字符：`0-9`、`A-Z`、`.`、`/`、`:`、`-`、`+`，并自动补充人工标签里出现的新字符。
- 标签统一转大写，避免 `h/H`、`x/X` 混进两个类别。
- `main_indicator=acc`，让 `best_accuracy` 真正按完整书号精确命中选择。
- `learning_rate=0.00005`，`epochs=20`，`warmup_epochs=1`。
- `image_width=480`，`max_text_length=32`。
- 默认关闭 `RecConAug`，避免把多个短书号拼成不真实样本。
- 每个 `dataset-dir` 会自带一份 `ppocrv5_mobile_rec_bookcall_base.yml`，避免不同训练批次互相覆盖字典配置。
- 训练、评估、导出和推理都保留 `D:\upload_ascii` 这类 ASCII junction 路径，不再用 `Path.resolve()` 解析回中文路径。

`prepare_rec_full_dict_training.py` 保留为备用实验入口，但不再作为默认推荐方案。

## 修改前后差异

| 项目 | 修改前 | 修改后 |
|---|---|---|
| 默认字典 | PP-OCRv5 全量中文大字典 | 书号紧凑字典 |
| 输出类别 | 很大，容易学成乱码模板 | 小而稳定，贴合索书号 |
| 标签大小写 | 依赖人工输入 | 自动统一大写 |
| best 模型指标 | 可能按相似度选到不可用模型 | 默认按 exact `acc` 选 |
| 推理接入 | 官方/自训练结果可能共用缓存 | 缓存 key 带 OCR 模型标识 |
| 基础配置路径 | 多批训练可能共用同一份 base config | 每个训练数据目录独立保存 |
| 路径处理 | ASCII junction 会被解析回中文路径 | 保留传入的 ASCII 绝对路径 |
| Web 默认模型 | 易误接入坏模型 | 默认仍用官方模型；自训练模型需显式指定 |

## 部署闸门

自训练模型接入 Web 前必须同时通过：

- `evaluate_rec_checkpoint.py` 验证集 exact accuracy 至少 80%。
- 平均归一化相似度至少 0.95。
- 用 `compare_rec_model_efficiency.py` 在未参与训练的整架照片上对比，绿色结果、耗时、调架建议不能明显劣于 `web_current`。

只有通过这些检查，才设置：

```powershell
$env:BOOK_OCR_REC_MODEL_DIR = "D:\upload_ascii\stage5_mobile_results\training_data\recognition\runs\<run>\best_accuracy\inference"
python mobile_server.py
```

如果不设置 `BOOK_OCR_REC_MODEL_DIR`，Web 端继续使用当前官方 PP-OCRv5 模型。

## 数据补充重点

下一批优先补：

- 薄书、窄书脊、相邻书号很近的 crop。
- `XXW/XY3`、`LJJ4/LJJ5` 这类容易串号的相邻后缀。
- 带 `2026`、冒号、数字后缀、短横线后缀的完整书号。
- 非 `D669.3` 类号样本，避免模型固化成单一模板。
- 倾斜、右侧漏检、双排红标签场景。

明显残缺、串入相邻书号、纯白/纯黑、图文不对应的 crop 不进入训练。
