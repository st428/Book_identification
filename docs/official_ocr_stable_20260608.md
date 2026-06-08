# 官方 OCR Web 稳定版说明

日期：2026-06-08

## 版本定位

本版本将项目收束为“官方 OCR 稳定版”。Web/手机端固定使用官方 PaddleOCR mobile OCR，不再把自训练识别模型作为默认或推荐路径。

核心判断：

- 当前主要瓶颈是检测框、crop 完整性、拍摄姿态和漏框补框。
- 官方 OCR 对清晰完整 crop 已经足够稳定。
- 继续训练自定义识别模型收益不稳定，且截断/补框/模糊样本容易污染训练。

## Web OCR 策略

`mobile_server.py` 使用：

- `WEB_OCR_POLICY = "official_paddleocr_v5_mobile"`
- `load_paddle_ocr(use_env=False)`
- `ocr_model_cache_tag(use_env=False)`

这意味着 Web 端会忽略 `BOOK_OCR_REC_MODEL_DIR`。即使系统环境变量里配置了实验模型，Web 也仍然使用官方 OCR。

命令行脚本仍保留实验入口，便于以后单独做离线对比；但 Web 稳定版不使用它。

## Web 模式

普通模式：

- `max_side = 1600`
- 不开启多变体 crop retry
- 用于日常快速巡检

精细模式：

- `max_side = 2000`
- 开启 crop retry
- 用于 OCR 失败、黄色较多、边缘 crop 可疑的图片

## 缓存策略

当前缓存版本：

```text
web_official_ocr_stable_20260608_v1
```

缓存 key 同时包含：

- Web 规则版本
- OCR 模型标签
- 处理模式
- 图片内容 hash

修改以下内容后应更新 `WEB_CACHE_VERSION`：

- Web 默认 OCR 策略
- 检测框/crop 规则
- OCR 后处理规则
- 黄色/红色判定规则
- 普通/精细模式参数

## 最近全量回归基线

最终回归目录：

```text
D:\upload_ascii\stage5_mobile_results\regression_20260607_hardcase_patch_full_v2
```

统计：

- 总检测：991
- 绿色：925
- 黄色：63
- 红色：3

红色项保留为较可靠错架提示：

- `35a7ea6497538bff_IMG20260523100857` 第 4 项 `D669.6/LNN`
- `c973e1dd854ad2be_IMG20260523100852` 第 26 项 `D669.6/LNN`
- `f4abebd44362efca_IMG20260517094107` 第 9 项 `D669.3/NCJ`

## 后续优化方向

优先继续优化：

- 边缘 crop 截断提示。
- 书号过近时的重叠 crop 策略。
- 薄书和漏检补框。
- 低置信 OCR 的黄色分类。
- 姿态差图片的重拍提示。
- Web 结果展示和诊断页可读性。

暂不优化：

- OCR 训练模型接入 Web。
- 数据库馆藏比对。
- 缺失书籍自动判断。
