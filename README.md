# Library Shelf Inspector

图书馆书架智能巡检与索书号识别系统。

当前稳定主线是“官方 OCR + Web/手机端 + 检测/crop/规则优化”。系统不做数据库馆藏比对，不自动判断缺失书籍，也不再默认接入自训练 OCR 识别模型。

## 核心文件

- `shelf_inspector_fast.py`：检测、裁切、OCR、排序判断、后处理、诊断页导出。
- `mobile_server.py`：Flask Web/手机端服务。
- `mobile_web/`：前端页面。
- `stage5_mobile_results/`：上传、缓存、结果、诊断页和最终演示测试输出。
- `data/`：本地训练/测试图片，包含原始样本、hard case 上传图和额外整图素材；图片总量较大，GitHub 默认只保留 `data/README.md` 说明。
- `docs/`：最终说明、测试记录、已知限制和策略文档。

## Web 稳定版

Web 端固定使用官方 PaddleOCR mobile 模型，不读取 `BOOK_OCR_REC_MODEL_DIR`，避免误接入实验训练模型。

处理模式：

- 普通模式：`max_side=1200`，速度优先，适合日常上传。
- 精细模式：`max_side=1600`，开启可疑 crop 重试，速度较慢，用于 OCR 失败、边缘 crop 或黄色较多的图片。

当前缓存版本：`web_official_ocr_stable_20260608_v5`。

缓存版本由 `mobile_server.py` 中的 `WEB_CACHE_VERSION` 控制。修改 Web 行为、OCR 策略、crop 规则或后处理规则后，应 bump 缓存版本，避免用户看到旧结果。

## 运行

```powershell
python mobile_server.py
```

浏览器打开：

- 本机：`http://127.0.0.1:5000`
- 手机：启动日志里显示的局域网地址

如果手机无法访问电脑服务，优先检查是否连接同一个网络、电脑防火墙是否允许 Python/Flask 入站。

## 当前优化重点

- 横排书架检测框稳定性。
- crop 完整性和边缘截断提示。
- 漏检补框，但补框只标黄，不直接作为红色错架依据。
- 黄色复核体系。
- Web 多图上传、缓存复用和结果展示。
- 姿态差图片提示用户重拍。

## 不再作为当前主线

- 不继续默认训练识别模型。
- 不把补框、截断、遮挡、模糊 crop 加入训练。
- 不在 Web 端部署 `manual_only_20260602_guarded_ep2`。
- 不做馆藏数据库比对和缺失书籍自动判断。

## 交付文档

- `docs/final_project_summary.md`：项目整体流程和最终策略。
- `docs/final_demo_test_20260608.md`：最终演示集测试记录。
- `docs/known_limitations.md`：已知限制和使用建议。
- `docs/project_file_and_code_guide_20260608.md`：final 目录文件功能和 Python 代码结构说明。
- `docs/report_writing_guidance_20260608.md`：课程项目报告写作指导。
