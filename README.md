# Library Shelf Inspector

图书馆书架智能巡检与索书号识别系统。当前主线是“提速版 + Web/手机端”：通过红色书脊标签定位、OCR 识别索书号、横向排序判断和手机端多图上传，辅助发现疑似错架位置。

## 当前可用能力

- Web/手机端 Flask 服务：`mobile_server.py`
- 核心识别与排序逻辑：`shelf_inspector_fast.py`
- 前端页面：`mobile_web/`
- Windows 启动入口：`启动Web端.exe`
- 训练/标注辅助脚本：
  - `export_rec_dataset.py`
  - `build_review_queue.py`
  - `import_corrected_review.py`
  - `organize_project_data.py`
  - `prepare_rec_training.py`
  - `prepare_rec_full_dict_training.py`
  - `run_rec_training.py`
  - `evaluate_rec_checkpoint.py`

## 运行

安装依赖后运行：

```powershell
python mobile_server.py
```

然后在电脑或同一网络下的手机浏览器打开服务地址。校园网可能存在 AP 隔离，手机无法访问电脑时，优先使用电脑连接手机热点或确认电脑防火墙已允许 Python/Flask 入站。

## 数据与缓存

`stage5_mobile_results/` 主要是上传图片、识别结果、训练导出、训练输出和评估缓存。这些内容通常不进入 Git；需要复现实验时，用脚本从原始照片重新导出。

后续训练路线见 [docs/training_plan.md](docs/training_plan.md)。
