# Web 版使用说明

本文件夹是最终官方 OCR 稳定版。

## 启动

双击：

```text
启动Web端.exe
```

或在命令行运行：

```powershell
cd D:\个人\university\2026spring\lib\final
python mobile_server.py
```

启动后打开：

```text
http://127.0.0.1:5000
```

手机访问时使用启动日志中显示的局域网地址，并确保手机和电脑连接同一网络。

## 模式

- 普通模式：默认模式，速度优先，适合日常识别。
- 精细模式：较慢，会对可疑 crop 进行重试，适合黄色较多或 OCR 失败的图片。

当前参数：

- 普通模式：`max_side=1200`
- 精细模式：`max_side=1600`，开启 crop retry

## OCR 策略

Web 端固定使用官方 PaddleOCR mobile OCR，不读取 `BOOK_OCR_REC_MODEL_DIR`，不会误接入训练模型。

当前缓存版本：

```text
web_official_ocr_stable_20260608_v5
```

## 使用建议

- 日常先用普通模式。
- 黄色较多、crop 可疑或 OCR 失败时，再用精细模式。
- 如果系统提示图像姿态较差，应优先重新拍摄，而不是反复切换模式。
