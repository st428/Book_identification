# Final Stable Version

This folder is the runnable final Web version of the library shelf inspector.

Run:

```powershell
python mobile_server.py
```

Or double-click:

```text
启动Web端.exe
```

Web policy:

- Official PaddleOCR only.
- Normal mode for daily use: `max_side=1200`.
- Fine mode for suspicious OCR/crop cases: `max_side=1600` with crop retry.
- Runtime uploads and results are written under `stage5_mobile_results/`.
- Cache version: `web_official_ocr_stable_20260608_v5`.

Key docs:

- `docs/final_project_summary.md`
- `docs/final_demo_test_20260608.md`
- `docs/known_limitations.md`
