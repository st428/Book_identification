from __future__ import annotations

import os
from pathlib import Path


PADDLEOCR_REPO = (
    Path(r"D:\CPPTicketManager-main\CPPTicketManager-main\Lib\site-packages")
    / "paddlex"
    / "repo_manager"
    / "repos"
    / "PaddleOCR"
)


def main() -> None:
    os.environ.setdefault("PADDLE_PDX_PADDLEOCR_PATH", str(PADDLEOCR_REPO))
    import paddlex.repo_apis.PaddleOCR_api.text_rec.register  # noqa: F401
    from paddlex.engine import Engine

    Engine().run()


if __name__ == "__main__":
    main()
