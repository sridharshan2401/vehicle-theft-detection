"""Print what is actually installed and usable: Python, PyTorch, CUDA, GPU, OpenCV, models.

    python scripts/check_environment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_config, resolve_path  # noqa: E402
from app.hardware import inspect_hardware  # noqa: E402
from app.model_registry import as_table  # noqa: E402


def main() -> int:
    hw = inspect_hardware()
    print("== Hardware and libraries ==")
    print(hw.describe())
    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        print(f"note: tested with Python 3.11/3.12; you are on {sys.version.split()[0]}")

    try:
        import cv2

        build = cv2.getBuildInformation()
        gui = [line.strip() for line in build.splitlines() if line.strip().startswith(("GUI:", "Win32 UI:"))]
        video = [line.strip() for line in build.splitlines() if line.strip().startswith(("FFMPEG:", "MSMF:", "DirectShow:"))]
        print("OpenCV GUI:", "; ".join(gui) or "unknown", "| video I/O:", "; ".join(video) or "unknown")
    except ImportError:
        pass
    for pkg in ("transformers", "torchvision", "torchreid"):
        try:
            mod = __import__(pkg)
            print(f"{pkg}: {getattr(mod, '__version__', 'installed')}")
        except ImportError:
            print(f"{pkg}: not installed" + (" (optional)" if pkg == "torchreid" else ""))

    print("\n== Models ==")
    cfg = load_config()
    for m in as_table():
        on_disk = "?"
        if not m["path"].startswith("<"):
            on_disk = "present" if resolve_path(m["path"].rstrip("/")).exists() else "missing"
        flag = "required" if m["required"] else "optional"
        print(f"- {m['key']:<14} {on_disk:<8} verified={m.get('verified', False)!s:<5} ({flag}) {m['purpose']}")
        if on_disk == "missing":
            print(f"    setup: {m['setup']}")
    print(f"\nConfigured detector: {cfg.detector.weights}")

    ok = hw.cuda_works or hw.torch is not None
    print("\nResult:", "GPU ready" if hw.cuda_works else "CPU only (slow)" if hw.torch else "PyTorch missing")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
