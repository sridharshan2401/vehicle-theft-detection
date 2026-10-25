"""Start live webcam mode from any folder.

    C:\\path\\to\\venv\\Scripts\\python.exe E:\\vehicle-theft-detection\\run_webcam.py [--camera 1] [other options]

`python -m app.main` only works when the current folder is the project folder; this
launcher works from anywhere (for example the VS Code terminal opened in your home
folder). It also refuses to start under a Python that lacks the project's packages,
which is the usual reason for "No module named ..." errors or a CPU-only slow run.
All options are passed to app.main; with no options it opens camera 0.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = ["cv2", "numpy", "yaml", "torch", "ultralytics"]


def missing_packages() -> list[str]:
    """Packages this Python cannot find (checked without importing them, so it is instant)."""
    return [name for name in REQUIRED if importlib.util.find_spec(name) is None]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(a in ("--video", "--pick-video") or a.startswith("--video=") for a in argv):
        print("run_webcam.py is for the live camera. For a video file use:  python -m app.main --video <file>")
        return 2
    missing = missing_packages()
    if missing:
        print(f"This Python ({sys.executable}) is missing: {', '.join(missing)}.\n"
              "Start the app with the project's virtual environment, e.g.\n"
              "  C:\\path\\to\\venv\\Scripts\\python.exe run_webcam.py\n"
              "(see README > Quick start), or install the requirements into this one.")
        return 2
    sys.path.insert(0, str(ROOT))
    from app.main import main as app_main

    return app_main(argv)


if __name__ == "__main__":
    sys.exit(main())
