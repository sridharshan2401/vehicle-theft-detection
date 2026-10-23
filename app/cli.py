"""Command-line arguments and the video file picker."""

from __future__ import annotations

import argparse
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.main",
        description="Analyse a prerecorded parking video for suspicious vehicle interactions.",
    )
    p.add_argument("--video", help="path to the video file; a file picker opens if omitted")
    p.add_argument("--config", help="settings YAML (default: configs/settings.yaml)")
    p.add_argument("--mode", choices=["responsive", "complete"], help="responsive preview or complete evidence")
    p.add_argument("--no-display", action="store_true", help="run without a window (for batch runs and benchmarks)")
    p.add_argument("--device", help="auto, cpu, cuda or cuda:N")
    p.add_argument("--weights", help="detector weights, e.g. models/yolo11n-seg.pt")
    p.add_argument("--imgsz", type=int, help="detector input size (multiple of 32)")
    p.add_argument("--tracker", choices=["bytetrack", "botsort"])
    p.add_argument("--zones", help="zone file (JSON) made with python -m app.zone_editor")
    p.add_argument("--max-frames", type=int, help="stop after this many source frames")
    p.add_argument("--start-frame", type=int, default=0, help="start at this source frame")
    p.add_argument("--no-reid", action="store_true", help="disable person re-identification")
    p.add_argument("--no-clothing", action="store_true", help="disable clothing analysis")
    p.add_argument("--no-save-video", action="store_true", help="do not write the full annotated video")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    o: dict[str, Any] = {}
    if args.mode:
        o["app.mode"] = args.mode
    if args.no_display:
        o["app.display"] = False
    if args.device:
        o["device.device"] = args.device
    if args.weights:
        o["detector.weights"] = args.weights
    if args.imgsz:
        o["detector.imgsz"] = args.imgsz
        o["detector.min_imgsz"] = min(args.imgsz, 416)
    if args.tracker:
        o["tracker.type"] = args.tracker
    if args.zones:
        o["zones.file"] = args.zones
    if args.no_reid:
        o["reid.enabled"] = False
    if args.no_clothing:
        o["clothing.enabled"] = False
    if args.no_save_video:
        o["evidence.save_annotated_video"] = False
    if args.log_level:
        o["app.log_level"] = args.log_level
    return o


def pick_video_file() -> str | None:
    """Open a native file dialog. Returns None if cancelled or no GUI is available."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askopenfilename(
            title="Select a parking video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v *.wmv"), ("All files", "*.*")],
        )
    finally:
        root.destroy()
    return path or None
