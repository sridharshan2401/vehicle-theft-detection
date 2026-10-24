"""Command-line arguments and the video file picker."""

from __future__ import annotations

import argparse
from typing import Any


def _resolution(text: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in text.lower().split("x"))
        return w, h
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use WIDTHxHEIGHT, e.g. 1280x720") from exc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.main",
        description="Watch the live webcam (default) or a video file for suspicious behaviour around vehicles. "
                    "Reports risk levels with reasons; it cannot verify that a vehicle is stolen.",
    )
    src = p.add_argument_group("input (default: live camera 0)")
    src.add_argument("--camera", type=int, metavar="N", help="camera index: 0 = built-in webcam, 1, 2... = other cameras")
    src.add_argument("--resolution", type=_resolution, metavar="WxH", help="requested camera resolution, e.g. 1280x720")
    src.add_argument("--camera-backend", choices=["auto", "dshow", "msmf", "any"], help="OpenCV camera backend")
    src.add_argument("--record", action="store_true", help="live camera: also save the full annotated feed (off by default)")
    src.add_argument("--alert-clips", action="store_true", help="live camera: save short clips around MEDIUM+ alerts")
    src.add_argument("--video", help="analyse this prerecorded video file instead of the camera")
    src.add_argument("--pick-video", action="store_true", help="choose a video file in a dialog instead of using the camera")
    p.add_argument("--config", help="settings YAML (default: configs/settings.yaml)")
    p.add_argument("--mode", choices=["responsive", "complete"],
                   help="video files only: responsive preview or complete evidence (the camera is always live)")
    p.add_argument("--no-display", action="store_true", help="run without a window (for batch runs and benchmarks)")
    p.add_argument("--device", help="auto, cpu, cuda or cuda:N")
    p.add_argument("--weights", help="detector weights, e.g. models/yolo11n-seg.pt")
    p.add_argument("--imgsz", type=int, help="detector input size (multiple of 32)")
    p.add_argument("--tracker", choices=["bytetrack", "botsort"])
    p.add_argument("--zones", help="zone file (JSON) made with python -m app.zone_editor")
    p.add_argument("--max-frames", type=int, help="stop after this many source/camera frames")
    p.add_argument("--start-frame", type=int, default=0, help="video files only: start at this frame")
    p.add_argument("--no-reid", action="store_true", help="disable person re-identification")
    p.add_argument("--no-clothing", action="store_true", help="disable clothing analysis")
    p.add_argument("--no-save-video", action="store_true", help="do not write the full annotated video")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    o: dict[str, Any] = {}
    if args.video or args.pick_video:
        o["app.source"] = "video"
    if args.camera is not None:
        o["camera.index"] = args.camera
        o["app.source"] = "camera"
    if args.resolution:
        o["camera.width"], o["camera.height"] = args.resolution
    if args.camera_backend:
        o["camera.backend"] = args.camera_backend
    if args.record:
        o["camera.save_annotated_video"] = True
    if args.alert_clips:
        o["camera.save_alert_clips"] = True
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
