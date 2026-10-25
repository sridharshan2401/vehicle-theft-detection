"""Entry point.

    python -m app.main                      live webcam (camera 0), the normal way to run it
    python -m app.main --camera 1           another connected camera
    python -m app.main --video clip.mp4     analyse a prerecorded video file instead

Exit codes: 0 ok, 1 analysis failed, 2 bad input or config, 3 model problem, 4 GPU/device problem,
5 camera unavailable or disconnected.
"""

from __future__ import annotations

import json
import logging
import sys

from app.camera import CameraError, CameraSource
from app.cli import build_parser, overrides_from_args, pick_video_file
from app.config import ConfigError, load_config
from app.hardware import DeviceError, inspect_hardware, select_device
from app.video_reader import VideoError, probe_video

log = logging.getLogger("app")

LIMITATION = ("Note: this reports suspicious behaviour around vehicles with a risk level and reasons. "
              "It cannot verify that a vehicle is stolen (no number-plate reading or stolen-vehicle database).")


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level, logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


def print_summary(summary: dict) -> None:
    acc = summary["stats"]
    print("\n=== Session summary ===")
    print(f"Session:        {summary['session_id']} ({summary['status']})")
    print(f"Source:         {summary.get('source', '?')}")
    print(f"Mode:           {summary['mode']} on {summary['device']}{' FP16' if summary['fp16'] else ''}")
    print(f"Frames:         read {acc['frames_read']}, processed {acc['frames_processed']}, "
          f"skipped {acc['frames_skipped']} (balanced: {summary['frames_balanced']})")
    print(f"Wall time:      {summary['wall_time_seconds']}s, average {summary['average_processing_fps']} processed FPS")
    print(f"Detector:       mean {acc['detector_latency']['mean_ms']} ms per frame")
    print(f"Alerts:         {json.dumps({k: v for k, v in summary['alerts_by_level'].items() if v}) or 'none'}")
    print(f"Associations:   {summary['associations'] or 'none'}")
    print(f"ReID:           {summary['models']['reid']}")
    print(f"Clothing:       {summary['models']['clothing']}")
    print(f"Evidence:       {summary['evidence_dir']}")
    if "camera" in summary:
        cam = summary["camera"]
        print(f"Camera:         backend {cam['backend']}, released {cam['released']}, "
              f"full recording {'on' if cam['annotated_video_saved'] else 'off'}, "
              f"alert clips {'on' if cam['alert_clips_saved'] else 'off'}")
    if summary.get("write_errors"):
        print(f"WRITE ERRORS:   {summary['write_errors']}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.camera is not None and (args.video or args.pick_video):
        print("error: choose either --camera or --video/--pick-video, not both")
        return 2
    try:
        settings = load_config(args.config, overrides_from_args(args))
    except ConfigError as exc:
        setup_logging("INFO")
        log.error("%s", exc)
        return 2
    setup_logging(settings.app.log_level)

    print(f"Python: {sys.executable}")
    hw = inspect_hardware()
    print(hw.describe())
    try:
        device, half = select_device(settings.device.device, settings.device.half, hw)
    except DeviceError as exc:
        log.error("%s", exc)
        return 4
    if device == "cpu" and settings.device.device == "auto":
        print("WARNING: no usable CUDA GPU in this Python (PyTorch " + (hw.torch or "missing") + "), so analysis "
              "runs on the CPU and the live view will be slow. If you have an NVIDIA GPU, start the app with the "
              "project's virtual environment instead (see README > Quick start).")

    camera = None
    if settings.app.source == "camera":
        camera = CameraSource(settings.camera)
        print(f"Opening camera {settings.camera.index}...")
        try:
            info = camera.open()
        except CameraError as exc:
            log.error("%s", exc)
            return 5
        print(f"Camera: {info.describe()}")
        print("Controls: Q or Esc quit | Space pause | S snapshot | R overlays")
    else:
        video = args.video
        if not video:
            print("Opening a file picker...")
            video = pick_video_file()
            if not video:
                log.error("No video selected. Run without --video/--pick-video to use the camera, "
                          "or pass --video \"C:\\path\\to\\video.mp4\".")
                return 2
        try:
            info = probe_video(video)
        except VideoError as exc:
            log.error("%s", exc)
            return 2
        print(f"Video: {info.describe()}")
    print(f"Using device {device}{' with FP16' if half else ''}; "
          f"mode: {'live' if info.live else settings.app.mode}")
    print(LIMITATION)

    from app.detector import ModelUnavailableError
    from app.pipeline import Pipeline, PipelineError

    pipeline = Pipeline(settings, info, device, half, max_frames=args.max_frames, start_frame=args.start_frame,
                        hardware=hw.as_dict(), camera=camera)
    try:
        summary = pipeline.run()
    except ModelUnavailableError as exc:
        log.error("%s", exc)
        return 3
    except PipelineError as exc:
        log.error("%s", exc)
        return 5 if isinstance(exc.__cause__, CameraError) else 1
    finally:
        if camera is not None:
            camera.release()
    print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
