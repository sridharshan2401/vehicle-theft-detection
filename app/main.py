"""Entry point: python -m app.main --video "C:\\path\\to\\video.mp4"

Exit codes: 0 ok, 1 analysis failed, 2 bad input or config, 3 model problem, 4 GPU/device problem.
"""

from __future__ import annotations

import json
import logging
import sys

from app.cli import build_parser, overrides_from_args, pick_video_file
from app.config import ConfigError, load_config
from app.hardware import DeviceError, inspect_hardware, select_device
from app.video_reader import VideoError, probe_video

log = logging.getLogger("app")


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level, logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


def print_summary(summary: dict) -> None:
    acc = summary["stats"]
    print("\n=== Session summary ===")
    print(f"Session:        {summary['session_id']} ({summary['status']})")
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
    if summary.get("write_errors"):
        print(f"WRITE ERRORS:   {summary['write_errors']}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_config(args.config, overrides_from_args(args))
    except ConfigError as exc:
        setup_logging("INFO")
        log.error("%s", exc)
        return 2
    setup_logging(settings.app.log_level)

    video = args.video
    if not video:
        print("No --video given; opening a file picker...")
        video = pick_video_file()
        if not video:
            log.error("No video selected. Pass one with --video \"C:\\path\\to\\video.mp4\".")
            return 2
    try:
        info = probe_video(video)
    except VideoError as exc:
        log.error("%s", exc)
        return 2
    print(f"Video: {info.describe()}")

    hw = inspect_hardware()
    print(hw.describe())
    try:
        device, half = select_device(settings.device.device, settings.device.half, hw)
    except DeviceError as exc:
        log.error("%s", exc)
        return 4
    print(f"Using device {device}{' with FP16' if half else ''}; mode: {settings.app.mode}")

    from app.detector import ModelUnavailableError
    from app.pipeline import Pipeline, PipelineError

    pipeline = Pipeline(settings, info, device, half, max_frames=args.max_frames, start_frame=args.start_frame,
                        hardware=hw.as_dict())
    try:
        summary = pipeline.run()
    except ModelUnavailableError as exc:
        log.error("%s", exc)
        return 3
    except PipelineError as exc:
        log.error("%s", exc)
        return 1
    print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
