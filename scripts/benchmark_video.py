"""Run the pipeline headless on a video and record real performance numbers.

    python scripts/benchmark_video.py --video data\\samples\\clip.mp4 --mode both
    python scripts/benchmark_video.py --video clip.mp4 --mode responsive --upscale 1920x1080

--upscale writes a resized copy first (handy for testing 1080p throughput when
you only have lower-resolution footage; the content is the same, just bigger).
Results go to evidence/benchmarks/<timestamp>.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from app.config import load_config, resolve_path  # noqa: E402
from app.hardware import inspect_hardware, select_device  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.video_reader import probe_video  # noqa: E402


def upscale(src: Path, size: str, out_dir: Path, fps: float | None = None) -> Path:
    w, h = (int(v) for v in size.lower().split("x"))
    info = probe_video(src)
    fps = fps or info.fps
    dst = out_dir / f"{src.stem}_{w}x{h}_{fps:g}fps.mp4"
    if dst.exists():
        return dst
    cap = cv2.VideoCapture(str(src))
    wr = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        wr.write(cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC))
    cap.release()
    wr.release()
    return dst


def run_once(video: Path, mode: str, args, device: str, half: bool, hw: dict) -> dict:
    overrides = {"app.mode": mode, "app.display": False}
    if args.imgsz:
        overrides["detector.imgsz"] = args.imgsz
        overrides["detector.min_imgsz"] = min(416, args.imgsz)
    if args.weights:
        overrides["detector.weights"] = args.weights
    if args.no_secondary:
        overrides.update({"reid.enabled": False, "clothing.enabled": False})
    if args.no_save_video:
        overrides["evidence.save_annotated_video"] = False
    s = load_config(args.config, overrides)
    info = probe_video(video)
    summary = Pipeline(s, info, device, half, max_frames=args.max_frames, hardware=hw).run()
    st = summary["stats"]
    return {
        "video": info.describe(),
        "mode": mode,
        "device": device,
        "fp16": summary["fp16"],
        "detector": s.detector.weights,
        "imgsz": s.detector.imgsz,
        "final_imgsz": st["final_imgsz"],
        "reid": summary["models"]["reid"].split(" |")[0],
        "clothing": summary["models"]["clothing"],
        "source_fps": info.fps,
        "frames_read": st["frames_read"],
        "frames_processed": st["frames_processed"],
        "frames_skipped": st["frames_skipped"],
        "frames_balanced": summary["frames_balanced"],
        "wall_time_s": summary["wall_time_seconds"],
        "avg_processed_fps": summary["average_processing_fps"],
        "detector_ms_mean": st["detector_latency"]["mean_ms"],
        "detector_ms_max": st["detector_latency"]["max_ms"],
        "frame_ms_mean": st["frame_latency"]["mean_ms"],
        "reid_ms_per_crop_mean": st["reid_latency_per_crop"]["mean_ms"],
        "clothing_ms_mean": st["clothing_latency"]["mean_ms"],
        "secondary_jobs_done": st["secondary_jobs_done"],
        "secondary_jobs_dropped": st["secondary_jobs_dropped"],
        "input_queue_max_depth": st["input_queue_max_depth"],
        "gpu_memory": st["gpu_memory"],
        "session": summary["session_id"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--mode", choices=["responsive", "complete", "both"], default="both")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--weights")
    ap.add_argument("--config")
    ap.add_argument("--upscale", help="e.g. 1920x1080")
    ap.add_argument("--retime-fps", type=float,
                    help="with --upscale: write the copy at this frame rate (same frames, played faster); "
                         "a throughput test, not real footage at that rate")
    ap.add_argument("--no-secondary", action="store_true", help="disable ReID and clothing")
    ap.add_argument("--no-save-video", action="store_true", help="do not write the full annotated video")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    out_dir = resolve_path("evidence/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    video = Path(args.video)
    if args.upscale:
        video = upscale(video, args.upscale, out_dir, args.retime_fps)
    hw = inspect_hardware()
    cfg = load_config(args.config)
    device, half = select_device(cfg.device.device, cfg.device.half, hw)
    modes = ["complete", "responsive"] if args.mode == "both" else [args.mode]
    results = [run_once(video, m, args, device, half, hw.as_dict()) for m in modes]

    record = {"when": datetime.now().isoformat(timespec="seconds"), "hardware": hw.as_dict(), "results": results}
    out = out_dir / f"benchmark_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    for r in results:
        print(f"\n[{r['mode']}] {r['video']}")
        for k in ("device", "fp16", "imgsz", "final_imgsz", "frames_read", "frames_processed", "frames_skipped",
                  "frames_balanced", "wall_time_s", "avg_processed_fps", "detector_ms_mean", "frame_ms_mean",
                  "reid_ms_per_crop_mean", "clothing_ms_mean", "secondary_jobs_dropped", "gpu_memory"):
            print(f"  {k:<24} {r[k]}")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
