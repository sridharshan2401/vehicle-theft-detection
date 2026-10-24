"""Draw parking / restricted / exit zones over a frame of the camera or a video.

    python -m app.zone_editor --camera 0 [--out configs/zones.json]
    python -m app.zone_editor --video "C:\\videos\\lot.mp4" [--frame 120] [--out configs/zones.json]

With --camera, one frame is captured and the camera is released straight away;
point the camera where it will stay, because zones are tied to the view.

Left click      add a point
Enter / right   finish the current polygon (name and kind are asked in the terminal)
u               undo the last point
d               delete the last finished zone
s               save
q / Esc         quit
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from app.config import load_config, resolve_path
from app.display import ZONE_COLORS
from app.parking_zones import ZONE_KINDS, Zone, ZoneError, ZoneSet
from app.video_reader import VideoError, probe_video

KIND_KEYS = {"p": "parking", "r": "restricted", "z": "proximity", "x": "exit"}


def read_frame(path: str, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise VideoError(f"Could not read frame {index}")
    return frame


def ask_zone_details(existing: set[str]) -> tuple[str, str] | None:
    kinds = ", ".join(f"{k}={v}" for k, v in KIND_KEYS.items())
    kind = input(f"Zone kind ({kinds}) [p]: ").strip().lower() or "p"
    kind = KIND_KEYS.get(kind, kind)
    if kind not in ZONE_KINDS:
        print(f"Unknown kind; choose one of {ZONE_KINDS}")
        return None
    default = f"{kind}_{len(existing) + 1}"
    name = input(f"Zone name [{default}]: ").strip() or default
    if name in existing:
        print("A zone with that name already exists.")
        return None
    return name, kind


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.zone_editor")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int, metavar="N", help="capture one frame from camera N")
    src.add_argument("--video", help="draw on a frame of this video")
    ap.add_argument("--frame", type=int, default=0, help="video frame to draw on")
    ap.add_argument("--out", help="zone file (default: zones.file from settings)")
    args = ap.parse_args(argv)

    if args.camera is not None:
        from dataclasses import replace

        from app.camera import CameraError, CameraSource

        cam = CameraSource(replace(load_config().camera, index=args.camera))
        try:
            cam.open()
            frame = cam.first_frame
        except CameraError as exc:
            print(f"error: {exc}")
            return 5
        finally:
            cam.release()
    else:
        try:
            info = probe_video(args.video)
            frame = read_frame(info.path, args.frame)
        except VideoError as exc:
            print(f"error: {exc}")
            return 2
    out = resolve_path(args.out) if args.out else resolve_path(load_config().zones.file)
    try:
        zones = ZoneSet.load(out)
    except ZoneError as exc:
        print(f"error: {exc}")
        return 2
    h, w = frame.shape[:2]
    scale = min(1.0, 1280 / w)
    current: list[list[float]] = []
    dirty = False
    win = "Zone editor"

    def on_mouse(event, x, y, flags, _):
        nonlocal dirty
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append([min(1.0, max(0.0, x / scale / w)), min(1.0, max(0.0, y / scale / h))])
        elif event == cv2.EVENT_RBUTTONDOWN:
            finish()

    def finish():
        nonlocal dirty
        if len(current) < 3:
            print("A zone needs at least 3 points.")
            return
        details = ask_zone_details({z.name for z in zones.zones})
        if details:
            zones.zones.append(Zone(details[0], details[1], [p[:] for p in current]))
            dirty = True
            print(f"Added {details[1]} zone '{details[0]}'.")
        current.clear()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print(__doc__)
    while True:
        view = frame.copy()
        for z in zones.zones:
            pts = np.round(z.scaled(w, h)).astype(np.int32)
            col = ZONE_COLORS.get(z.kind, (255, 255, 255))
            cv2.polylines(view, [pts], True, col, 2)
            cv2.putText(view, f"{z.kind}:{z.name}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        if current:
            pts = np.round(np.array(current) * [w, h]).astype(np.int32)
            cv2.polylines(view, [pts], False, (0, 255, 0), 2)
            for p in pts:
                cv2.circle(view, tuple(p), 4, (0, 255, 0), -1)
        if scale != 1.0:
            view = cv2.resize(view, (int(w * scale), int(h * scale)))
        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):
            finish()
        elif key == ord("u") and current:
            current.pop()
        elif key == ord("d") and zones.zones:
            print(f"Removed zone '{zones.zones.pop().name}'.")
            dirty = True
        elif key == ord("s"):
            zones.save(out)
            dirty = False
            print(f"Saved {len(zones.zones)} zone(s) to {out}")
        elif key in (ord("q"), 27):
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    if dirty:
        print("Unsaved changes were discarded (press 's' to save next time).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
