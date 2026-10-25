"""Drawing the annotated frame and running the OpenCV window.

The same annotated frame is shown on screen, written to the annotated video
and used for evidence snapshots, so it is rendered exactly once per frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.datatypes import GROUP_PERSON, GROUP_VEHICLE, Detection
from app.parking_zones import ZoneSet
from app.segmenter import render_instance_masks

LEVEL_COLORS = {
    "INFORMATIONAL": (170, 170, 170),
    "LOW": (0, 215, 255),
    "MEDIUM": (0, 140, 255),
    "HIGH": (0, 0, 255),
    "CRITICAL": (255, 0, 255),
}
ZONE_COLORS = {"parking": (255, 200, 0), "restricted": (0, 0, 255), "proximity": (0, 255, 255), "exit": (255, 0, 255)}
VEHICLE_PALETTE = [(255, 128, 0), (255, 191, 64), (200, 120, 40), (255, 160, 120), (230, 90, 30), (255, 220, 120)]
PERSON_PALETTE = [(60, 220, 60), (0, 200, 255), (180, 60, 255), (60, 255, 200), (120, 160, 255), (40, 255, 120),
                  (255, 90, 200), (0, 255, 255)]
OVERLAY_MODES = ["full", "boxes", "masks", "minimal"]


def track_color(det: Detection) -> tuple[int, int, int]:
    if det.track_id is None:
        return (128, 128, 128)
    n = int(det.track_id[1:])
    palette = VEHICLE_PALETTE if det.group == GROUP_VEHICLE else PERSON_PALETTE
    return palette[n % len(palette)]


@dataclass
class RenderContext:
    detections: list[Detection]
    labels: dict[str, list[str]] = field(default_factory=dict)  # track id -> label lines
    vehicle_levels: dict[str, str] = field(default_factory=dict)  # track id -> risk level
    hud: list[str] = field(default_factory=list)
    events: list[tuple[str, str]] = field(default_factory=list)  # (level, text), newest last
    banner: str | None = None
    zones: ZoneSet | None = None


class Renderer:
    def __init__(self, frame_size: tuple[int, int]):
        w, _ = frame_size
        self.scale = max(0.4, min(1.0, w / 1920 * 0.75))
        self.thick = 1 if w < 1600 else 2
        self.overlay_mode = 0

    def cycle_overlay(self) -> str:
        self.overlay_mode = (self.overlay_mode + 1) % len(OVERLAY_MODES)
        return OVERLAY_MODES[self.overlay_mode]

    @property
    def mode_name(self) -> str:
        return OVERLAY_MODES[self.overlay_mode]

    def _text_size(self, text: str) -> tuple[int, int]:
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self.scale, self.thick)
        return tw, th + base

    def _fit(self, text: str, max_w: int) -> str:
        """Shorten text with '...' until it fits ``max_w`` pixels (full text stays in the logs)."""
        if self._text_size(text)[0] <= max_w:
            return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._text_size(text[:mid] + "...")[0] <= max_w:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo] + "..."

    def _label(self, img, lines, x, top, bottom, color, placed) -> None:
        """Draw a stacked label above the box, or below it, avoiding the HUD and labels already placed."""
        sizes = [self._text_size(t) for t in lines]
        w = max(s[0] for s in sizes) + 8
        h = sum(s[1] for s in sizes) + 4 * len(lines)
        H, W = img.shape[:2]
        x = int(min(max(0, x), max(0, W - w)))
        step = h + 2
        candidates = [top - h, bottom + 2] + [top - h - k * step for k in range(1, 4)] + \
                     [bottom + 2 + k * step for k in range(1, 10)]

        def free(y0):
            rect = (x, y0, x + w, y0 + h)
            return 0 <= y0 <= H - h and not any(
                not (rect[2] < r[0] or rect[0] > r[2] or rect[3] < r[1] or rect[1] > r[3]) for r in placed)

        y0 = next((int(c) for c in candidates if free(int(c))), int(min(max(0, top - h), max(0, H - h))))
        placed.append((x, y0, x + w, y0 + h))
        sub = img[y0:y0 + h, x:x + w]
        sub[:] = (sub * 0.35).astype(np.uint8)
        cv2.rectangle(img, (x, y0), (x + w, y0 + h), color, 1)
        ty = y0
        for text, (_, th) in zip(lines, sizes):
            ty += th + 2
            cv2.putText(img, text, (x + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, self.scale, (255, 255, 255), self.thick, cv2.LINE_AA)

    def _panel_rect(self, img, lines, x, y, anchor_right=False) -> tuple[int, int, int, int]:
        sizes = [self._text_size(t) for t in lines]
        w = max(s[0] for s in sizes) + 12
        h = sum(s[1] + 4 for s in sizes) + 8
        if anchor_right:
            x = img.shape[1] - w - x
        x0, y0 = max(0, x), max(0, y)
        return x0, y0, min(img.shape[1], x0 + w), min(img.shape[0], y0 + h)

    def _panel(self, img, lines, x, y, colors=None, anchor_right=False) -> None:
        if not lines:
            return
        sizes = [self._text_size(t) for t in lines]
        x0, y0, x1, y1 = self._panel_rect(img, lines, x, y, anchor_right)
        sub = img[y0:y1, x0:x1]
        sub[:] = (sub * 0.3).astype(np.uint8)
        ty = y0 + 4
        for i, (text, (_, th)) in enumerate(zip(lines, sizes)):
            ty += th + 4
            color = colors[i] if colors else (255, 255, 255)
            cv2.putText(img, text, (x0 + 6, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, self.scale, color, self.thick, cv2.LINE_AA)

    def render(self, frame: np.ndarray, ctx: RenderContext) -> np.ndarray:
        img = frame.copy()
        mode = self.mode_name
        if ctx.zones is not None and mode != "minimal":
            h, w = img.shape[:2]
            for z in ctx.zones.zones:
                pts = np.round(z.scaled(w, h)).astype(np.int32)
                col = ZONE_COLORS.get(z.kind, (255, 255, 255))
                cv2.polylines(img, [pts], True, col, 1, cv2.LINE_AA)
                cv2.putText(img, f"{z.kind}:{z.name}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, self.scale * 0.8, col, 1, cv2.LINE_AA)

        dets = ctx.detections
        if mode in ("full", "masks"):
            render_instance_masks(img, [(d.polygon, track_color(d)) for d in dets if d.polygon is not None], 0.45)

        # Reserve the HUD and banner areas so object labels are placed around them.
        placed: list[tuple[int, int, int, int]] = []
        if ctx.hud:
            placed.append(self._panel_rect(img, ctx.hud, 8, 8))
        if ctx.banner:
            placed.append(self._panel_rect(img, [ctx.banner], 8, 8, anchor_right=True))
        if mode != "minimal":
            # People last so their labels sit on top of the larger vehicle boxes.
            for d in sorted(dets, key=lambda d: d.group == GROUP_PERSON):
                color = track_color(d)
                x1, y1, x2, y2 = d.box.astype(int)
                if mode in ("full", "boxes"):
                    box_color = LEVEL_COLORS.get(ctx.vehicle_levels.get(d.track_id or ""), color)
                    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2 if d.group == GROUP_VEHICLE else 1)
                lines = ctx.labels.get(d.track_id or "") or [
                    f"{d.track_id or 'untracked'} {d.class_name} {d.conf:.2f}"
                ]
                if mode == "masks":
                    lines = lines[:1]
                self._label(img, lines, x1, y1, y2, color, placed)

        self._panel(img, ctx.hud, 8, 8)
        if ctx.events and mode != "minimal":
            ev = [(lv, self._fit(t, img.shape[1] - 30)) for lv, t in ctx.events[-6:]]
            sizes_h = sum(self._text_size(t)[1] + 4 for _, t in ev) + 8
            self._panel(img, [t for _, t in ev], 8, img.shape[0] - sizes_h - 8,
                        colors=[LEVEL_COLORS.get(lv, (255, 255, 255)) for lv, _ in ev])
        if ctx.banner:
            self._panel(img, [ctx.banner], 8, 8, colors=[(0, 255, 255)], anchor_right=True)
        return img


def render_status(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Start-up screen: the raw picture, darkened, with a few lines of status text on top.

    Used while the models load so the window (and, in live mode, the camera picture)
    appears straight away instead of after 10-20 s of silence.
    """
    img = (frame * 0.55).astype(np.uint8)
    h, w = img.shape[:2]
    scale = max(0.5, min(1.1, w / 1280 * 0.9))
    thick = 1 if w < 1100 else 2
    y = int(40 * scale) + 10
    for i, text in enumerate(lines):
        color = (0, 255, 255) if i == 0 else (255, 255, 255)
        cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        y += int(36 * scale) + 6
    return img


class DisplayWindow:
    """OpenCV window. Must be driven from the main thread on Windows."""

    QUIT_KEYS = {ord("q"), ord("Q"), 27}

    def __init__(self, name: str, max_width: int):
        self.name = name
        self.max_width = max_width
        self.opened = False

    def show(self, frame: np.ndarray) -> None:
        if not self.opened:
            cv2.namedWindow(self.name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            h, w = frame.shape[:2]
            scale = min(1.0, self.max_width / w)
            cv2.resizeWindow(self.name, int(w * scale), int(h * scale))
            self.opened = True
        cv2.imshow(self.name, frame)

    def poll_key(self, wait_ms: int = 1) -> int:
        key = cv2.waitKey(wait_ms) & 0xFF
        return key

    def show_status(self, frame: np.ndarray) -> None:
        """A start-up/status frame (same window; separate method so tests can tell them apart)."""
        self.show(frame)

    def poll_quit(self, wait_ms: int = 1) -> bool:
        """True if the user pressed Q / Esc or closed the window (used while starting up)."""
        return self.poll_key(wait_ms) in self.QUIT_KEYS or self.closed_by_user()

    def closed_by_user(self) -> bool:
        if not self.opened:
            return False
        try:
            return cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def close(self) -> None:
        if self.opened:
            cv2.destroyWindow(self.name)
            cv2.waitKey(1)
            self.opened = False
