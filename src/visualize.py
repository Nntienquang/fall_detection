"""Vẽ kết quả lên frame (TV3)."""
from __future__ import annotations

import cv2
import numpy as np

from .features import SKELETON
from .types import Person, State, TrackResult

COLORS = {  # BGR
    State.NORMAL: (80, 200, 80),
    State.SUSPECTED: (0, 190, 255),
    State.FALL: (40, 40, 230),
}


def draw_person(img: np.ndarray, p: Person, r: TrackResult | None, kp_thr: float) -> None:
    color = COLORS[r.state] if r else (200, 200, 200)
    x1, y1, x2, y2 = p.bbox.astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    for a, b in SKELETON:
        if p.kp_conf[a] >= kp_thr and p.kp_conf[b] >= kp_thr:
            pa = tuple(int(v) for v in p.keypoints[a])
            pb = tuple(int(v) for v in p.keypoints[b])
            cv2.line(img, pa, pb, color, 2, cv2.LINE_AA)
    for (x, y), c in zip(p.keypoints, p.kp_conf):
        if c >= kp_thr:
            cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), -1, cv2.LINE_AA)

    if r is not None:
        ang = "--" if r.torso_angle is None else f"{r.torso_angle:.0f}"
        label = f"ID {p.track_id} {r.state.value} | goc {ang} | W/H {r.aspect_ratio:.2f}"
    else:
        label = f"ID {p.track_id}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_text = max(y1 - 6, th + 4)
    cv2.rectangle(img, (x1, y_text - th - 4), (x1 + tw + 6, y_text + 3), color, -1)
    cv2.putText(img, label, (x1 + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hud(img: np.ndarray, fps: float, infer_ms: float, n_people: int, alarm: bool) -> None:
    txt = f"FPS {fps:4.1f} | model {infer_ms:5.0f} ms | nguoi {n_people}"
    cv2.rectangle(img, (0, 0), (360, 26), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if alarm:
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), COLORS[State.FALL], 8)
        msg = "CANH BAO: PHAT HIEN TE NGA!"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        x = (w - tw) // 2
        cv2.rectangle(img, (x - 10, h - th - 30), (x + tw + 10, h - 12), COLORS[State.FALL], -1)
        cv2.putText(img, msg, (x, h - 22), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


def render(frame: np.ndarray, people: list[Person], results: list[TrackResult],
           kp_thr: float, fps: float, infer_ms: float) -> np.ndarray:
    out = frame.copy()
    by_id = {r.track_id: r for r in results}
    for p in people:
        draw_person(out, p, by_id.get(p.track_id), kp_thr)
    alarm = any(r.state == State.FALL for r in results)
    draw_hud(out, fps, infer_ms, len(people), alarm)
    return out
