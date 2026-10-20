"""Lưu lịch sử sự kiện: CSV + ảnh chụp + video kết quả, mỗi lần chạy 1 thư mục riêng."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .types import FallEvent


class EventLogger:
    def __init__(self, root: str | Path = "outputs", run_name: str | None = None, meta: dict | None = None):
        name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.dir = Path(root) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / "events.csv"
        with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(
                ["wall_time", "t_sec", "track_id", "event", "torso_angle", "aspect_ratio", "snapshot", "detail"])
        if meta:
            (self.dir / "run_config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
        self._writer = None
        self.events: list[FallEvent] = []

    def log(self, ev: FallEvent, frame: np.ndarray | None = None) -> None:
        snap = ""
        if frame is not None and ev.kind == "FALL":
            snap = f"fall_id{ev.track_id}_{ev.t:08.2f}s.jpg"
            cv2.imwrite(str(self.dir / snap), frame)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([
                datetime.now().isoformat(timespec="seconds"), f"{ev.t:.2f}", ev.track_id, ev.kind,
                "" if ev.torso_angle is None else f"{ev.torso_angle:.1f}", f"{ev.aspect_ratio:.2f}", snap, ev.detail,
            ])
        self.events.append(ev)

    def write_frame(self, frame: np.ndarray, fps: float) -> None:
        if self._writer is None:
            h, w = frame.shape[:2]
            self._writer = cv2.VideoWriter(str(self.dir / "annotated.mp4"),
                                           cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
