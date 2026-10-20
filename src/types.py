"""Cấu trúc dữ liệu thống nhất giữa các module (chốt từ Ngày 1).

TV1 (pose_tracker) trả về list[Person] cho mỗi frame.
TV2 (features, fall_detector) nhận Person và trả về TrackResult / FallEvent.
TV3 (visualize, app) chỉ đọc các cấu trúc này để vẽ.
TV4 (tools/*) lưu/đọc các cấu trúc này dưới dạng JSON.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


@dataclass
class Person:
    track_id: int            # ID từ ByteTrack (âm nếu chưa được track)
    bbox: np.ndarray         # (4,) x1, y1, x2, y2 — pixel
    score: float             # confidence phát hiện người
    keypoints: np.ndarray    # (17, 2) toạ độ pixel theo chuẩn COCO
    kp_conf: np.ndarray      # (17,) confidence từng keypoint

    def to_json(self) -> dict:
        return {
            "id": int(self.track_id),
            "bbox": [round(float(v), 1) for v in self.bbox],
            "score": round(float(self.score), 3),
            "kps": [[round(float(x), 1), round(float(y), 1)] for x, y in self.keypoints],
            "kpc": [round(float(c), 3) for c in self.kp_conf],
        }

    @classmethod
    def from_json(cls, d: dict) -> "Person":
        return cls(
            track_id=int(d["id"]),
            bbox=np.asarray(d["bbox"], dtype=float),
            score=float(d["score"]),
            keypoints=np.asarray(d["kps"], dtype=float).reshape(17, 2),
            kp_conf=np.asarray(d["kpc"], dtype=float).reshape(17),
        )


class State(str, Enum):
    NORMAL = "NORMAL"
    SUSPECTED = "SUSPECTED"
    FALL = "FALL"


@dataclass
class FallEvent:
    t: float                 # thời điểm (giây) — thời gian video với MP4, đồng hồ thực với webcam
    track_id: int
    kind: str                # "FALL" | "RECOVERED"
    torso_angle: float | None
    aspect_ratio: float
    detail: str = ""


@dataclass
class TrackResult:
    track_id: int
    state: State
    horizontal: bool
    torso_angle: float | None
    aspect_ratio: float
    speed: float             # vận tốc hạ của hông (chiều dài cơ thể / giây), dương = đi xuống
