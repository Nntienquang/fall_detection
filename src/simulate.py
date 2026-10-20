"""Sinh chuỗi keypoints GIẢ LẬP (người que) cho các tình huống: đi, ngồi, cúi,
chủ động nằm, té ngã, té rồi đứng dậy, 2 người.

Dùng để: (1) unit test thuật toán không cần camera/YOLO, (2) chạy thử toàn bộ
quy trình cache → tune → evaluate trước khi nhóm có video thật.
KHÔNG dùng số liệu trên dữ liệu giả lập làm kết quả trong báo cáo.
"""
from __future__ import annotations

import math

import numpy as np

from .types import Person


def _rot(v: np.ndarray, r: float) -> np.ndarray:
    c, s = math.cos(r), math.sin(r)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def skeleton(cx: float, hip_y: float, L: float, angle_deg: float, sitting: float = 0.0,
             facing: int = 1, gait: float | None = None) -> np.ndarray:
    """17 keypoints COCO cho người dài L pixel, thân nghiêng angle_deg (0 = đứng, 90 = nằm).
    sitting ∈ [0, 1]: mức gập chân (đùi nằm ngang khi = 1).
    gait: pha bước chân (radian) khi đi bộ — chân và tay đánh ngược nhau."""
    a = math.radians(angle_deg)
    up = np.array([facing * math.sin(a), -math.cos(a)])          # hông → vai
    perp = np.array([math.cos(a), facing * math.sin(a)]) * 0.08 * L
    hip = np.array([cx, hip_y])
    sh = hip + up * 0.30 * L
    head = hip + up * 0.42 * L
    # chân: đứng thì thẳng xuống theo -up; ngồi thì đùi xoay về phía trước
    thigh_dir = (1 - sitting) * (-up) + sitting * np.array([facing * 1.0, 0.0])
    thigh_dir /= np.linalg.norm(thigh_dir) + 1e-9
    knee = hip + thigh_dir * 0.25 * L
    shin_dir = (1 - sitting) * (-up) + sitting * np.array([0.0, 1.0])
    shin_dir /= np.linalg.norm(shin_dir) + 1e-9
    ankle = knee + shin_dir * 0.25 * L
    elbow_dir = -up * 0.6 + np.array([facing * 0.2, 0])
    k = np.zeros((17, 2))
    k[0] = head
    k[1], k[2] = head - perp * 0.3, head + perp * 0.3
    k[3], k[4] = head - perp * 0.6, head + perp * 0.6
    k[5], k[6] = sh - perp, sh + perp
    k[7], k[8] = k[5] + elbow_dir * 0.15 * L, k[6] + elbow_dir * 0.15 * L
    k[9], k[10] = k[7] + elbow_dir * 0.13 * L, k[8] + elbow_dir * 0.13 * L
    k[11], k[12] = hip - perp * 0.7, hip + perp * 0.7
    k[13], k[14] = knee - perp * 0.7, knee + perp * 0.7
    k[15], k[16] = ankle - perp * 0.7, ankle + perp * 0.7
    if gait is not None:
        sw = 0.38 * math.sin(gait)
        for side, sgn in ((0, 1), (1, -1)):
            h = k[11 + side]
            th = _rot(-up, sgn * sw)
            kn = h + th * 0.25 * L
            sh_dir = _rot(-up, sgn * sw - 0.25 * max(0.0, -sgn * math.cos(gait)))
            k[13 + side] = kn
            k[15 + side] = kn + sh_dir * 0.25 * L
            arm = _rot(-up, -sgn * sw * 0.8)
            k[7 + side] = k[5 + side] + arm * 0.15 * L
            k[9 + side] = k[7 + side] + _rot(arm, -facing * 0.3) * 0.13 * L
    return k


def to_person(tid: int, kps: np.ndarray, rng: np.random.Generator | None = None, noise: float = 0.0,
              L: float = 200.0) -> Person:
    if rng is not None and noise > 0:
        kps = kps + rng.normal(0, noise * L, kps.shape)
    x1, y1 = kps.min(0) - 0.04 * L
    x2, y2 = kps.max(0) + 0.04 * L
    return Person(tid, np.array([x1, y1, x2, y2]), 0.9, kps, np.full(17, 0.9))


def _lerp(a, b, u):
    u = min(max(u, 0.0), 1.0)
    return a + (b - a) * u


def scenario(name: str, fps: float = 15.0, seed: int = 0, noise: float = 0.01,
             fall_dur: float = 0.6, L: float = 200.0, clean_out: list | None = None):
    """Trả về (frames, falls): frames = list[(t, list[Person])], falls = list[(start, end)].
    clean_out: nếu truyền list vào, mỗi frame được thêm list keypoints KHÔNG nhiễu (để vẽ cảnh)."""
    rng = np.random.default_rng(seed)
    ground = 420.0
    stand_hip = ground - 0.5 * L
    t0 = 2.0 + rng.uniform(0, 1.0)           # thời điểm bắt đầu hành động
    dur = 12.0
    frames, falls = [], []
    n = int(dur * fps)

    def pose_at(t, kind):
        """(cx, hip_y, angle, sitting) theo thời gian."""
        cx = 200 + 25 * t
        if kind == "walk":
            return cx, stand_hip + 3 * math.sin(6 * t), 3 * math.sin(3 * t), 0.0
        if kind == "sit":
            u = (t - t0) / 1.2
            return 260, _lerp(stand_hip, stand_hip + 0.25 * L, u), _lerp(0, 12, u), _lerp(0, 1, u)
        if kind == "bend":
            u = 1 - abs((t - t0) - 1.5) / 1.5 if t0 <= t <= t0 + 3 else 0.0
            return 260, stand_hip + 0.05 * L * u, 78 * min(u * 1.6, 1), 0.15 * u
        if kind == "lie_down":
            u = (t - t0) / 4.0
            return 260, _lerp(stand_hip, ground - 0.06 * L, u), _lerp(0, 88, u), 0.5 * math.sin(math.pi * min(max(u, 0), 1))
        if kind in ("fall", "fall_getup"):
            u = min(max(t - t0, 0.0) / fall_dur, 1.0) ** 2   # tăng tốc như rơi tự do
            pose = (260 + 0.3 * L * min(max(u, 0), 1), _lerp(stand_hip, ground - 0.06 * L, u), _lerp(0, 86, u), 0.0)
            if kind == "fall_getup":
                g = (t - (t0 + fall_dur + 3.5)) / 2.0  # nằm 3.5s rồi đứng dậy trong 2s
                if g > 0:
                    pose = (pose[0], _lerp(ground - 0.06 * L, stand_hip, g), _lerp(86, 0, g), 0.6 * math.sin(math.pi * min(g, 1)))
            return pose
        raise ValueError(kind)

    kinds = {"two_people": ["walk", "fall"]}.get(name, [name])
    for i in range(n):
        t = i / fps
        people, clean = [], []
        for tid, kind in enumerate(kinds, start=1):
            cx, hy, ang, sit = pose_at(t, kind)
            facing, gait = 1, (7.5 * t if kind == "walk" else None)
            if kind == "walk" and len(kinds) > 1:
                cx, facing = 520 - 20 * t, -1
            kps = skeleton(cx, hy, L, ang, sit, facing, gait)
            clean.append(kps)
            people.append(to_person(tid, kps, rng, noise, L))
        if clean_out is not None:
            clean_out.append(clean)
        frames.append((round(t, 4), people))
    if name in ("fall", "fall_getup", "two_people"):
        falls.append((round(t0, 2), round(t0 + fall_dur + 2.0, 2)))
    return frames, falls


SCENARIOS = ["walk", "sit", "bend", "lie_down", "fall", "fall_getup", "two_people"]
