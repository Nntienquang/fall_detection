"""Trích xuất đặc trưng hình học từ keypoints (TV2).

Chỉ số keypoint theo chuẩn COCO-17:
 0 mũi | 1,2 mắt T/P | 3,4 tai T/P | 5,6 vai T/P | 7,8 khuỷu T/P
 9,10 cổ tay T/P | 11,12 hông T/P | 13,14 gối T/P | 15,16 cổ chân T/P
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# Các cặp nối để vẽ skeleton
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]


@dataclass
class PoseFeatures:
    torso_angle: float | None   # độ lệch trục vai→hông so với phương thẳng đứng (0 = đứng, 90 = nằm)
    aspect_ratio: float         # W/H của bounding box
    hip_y: float                # toạ độ y của tâm hông (pixel, lớn hơn = thấp hơn)
    center: tuple[float, float] # tâm bounding box (pixel)
    body_len: float             # thước đo kích thước người (pixel), dùng chuẩn hoá chuyển động
    n_valid: int                # số keypoint đủ tin cậy
    arm_drop: float = 0.0       # cổ tay thấp hơn vai bao nhiêu (theo body_len), chỉ tính tay gần thẳng đứng
    truncated: bool = False     # người bị cắt ở mép ảnh (chạm mép + không thấy cổ chân) → khung xương không đáng tin
    anchor: tuple[float, float] | None = None  # trung điểm vai (không thấy vai thì tâm khung) — dùng đo mức cử động
    legs_upright: bool = False  # hông ở ngay trên cổ chân (chân dựng đứng) — dấu hiệu đang đứng/cúi

    @property
    def valid(self) -> bool:
        return self.body_len > 1.0


def _mid(kps: np.ndarray, conf: np.ndarray, a: int, b: int, thr: float):
    """Trung điểm của 2 keypoint; nếu chỉ thấy 1 bên thì dùng bên đó; không thấy thì None."""
    ok_a, ok_b = conf[a] >= thr, conf[b] >= thr
    if ok_a and ok_b:
        return (kps[a] + kps[b]) / 2.0
    if ok_a:
        return kps[a].copy()
    if ok_b:
        return kps[b].copy()
    return None


def torso_angle(kps: np.ndarray, conf: np.ndarray, thr: float) -> float | None:
    """Góc (độ) giữa trục thân (trung điểm vai → trung điểm hông) và phương thẳng đứng."""
    sh = _mid(kps, conf, L_SHOULDER, R_SHOULDER, thr)
    hp = _mid(kps, conf, L_HIP, R_HIP, thr)
    if sh is None or hp is None:
        return None
    dx, dy = float(hp[0] - sh[0]), float(hp[1] - sh[1])
    if math.hypot(dx, dy) < 1e-6:
        return None
    return math.degrees(math.atan2(abs(dx), abs(dy)))


ARM_MAX_ANGLE = 35.0  # độ: tay lệch khỏi phương thẳng đứng tối đa để coi là "chống" xuống sàn


WRIST_FLOOR = 0.2    # cổ tay cách đáy khung người <= 20% chiều cao khung = đang chạm sàn


def arm_drop(kps: np.ndarray, conf: np.ndarray, thr: float, body_len: float, bbox=None) -> float:
    """Tay chống xuống sàn (chống đẩy, bò, đang chống tay đứng dậy): cổ tay ở ngay dưới vai,
    thấp hơn vai đáng kể VÀ chạm sàn (sát đáy khung người).
      - Người nằm bẹp trên sàn: vai và cổ tay gần cùng độ cao → 0.
      - Người đang rơi, tay buông thõng: cổ tay lơ lửng giữa khung, chưa chạm sàn → 0.
    Trả về độ chênh lớn nhất giữa 2 tay, chuẩn hoá theo body_len (0 = không chống)."""
    best = 0.0
    for sh, wr in ((L_SHOULDER, 9), (R_SHOULDER, 10)):
        if conf[sh] < thr or conf[wr] < thr:
            continue
        dx, dy = float(kps[wr][0] - kps[sh][0]), float(kps[wr][1] - kps[sh][1])
        if dy <= 0:  # cổ tay cao hơn vai (trục y ảnh hướng xuống)
            continue
        if bbox is not None:
            y1, y2 = float(bbox[1]), float(bbox[3])
            if (y2 - float(kps[wr][1])) > WRIST_FLOOR * max(y2 - y1, 1e-6):
                continue  # cổ tay chưa chạm sàn
        if math.degrees(math.atan2(abs(dx), dy)) <= ARM_MAX_ANGLE:
            best = max(best, dy / max(body_len, 1e-6))
    return best


def compute_features(bbox: np.ndarray, kps: np.ndarray, conf: np.ndarray, kp_thr: float) -> PoseFeatures:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)

    hp = _mid(kps, conf, L_HIP, R_HIP, kp_thr)
    hip_y = float(hp[1]) if hp is not None else (y1 + y2) / 2.0

    # Chiều dài cơ thể gần như không đổi khi người đổi tư thế đứng ↔ nằm,
    # nên dùng cạnh dài nhất của bbox làm thước đo chuẩn hoá.
    body_len = max(w, h)

    return PoseFeatures(
        torso_angle=torso_angle(kps, conf, kp_thr),
        aspect_ratio=w / h,
        hip_y=hip_y,
        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        body_len=body_len,
        n_valid=int((conf >= kp_thr).sum()),
        arm_drop=arm_drop(kps, conf, kp_thr, body_len, bbox),
        anchor=_anchor(kps, conf, kp_thr, ((x1 + x2) / 2.0, (y1 + y2) / 2.0)),
        legs_upright=_legs_upright(kps, conf, kp_thr, body_len),
    )


LEG_MAX_ANGLE = 25.0   # độ: trục hông → cổ chân lệch phương thẳng đứng tối đa
LEG_MIN_DROP = 0.35    # cổ chân thấp hơn hông ít nhất 35% chiều dài người


def _legs_upright(kps, conf, thr, body_len) -> bool:
    """Chân dựng đứng: cổ chân ở ngay dưới hông và thấp hơn hông rõ rệt (đứng, cúi người)."""
    hp = _mid(kps, conf, L_HIP, R_HIP, thr)
    an = _mid(kps, conf, L_ANKLE, R_ANKLE, thr)
    if hp is None or an is None:
        return False
    dx, dy = float(an[0] - hp[0]), float(an[1] - hp[1])
    if dy < LEG_MIN_DROP * body_len:
        return False
    return math.degrees(math.atan2(abs(dx), dy)) <= LEG_MAX_ANGLE


def _anchor(kps, conf, thr, fallback):
    sh = _mid(kps, conf, L_SHOULDER, R_SHOULDER, thr)
    return (float(sh[0]), float(sh[1])) if sh is not None else fallback
