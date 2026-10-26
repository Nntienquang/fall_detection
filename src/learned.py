"""Phương pháp D — học máy: mạng nơ-ron nhỏ (MLP) quyết định một khung hình là "đang nằm sau ngã".

Thay cho ngưỡng góc thân của B, D học từ dữ liệu (GMDCSA24 + UR Fall) cách kết hợp nhiều đặc trưng:
  - tư thế 17 khớp, chuẩn hoá theo khung người                      (51 số)
  - hình học: góc thân, W/H, độ co ngắn thân/chân so với bề rộng vai (6 số)
  - theo thời gian: hông hạ bao nhiêu và khung người thấp đi bao nhiêu
    so với lần đứng thẳng gần nhất, tốc độ hạ, mức cử động           (5 số)
Mô hình huấn luyện bằng scikit-learn (tools/train_fall_model.py) rồi xuất trọng số ra .npz,
nên lúc chạy chỉ cần numpy.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .features import _mid, torso_angle

L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = 5, 6, 11, 12, 15, 16
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]   # đổi trái ↔ phải khi lật ngang
FEATURE_NAMES = (
    [f"k{i}_{a}" for i in range(17) for a in ("x", "y")] + [f"v{i}" for i in range(17)]
    + ["angle", "angle_missing", "aspect", "torso_sw", "leg_sw", "legs_upright"]
    + ["hip_drop", "height_ratio", "speed", "motion", "has_ref"]
)
N_FEATURES = len(FEATURE_NAMES)   # 62


def static_features(bbox, kps, conf, thr: float, legs_upright: bool, flip: bool = False) -> np.ndarray:
    """Đặc trưng của một khung hình (không cần lịch sử)."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    cx, cy, L = (x1 + x2) / 2, (y1 + y2) / 2, max(w, h)
    vis = conf >= thr
    rel = np.zeros((17, 2))
    rel[vis, 0] = (kps[vis, 0] - cx) / L
    rel[vis, 1] = (kps[vis, 1] - cy) / L
    v = vis.astype(float)
    if flip:
        rel, v = rel[FLIP].copy(), v[FLIP].copy()
        rel[:, 0] = -rel[:, 0]
    ang = torso_angle(kps, conf, thr)
    torso_sw = leg_sw = 0.0
    if conf[L_SH] >= thr and conf[R_SH] >= thr:
        sw = float(np.linalg.norm(kps[L_SH] - kps[R_SH]))
        sh, hp, an = _mid(kps, conf, L_SH, R_SH, thr), _mid(kps, conf, L_HIP, R_HIP, thr), _mid(kps, conf, L_ANK, R_ANK, thr)
        if sw > 1.0 and hp is not None:
            torso_sw = min(float(np.linalg.norm(hp - sh)) / sw, 5.0)
            if an is not None:
                leg_sw = min(float(np.linalg.norm(an - hp)) / sw, 5.0)
    geo = [0.0 if ang is None else ang / 90.0, float(ang is None), min(w / h, 4.0), torso_sw, leg_sw, float(legs_upright)]
    return np.concatenate([rel.ravel(), v, geo])


def temporal_features(hip_drop: float | None, height_ratio: float | None, speed: float, motion: float) -> np.ndarray:
    has = hip_drop is not None
    return np.array([
        float(np.clip(hip_drop, -1.0, 2.0)) if has else 0.0,
        float(np.clip(height_ratio, 0.0, 3.0)) if has else 1.0,
        float(np.clip(speed, 0.0, 3.0)),
        2.0 if not math.isfinite(motion) else float(np.clip(motion, 0.0, 2.0)),
        float(has),
    ])


class LearnedModel:
    """Chạy MLP đã xuất từ scikit-learn: chuẩn hoá → các lớp ẩn ReLU → sigmoid."""

    def __init__(self, path: str | Path):
        d = np.load(path, allow_pickle=False)
        self.mean, self.scale = d["mean"], d["scale"]
        n = int(d["n_layers"])
        self.W = [d[f"W{i}"] for i in range(n)]
        self.b = [d[f"b{i}"] for i in range(n)]
        if self.mean.shape[0] != N_FEATURES:
            raise ValueError(f"Mô hình có {self.mean.shape[0]} đặc trưng, code cần {N_FEATURES}: hãy huấn luyện lại")

    def proba(self, x: np.ndarray) -> float:
        a = (x - self.mean) / self.scale
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            a = a @ W + b
            if i < len(self.W) - 1:
                a = np.maximum(a, 0.0)
        return float(1.0 / (1.0 + np.exp(-a[0])))


def export_sklearn(mlp, scaler, path: str | Path) -> None:
    arrays = {"mean": scaler.mean_, "scale": scaler.scale_, "n_layers": np.array(len(mlp.coefs_))}
    for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        arrays[f"W{i}"], arrays[f"b{i}"] = W, b
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
