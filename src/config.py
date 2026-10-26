"""Đọc cấu hình từ config.yaml thành dataclass (TV1 + TV2 dùng chung)."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    weights: str = "yolo11n-pose.pt"
    imgsz: int = 320
    conf: float = 0.35
    iou: float = 0.5
    tracker: str = "bytetrack.yaml"
    device: str = "cpu"
    threads: int = 4


@dataclass
class DetectorConfig:
    method: str = "C"
    kp_conf: float = 0.35
    angle_thresh: float = 60.0
    ratio_thresh: float = 1.15
    drop_thresh: float = 0.22
    drop_window: float = 1.0
    min_descent_speed: float = 0.6
    descent_memory: float = 1.2
    confirm_time: float = 1.5
    confirm_ratio: float = 0.7
    recover_time: float = 2.0
    alert_cooldown: float = 10.0
    track_ttl: float = 3.0
    reassoc_dist: float = 0.6
    dup_overlap: float = 0.6
    ratio_fallback: bool = True
    still_thresh: float = 0.0      # > 0: chỉ xác nhận ngã khi vai cử động < giá trị này × body_len trong still_window giây (0 = tắt)
    still_window: float = 1.0      # cửa sổ đo mức cử động (giây)
    bend_guard: bool = False       # thân ngang + chân dựng đứng + hông chưa hạ (< bend_hip_drop) → đang cúi, không phải nằm
    bend_hip_drop: float = 0.15    # hông hạ so với lúc đứng gần nhất, theo chiều dài người
    edge_guard: bool = False       # người bị cắt ở mép ảnh, không thấy cổ chân → không kết luận "nằm" (demo webcam: true)
    support_drop: float = 0.0      # B/C: luật chống tay (chống đẩy). Cổ tay thấp hơn vai >= giá trị này × body_len, chạm sàn → không tính là nằm. 0 = tắt (đã thử 26/10, không hiệu quả trên webcam thật)
    support_memory: float = 1.5    # cửa sổ xét chống tay (giây)
    support_ratio: float = 0.3     # pha thấp của chống đẩy: >= 30% khung thân-ngang trong cửa sổ có chống tay → vẫn là chống


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _fill(cls, data: dict | None):
    data = data or {}
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Khoá cấu hình không hợp lệ cho {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path | None = "config.yaml") -> AppConfig:
    """Đọc file YAML; nếu không có file thì dùng giá trị mặc định."""
    if path is None or not Path(path).exists():
        return AppConfig()
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return AppConfig(
        model=_fill(ModelConfig, raw.get("model")),
        detector=_fill(DetectorConfig, raw.get("detector")),
    )


def save_config(cfg: AppConfig, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, allow_unicode=True, sort_keys=False)
