"""Lưu / đọc kết quả YOLO Pose của cả video (cache) và chạy lại thuật toán trên cache.

Mục đích: YOLO trên CPU chậm, nhưng thuật toán té ngã rất nhanh. Chỉ cần chạy YOLO
MỘT lần cho mỗi video, sau đó có thể thử hàng trăm bộ ngưỡng trong vài giây.
"""
from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

from .config import DetectorConfig
from .fall_detector import FallDetector
from .types import FallEvent, Person


def save_cache(path: str | Path, video: str, fps: float, frames: list[dict], infer_ms_median: float,
               frame_shape: tuple[int, int] | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {"video": video, "fps": fps, "n_frames": len(frames),
            "infer_ms_median": infer_ms_median, "frames": frames}
    if frame_shape:
        data["frame_shape"] = [int(frame_shape[0]), int(frame_shape[1])]  # (cao, rộng)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(data, fh)


def load_cache(path: str | Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def iter_frames(cache: dict):
    for fr in cache["frames"]:
        yield fr["t"], [Person.from_json(p) for p in fr["people"]]


def run_detector(cache: dict, cfg: DetectorConfig, hold_last: float = 0.0) -> list[FallEvent]:
    """hold_last > 0: lặp lại frame cuối thêm hold_last giây (giả định người giữ nguyên tư thế
    cuối). Dùng cho bộ dữ liệu có clip bị cắt ngay sau khi chạm sàn (UR Fall). Phải ghi rõ
    trong báo cáo, và áp dụng như nhau cho mọi video, mọi phương pháp."""
    det = FallDetector(cfg)
    events: list[FallEvent] = []
    t, people = 0.0, []
    for t, people in iter_frames(cache):
        _, evs = det.update(people, t)
        events.extend(evs)
    if hold_last > 0 and cache.get("fps"):
        dt = 1.0 / cache["fps"]
        for k in range(1, int(round(hold_last / dt)) + 1):
            _, evs = det.update(people, round(t + k * dt, 4))
            events.extend(evs)
    return events


def cache_duration(cache: dict, hold_last: float = 0.0) -> float:
    return (cache["n_frames"] / cache["fps"] + hold_last) if cache["fps"] else 0.0


def read_labels(path: str | Path) -> dict[str, dict]:
    """labels.csv: video,split,fall_start,fall_end,note — mỗi dòng 1 cú ngã;
    video không có cú ngã: để trống fall_start/fall_end."""
    out: dict[str, dict] = defaultdict(lambda: {"split": "test", "falls": [], "note": ""})
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            v = row["video"].strip()
            if not v or v.startswith("#"):
                continue
            item = out[v]
            item["split"] = (row.get("split") or "test").strip()
            item["note"] = (row.get("note") or item["note"]).strip()
            s, e = (row.get("fall_start") or "").strip(), (row.get("fall_end") or "").strip()
            if s and e:
                item["falls"].append((float(s), float(e)))
    return dict(out)


def cache_path_for(cache_dir: str | Path, video: str) -> Path:
    """ur_fall-01.mp4 → ur_fall-01.json.gz (giữ tên cũ);
    "Subject 1/Fall/01.mp4" → Subject_1_Fall_01.json.gz (video nằm trong thư mục con, tránh trùng tên)."""
    p = Path(video)
    if len(p.parts) == 1:
        return Path(cache_dir) / (p.stem + ".json.gz")
    name = "_".join(p.with_suffix("").parts).replace(" ", "_")
    return Path(cache_dir) / (name + ".json.gz")
