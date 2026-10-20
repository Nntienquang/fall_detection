"""Vẽ biểu đồ đặc trưng theo thời gian của 1 video (dùng cho Ngày 3 và báo cáo).

  python tools/plot_features.py --video V05_fall.mp4
Đầu ra: results/features_<video>.csv và results/features_<video>.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import load_config  # noqa: E402
from src.fall_detector import FallDetector  # noqa: E402
from src.replay import cache_path_for, iter_frames, load_cache, read_labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cache = load_cache(cache_path_for(args.cache, args.video))
    det = FallDetector(cfg.detector)
    rows, alerts = [], []
    for t, people in iter_frames(cache):
        res, evs = det.update(people, t)
        alerts += [e.t for e in evs if e.kind == "FALL"]
        for r in res:
            rows.append({"t": t, "id": r.track_id, "angle": r.torso_angle, "ratio": r.aspect_ratio,
                         "speed": r.speed, "state": r.state.value})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem
    with open(out / f"features_{stem}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["t", "id", "angle", "ratio", "speed", "state"])
        w.writeheader()
        w.writerows(rows)

    falls = []
    if Path(args.labels).exists():
        falls = read_labels(args.labels).get(args.video, {}).get("falls", [])

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    series = [("angle", "Góc thân (độ)", cfg.detector.angle_thresh),
              ("ratio", "Tỷ lệ W/H", cfg.detector.ratio_thresh),
              ("speed", "Vận tốc hạ hông (thân/giây)", cfg.detector.min_descent_speed)]
    for ax, (key, title, thr) in zip(axes, series):
        for tid in sorted({r["id"] for r in rows}):
            pts = [(r["t"], r[key]) for r in rows if r["id"] == tid and r[key] is not None]
            if pts:
                ax.plot(*zip(*pts), lw=1.4, label=f"ID {tid}")
        ax.axhline(thr, ls="--", lw=1, color="gray")
        for s, e in falls:
            ax.axvspan(s, e, color="tab:red", alpha=0.12)
        for a in alerts:
            ax.axvline(a, color="tab:red", lw=1.2)
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_title(f"{args.video} — vùng đỏ nhạt: nhãn té ngã · vạch đỏ: cảnh báo của hệ thống")
    axes[-1].set_xlabel("Thời gian (giây)")
    fig.tight_layout()
    fig.savefig(out / f"features_{stem}.png", dpi=130)
    print(f"Đã lưu {out / f'features_{stem}.csv'} và {out / f'features_{stem}.png'}")


if __name__ == "__main__":
    main()
