"""Tìm bộ ngưỡng tốt nhất trên tập VALIDATION (grid search), lưu config.tuned.yaml.

  python tools/tune.py --method C --split val
Sau đó đánh giá CUỐI CÙNG trên tập test:
  python tools/evaluate.py --config config.tuned.yaml --split test
"""
from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, save_config  # noqa: E402
from tools.evaluate import evaluate, load_split  # noqa: E402

GRID = {
    "A": {"ratio_thresh": [0.9, 1.0, 1.15, 1.3, 1.5], "confirm_time": [0.5, 1.0, 1.5, 2.0]},
    "B": {"angle_thresh": [45, 50, 55, 60, 65, 70], "ratio_thresh": [1.0, 1.15, 1.3],
          "confirm_time": [0.5, 1.0, 1.5, 2.0]},
    "C": {"angle_thresh": [50, 55, 60, 65, 70], "drop_thresh": [0.12, 0.17, 0.22, 0.28],
          "min_descent_speed": [0.4, 0.6, 0.8], "confirm_time": [1.0, 1.5, 2.0]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--method", default="C", choices=["A", "B", "C"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--hold-last", type=float, default=0.0)
    ap.add_argument("--out", default="config.tuned.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    labels, caches = load_split(args.labels, args.cache, args.split)
    if not labels:
        print("Không có video validation (cột split = val).")
        return

    grid = GRID[args.method]
    keys = list(grid)
    best = None
    combos = list(itertools.product(*grid.values()))
    print(f"Thử {len(combos)} tổ hợp cho phương pháp {args.method} trên {len(labels)} video '{args.split}'…")
    for vals in combos:
        det = replace(cfg.detector, method=args.method, **dict(zip(keys, vals)))
        total, _, _ = evaluate(labels, caches, det, 0.5, 3.0, args.hold_last)
        lat = total.mean_latency if total.mean_latency is not None else 99
        score = (round(total.f1, 4), -total.fp, -lat)   # F1 cao → ít FP → trễ thấp
        if best is None or score > best[0]:
            best = (score, det, total)

    (_, det, total) = best
    print("Bộ ngưỡng tốt nhất:", {k: getattr(det, k) for k in keys})
    print(f"  val: P={total.precision:.3f} R={total.recall:.3f} F1={total.f1:.3f} TP={total.tp} FP={total.fp} FN={total.fn}")
    cfg.detector = det
    save_config(cfg, args.out)
    print(f"Đã lưu {args.out}. Báo cáo kết quả trên tập TEST, không dùng số liệu val.")


if __name__ == "__main__":
    main()
