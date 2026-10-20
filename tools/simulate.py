"""Tạo bộ dữ liệu GIẢ LẬP (cache + nhãn) để chạy thử tune/evaluate khi chưa có video thật.

  python tools/simulate.py                     # → cache_sim/, data/labels_sim.csv
  python tools/tune.py --labels data/labels_sim.csv --cache cache_sim --split val
  python tools/evaluate.py --labels data/labels_sim.csv --cache cache_sim --split test --out results_sim
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.replay import save_cache  # noqa: E402
from src.simulate import SCENARIOS, scenario  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_sim")
    ap.add_argument("--labels", default="data/labels_sim.csv")
    ap.add_argument("--per-split", type=int, default=3, help="số biến thể mỗi tình huống cho mỗi split")
    args = ap.parse_args()

    Path(args.labels).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    seed = 0
    for split in ("val", "test"):
        for name in SCENARIOS:
            for k in range(args.per_split):
                seed += 1
                fall_dur = 0.45 + 0.1 * (seed % 4)
                frames, falls = scenario(name, seed=seed, fall_dur=fall_dur)
                video = f"sim_{split}_{name}_{k}.mp4"
                save_cache(Path(args.cache) / f"sim_{split}_{name}_{k}.json.gz", video, 15.0,
                           [{"t": t, "people": [p.to_json() for p in ps]} for t, ps in frames], 0.0)
                if falls:
                    rows += [{"video": video, "split": split, "fall_start": s, "fall_end": e, "note": name}
                             for s, e in falls]
                else:
                    rows.append({"video": video, "split": split, "fall_start": "", "fall_end": "", "note": name})
    with open(args.labels, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["video", "split", "fall_start", "fall_end", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"Đã tạo {len({r['video'] for r in rows})} video giả lập → {args.cache}/, nhãn: {args.labels}")


if __name__ == "__main__":
    main()
