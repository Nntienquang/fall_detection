"""So sánh 3 phương pháp A/B/C trên tập test (TV4).

  python tools/evaluate.py --labels data/labels.csv --split test
  python tools/evaluate.py --config config.tuned.yaml --split test --methods A B C

Đầu ra: results/metrics.csv, results/per_video.csv, results/metrics.md (dán vào báo cáo).
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.metrics import MatchResult, match_events  # noqa: E402
from src.replay import cache_duration, cache_path_for, load_cache, read_labels, run_detector  # noqa: E402

METHOD_NAMES = {"A": "A — Bounding box (W/H)", "B": "B — Pose tĩnh (góc thân)", "C": "C — Pose + thời gian"}


def evaluate(labels: dict, caches: dict, det_cfg, pre_tol: float, post_tol: float, hold_last: float = 0.0):
    total = MatchResult()
    per_video = []
    hours = 0.0
    for video, lab in labels.items():
        cache = caches[video]
        alerts = [e.t for e in run_detector(cache, det_cfg, hold_last) if e.kind == "FALL"]
        m = match_events(alerts, lab["falls"], pre_tol, post_tol)
        total += m
        hours += cache_duration(cache, hold_last) / 3600
        per_video.append({"video": video, "falls": len(lab["falls"]), "alerts": len(alerts),
                          "tp": m.tp, "fp": m.fp, "fn": m.fn,
                          "alert_times": " ".join(f"{a:.1f}" for a in alerts)})
    return total, per_video, hours


def load_split(labels_path, cache_dir, split):
    labels = read_labels(labels_path)
    if split != "all":
        labels = {k: v for k, v in labels.items() if v["split"] == split}
    caches, missing = {}, []
    for v in labels:
        p = cache_path_for(cache_dir, v)
        if p.exists():
            caches[v] = load_cache(p)
        else:
            missing.append(v)
    if missing:
        print(f"[CẢNH BÁO] Thiếu cache cho {len(missing)} video, chạy tools/cache_poses.py trước: {missing}")
        labels = {k: v for k, v in labels.items() if k in caches}
    return labels, caches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", default="test", help="train | val | test | all")
    ap.add_argument("--methods", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--pre-tol", type=float, default=0.5)
    ap.add_argument("--post-tol", type=float, default=3.0)
    ap.add_argument("--hold-last", type=float, default=0.0,
                    help="giữ frame cuối thêm N giây (UR Fall: 2). Ghi rõ trong báo cáo")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    labels, caches = load_split(args.labels, args.cache, args.split)
    if not labels:
        print("Không có video nào để đánh giá.")
        return
    n_falls = sum(len(v["falls"]) for v in labels.values())
    infer = statistics.median(c["infer_ms_median"] for c in caches.values())
    speed = f"YOLO trung vị {infer:.0f} ms/frame (~{1000 / infer:.1f} FPS)" if infer > 0 else "dữ liệu giả lập, không có số đo YOLO"
    print(f"Tập '{args.split}': {len(labels)} video, {n_falls} cú ngã | {speed}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, pv_rows = [], []
    for m in args.methods:
        total, per_video, hours = evaluate(labels, caches, replace(cfg.detector, method=m),
                                           args.pre_tol, args.post_tol, args.hold_last)
        lat = total.mean_latency
        rows.append({
            "method": METHOD_NAMES[m], "TP": total.tp, "FP": total.fp, "FN": total.fn,
            "precision": round(total.precision, 3), "recall": round(total.recall, 3), "f1": round(total.f1, 3),
            "false_alarms_per_hour": round(total.fp / hours, 2) if hours else "",
            "mean_latency_s": "" if lat is None else round(lat, 2),
        })
        pv_rows += [{"method": m, **r} for r in per_video]

    with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "per_video.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pv_rows[0]))
        w.writeheader()
        w.writerows(pv_rows)

    md = ["| Phương pháp | TP | FP | FN | Precision | Recall | F1 | Cảnh báo sai/giờ | Độ trễ TB (s) |",
          "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['method']} | {r['TP']} | {r['FP']} | {r['FN']} | {r['precision']} | {r['recall']} | "
           f"{r['f1']} | {r['false_alarms_per_hour']} | {r['mean_latency_s']} |" for r in rows]
    md.append(f"\nTập `{args.split}`: {len(labels)} video, {n_falls} cú ngã. "
              f"Model: {cfg.model.weights}, imgsz {cfg.model.imgsz}, {speed}.")
    (out / "metrics.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nĐã lưu: {out / 'metrics.csv'}, {out / 'per_video.csv'}, {out / 'metrics.md'}")


if __name__ == "__main__":
    main()
