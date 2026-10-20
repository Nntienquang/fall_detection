"""Chạy TOÀN BỘ thí nghiệm so sánh A / B / C một cách công bằng — 1 lệnh, ra bảng cho báo cáo.

  python tools/run_experiment.py --hold-last 2

Quy trình:
  1. Mỗi phương pháp A, B, C được tìm ngưỡng RIÊNG trên tập val (grid search, tiêu chí F1 → ít FP → trễ thấp).
  2. Mỗi phương pháp dùng ngưỡng của chính nó để chạy trên tập test (chưa từng dùng để chỉnh).
  3. Báo cáo cả 2 chế độ: không giữ frame cuối và giữ frame cuối N giây (--hold-last).

Đầu ra: results/experiment.md, results/experiment_per_video.csv, results/config_A.yaml / _B / _C
Yêu cầu: đã chạy tools/cache_poses.py cho mọi video trong labels.csv.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, save_config  # noqa: E402
from tools.evaluate import METHOD_NAMES, evaluate, load_split  # noqa: E402
from tools.tune import GRID  # noqa: E402


def tune(method, base, labels, caches, hold):
    keys = list(GRID[method])
    best = None
    for vals in itertools.product(*GRID[method].values()):
        det = replace(base, method=method, **dict(zip(keys, vals)))
        tot, _, _ = evaluate(labels, caches, det, 0.5, 3.0, hold)
        lat = tot.mean_latency if tot.mean_latency is not None else 99
        score = (round(tot.f1, 4), -tot.fp, -lat)
        if best is None or score > best[0]:
            best = (score, det, tot, {k: getattr(det, k) for k in keys})
    return best[1], best[2], best[3]


def row(name, tot, hours):
    lat = tot.mean_latency
    return (f"| {name} | {tot.tp} | {tot.fp} | {tot.fn} | {tot.precision:.2f} | {tot.recall:.2f} | "
            f"**{tot.f1:.2f}** | {'' if lat is None else f'{lat:.2f}'} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--tune-split", default="val")
    ap.add_argument("--eval-splits", nargs="+", default=["test"])
    ap.add_argument("--hold-last", type=float, default=2.0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    base = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    val_l, val_c = load_split(args.labels, args.cache, args.tune_split)
    if not val_l:
        raise SystemExit(f"Không có video nào ở split '{args.tune_split}'.")
    evals = {s: load_split(args.labels, args.cache, s) for s in args.eval_splits}

    md = ["# Kết quả thí nghiệm so sánh A / B / C", "",
          f"Ngưỡng của từng phương pháp được chọn riêng trên tập `{args.tune_split}` "
          f"({len(val_l)} video, {sum(len(v['falls']) for v in val_l.values())} cú ngã), "
          f"rồi giữ nguyên khi chạy trên tập đánh giá. Một cú ngã tính là phát hiện đúng nếu có cảnh báo "
          f"trong khoảng 0,5 s trước đến 3 s sau nhãn.", ""]
    pv_rows = []
    for hold in sorted({0.0, args.hold_last}):
        tuned = {}
        for m in "ABC":
            det, tot_val, params = tune(m, base.detector, val_l, val_c, hold)
            tuned[m] = (det, tot_val, params)
            if hold == args.hold_last:
                cfg = replace(base)
                cfg.detector = det
                save_config(cfg, out / f"config_{m}.yaml")
        mode = "không giữ frame cuối" if hold == 0 else f"giữ frame cuối {hold:g} s"
        md += [f"## Chế độ: {mode}", ""]
        md += ["Ngưỡng chọn được trên val:", ""]
        md += [f"- **{m}**: " + ", ".join(f"{k} = {v}" for k, v in tuned[m][2].items()) +
               f" (F1 trên val = {tuned[m][1].f1:.2f})" for m in "ABC"]
        md.append("")
        for split, (lab, cac) in evals.items():
            if not lab:
                continue
            n_f = sum(len(v["falls"]) for v in lab.values())
            md += [f"### Tập `{split}`: {len(lab)} video, {n_f} cú ngã, {len(lab) - sum(bool(v['falls']) for v in lab.values())} video không ngã", "",
                   "| Phương pháp | TP | FP | FN | Precision | Recall | F1 | Độ trễ TB (s) |", "|---|---|---|---|---|---|---|---|"]
            for m in "ABC":
                tot, per_video, hours = evaluate(lab, cac, tuned[m][0], 0.5, 3.0, hold)
                md.append(row(METHOD_NAMES[m], tot, hours))
                pv_rows += [{"hold_last": hold, "split": split, "method": m, **r} for r in per_video]
            md.append("")
    md += ["## Ghi chú cho báo cáo", "",
           "- TP: cú ngã phát hiện đúng · FP: cảnh báo nhầm · FN: cú ngã bị bỏ sót.",
           "- Clip UR Fall bị cắt ngay sau khi chạm sàn (nhiều clip còn < 1 s). Chế độ \"giữ frame cuối\" "
           "lặp lại khung hình cuối để mô phỏng người vẫn nằm; áp dụng như nhau cho mọi video và phương pháp.",
           "- Không báo \"cảnh báo sai/giờ\" vì tổng thời lượng video chỉ vài phút, con số này không có ý nghĩa thống kê.",
           "- Số video nhỏ: chênh lệch 1 video đổi F1 khoảng 0,05–0,1. Nên trình bày kèm phân tích từng ca sai "
           "(xem experiment_per_video.csv)."]
    (out / "experiment.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(out / "experiment_per_video.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pv_rows[0]))
        w.writeheader()
        w.writerows(pv_rows)
    print("\n".join(md))
    print(f"\nĐã lưu {out / 'experiment.md'}, {out / 'experiment_per_video.csv'}, config_A/B/C.yaml")


if __name__ == "__main__":
    main()
