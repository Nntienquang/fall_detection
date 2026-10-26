"""Huấn luyện phương pháp D (mạng nơ-ron nhỏ trên đặc trưng khung xương) và so sánh với B.

  python tools/train_fall_model.py \
      --train cache_gmd:data/labels_gmd.csv:train_gmd  cache_416:data/labels.csv:val,test_cu \
      --test  cache_gmd:data/labels_gmd.csv:test_gmd   cache_416:data/labels.csv:test

Quy trình:
 1. Chạy lại hệ thống trên cache (kết quả YOLO đã lưu) để lấy vectơ đặc trưng của người chính ở mỗi khung.
 2. Gán nhãn từng khung theo nhãn video: "đang nằm sau ngã" = từ giữa đoạn ngã (hoặc 1 s sau khi bắt đầu)
    đến hết đoạn ngã + 1 s; khung đang rơi (chuyển tiếp) bỏ qua; mọi khung khác = không nằm.
 3. Kiểm tra chéo theo NGƯỜI/BỘ DỮ LIỆU (GroupKFold) trên tập train để ước lượng độ tin cậy.
 4. Huấn luyện mô hình cuối trên toàn bộ train (kèm ảnh lật ngang), xuất models/fall_mlp.npz.
 5. Đánh giá theo SỰ KIỆN trên tập test (chưa dùng để huấn luyện): B so với D, cùng các luật webcam.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.fall_detector import FallDetector  # noqa: E402
from src.learned import FLIP, N_FEATURES, export_sklearn  # noqa: E402
from src.metrics import MatchResult, match_events  # noqa: E402
from src.replay import cache_path_for, iter_frames, load_cache, read_labels  # noqa: E402

DEFAULT_SHAPE = {"ur_": (240, 320)}   # cache UR cũ không lưu kích thước khung


def parse_sets(specs):
    out = []
    for s in specs:
        cache, labels, splits = s.split(":")
        out.append((cache, labels, splits.split(",")))
    return out


def videos_of(sets):
    for cache, labels, splits in sets:
        for v, lab in read_labels(labels).items():
            if lab["split"] in splits:
                p = cache_path_for(cache, v)
                if p.exists():
                    yield v, lab, p
                else:
                    print(f"  thiếu cache: {p}")


def group_of(video: str) -> str:
    return f"gmd{video.split('/')[0].split()[-1]}" if video.startswith("Subject") else "ur"


def frame_shape(cache: dict, video: str):
    if cache.get("frame_shape"):
        return tuple(cache["frame_shape"])
    return next((s for k, s in DEFAULT_SHAPE.items() if Path(video).name.startswith(k)), None)


def frame_label(t: float, falls) -> int | None:
    for s, e in falls:
        on = s + min(1.0, 0.5 * (e - s))
        if on <= t <= e + 1.0:
            return 1
        if s - 0.5 <= t < on:
            return None        # đang rơi: chuyển tiếp, không dùng để học
    return 0


def collect(sets, base):
    X, y, g, vids = [], [], [], []
    for v, lab, p in videos_of(sets):
        cache = load_cache(p)
        det = FallDetector(replace(base, method="B"))
        det.collect = []
        shape = frame_shape(cache, v)
        for t, people in iter_frames(cache):
            det.update(people, t, shape)
        main = {}
        for tid, t, L, x in det.collect:          # người chính = khung lớn nhất ở mỗi thời điểm
            if t not in main or L > main[t][0]:
                main[t] = (L, x)
        for t, (_, x) in main.items():
            lab_t = frame_label(t, lab["falls"])
            if lab_t is not None:
                X.append(x); y.append(lab_t); g.append(group_of(v)); vids.append(v)
    return np.array(X), np.array(y), np.array(g), np.array(vids)


def flip_rows(X):
    F = X.copy()
    k = X[:, :34].reshape(-1, 17, 2)[:, FLIP].copy()
    k[:, :, 0] *= -1
    F[:, :34] = k.reshape(-1, 34)
    F[:, 34:51] = X[:, 34:51][:, FLIP]
    return F


def make_model(seed=0):
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=400,
                         early_stopping=True, validation_fraction=0.15, random_state=seed)


def fit(X, y, seed=0):
    from sklearn.preprocessing import StandardScaler
    Xa, ya = np.vstack([X, flip_rows(X)]), np.concatenate([y, y])
    scaler = StandardScaler().fit(Xa)
    scaler.scale_[scaler.scale_ < 1e-6] = 1.0
    mlp = make_model(seed).fit(scaler.transform(Xa), ya)
    return scaler, mlp


def frame_metrics(y, p, thr=0.5):
    from sklearn.metrics import f1_score, roc_auc_score
    auc = roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")
    return auc, f1_score(y, p >= thr)


def run_events(det_cfg, cache, video, hold):
    det = FallDetector(det_cfg)
    shape = frame_shape(cache, video)
    ev, t, last = [], 0.0, []
    for t, people in iter_frames(cache):
        ev += det.update(people, t, shape)[1]
        if people:
            last = people
    dt = 1.0 / (cache.get("fps") or 30)
    for k in range(1, int(round(hold / dt)) + 1):
        ev += det.update(last, round(t + k * dt, 4), shape)[1]
    return [e.t for e in ev if e.kind == "FALL"]


def event_eval(sets, cfg, hold):
    by_group = {}
    for v, lab, p in videos_of(sets):
        cache = load_cache(p)
        m = match_events(run_events(cfg, cache, v, hold), lab["falls"])
        key = "UR Fall" if group_of(v) == "ur" else "GMDCSA24"
        by_group.setdefault(key, MatchResult())
        by_group[key] += m
    return by_group


def fmt(r: MatchResult):
    lat = "" if r.mean_latency is None else f"{r.mean_latency:.2f}"
    return f"| {r.tp} | {r.fp} | {r.fn} | {r.precision:.2f} | {r.recall:.2f} | **{r.f1:.2f}** | {lat} |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True, help="cache:labels.csv:split1,split2")
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--config", default="config_demo.yaml")
    ap.add_argument("--out", default="models/fall_mlp.npz")
    ap.add_argument("--report", default="results/learned.md")
    ap.add_argument("--hold-last", type=float, default=2.0)
    ap.add_argument("--thresh", type=float, default=0.2, help="ngưỡng xác suất của D (chọn bằng kiểm tra chéo trên train)")
    args = ap.parse_args()

    from sklearn.model_selection import GroupKFold
    base = load_config(args.config).detector
    train_sets, test_sets = parse_sets(args.train), parse_sets(args.test)
    lines = ["# Phương pháp D — học máy trên đặc trưng khung xương", ""]

    print("1) Trích đặc trưng tập train…")
    X, y, g, vids = collect(train_sets, base)
    assert X.shape[1] == N_FEATURES
    lines += [f"Train: {len(y)} khung ({int(y.sum())} khung đang nằm sau ngã), nhóm: {sorted(set(g))}", ""]
    print(f"   {len(y)} khung, {int(y.sum())} dương, nhóm {sorted(set(g))}")

    print("2) Kiểm tra chéo theo người/bộ dữ liệu…")
    lines += ["## Kiểm tra chéo trên train (mỗi lần bỏ ra một người/bộ dữ liệu)", "",
              "| Bỏ ra | Số khung | AUC | F1 khung |", "|---|---|---|---|"]
    cv_g = g if len(set(g)) >= 2 else vids            # chỉ có một nhóm → chia theo video, 5 phần
    n_splits = len(set(cv_g)) if len(set(g)) >= 2 else 5
    for tr_idx, va_idx in GroupKFold(n_splits=n_splits).split(X, y, cv_g):
        scaler, mlp = fit(X[tr_idx], y[tr_idx])
        p = mlp.predict_proba(scaler.transform(X[va_idx]))[:, 1]
        auc, f1 = frame_metrics(y[va_idx], p)
        name = g[va_idx][0] if cv_g is g else f"{len(set(vids[va_idx]))} video"
        lines.append(f"| {name} | {len(va_idx)} | {auc:.3f} | {f1:.3f} |")
        print(f"   bỏ ra {name}: AUC {auc:.3f}, F1 khung {f1:.3f}")

    print("3) Huấn luyện mô hình cuối…")
    scaler, mlp = fit(X, y)
    export_sklearn(mlp, scaler, args.out)
    print(f"   đã lưu {args.out}")

    print("4) Đánh giá trên tập test (chưa dùng để huấn luyện)…")
    Xt, yt, gt, _ = collect(test_sets, base)
    pt = mlp.predict_proba(scaler.transform(Xt))[:, 1]
    auc, f1 = frame_metrics(yt, pt)
    lines += ["", f"## Test theo khung: AUC {auc:.3f}, F1 {f1:.3f} ({len(yt)} khung)", ""]

    configs = {
        "B · demo (cúi + mép + nằm yên)": replace(base, method="B"),
        "B · cúi + mép ảnh": replace(base, method="B", still_thresh=0.0),
        "D · mép ảnh": replace(base, method="D", learned_model=args.out, learned_thresh=args.thresh,
                               still_thresh=0.0, bend_guard=False),
        "D · mép ảnh + nằm yên": replace(base, method="D", learned_model=args.out, learned_thresh=args.thresh,
                                         bend_guard=False),
    }
    lines += [f"## Test theo sự kiện (giữ khung cuối có người {args.hold_last:g} s)", "",
              "| Bộ dữ liệu | Cấu hình | TP | FP | FN | P | R | F1 | Trễ (s) |", "|---|---|---|---|---|---|---|---|---|"]
    for name, cfg in configs.items():
        for ds, r in event_eval(test_sets, cfg, args.hold_last).items():
            lines.append(f"| {ds} | {name} " + fmt(r))
            print(f"   {ds:9s} {name:24s} TP={r.tp} FP={r.fp} FN={r.fn} F1={r.f1:.2f}")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Đã ghi {args.report}")


if __name__ == "__main__":
    main()
