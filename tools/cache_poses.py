"""Chạy YOLO Pose + ByteTrack MỘT LẦN cho mọi video trong labels.csv và lưu cache.

  python tools/cache_poses.py --labels data/labels.csv --videos data/videos
  python tools/cache_poses.py --imgsz 640 --cache cache_640
  python tools/cache_poses.py --weights yolo11s-pose.pt --imgsz 640 --cache cache_s640
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.replay import cache_path_for, read_labels, save_cache  # noqa: E402
from src.video import VideoSource  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--videos", default="data/videos")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--weights", help="vd: yolo11s-pose.pt (mô hình lớn hơn, chậm hơn)")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--force", action="store_true", help="chạy lại kể cả khi đã có cache")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.imgsz: cfg.model.imgsz = args.imgsz
    if args.weights: cfg.model.weights = args.weights
    if args.threads: cfg.model.threads = args.threads

    from src.pose_tracker import PoseTracker
    tracker = PoseTracker(cfg.model)
    tracker.warmup()

    labels = read_labels(args.labels)
    for i, video in enumerate(labels, 1):
        out = cache_path_for(args.cache, video)
        if out.exists() and not args.force:
            print(f"[{i}/{len(labels)}] bỏ qua (đã có cache): {video}")
            continue
        path = Path(args.videos) / video
        if not path.exists():
            print(f"[{i}/{len(labels)}] KHÔNG TÌM THẤY: {path}")
            continue
        tracker.reset()
        src = VideoSource(str(path))
        frames, infer, shape = [], [], None
        t0 = time.perf_counter()
        while True:
            ok, frame, t = src.read()
            if not ok:
                break
            shape = shape or frame.shape[:2]
            people = tracker(frame)
            infer.append(tracker.last_infer_ms)
            frames.append({"t": round(t, 4), "people": [p.to_json() for p in people]})
        src.release()
        med = statistics.median(infer) if infer else 0.0
        save_cache(out, video, src.fps, frames, med, shape)
        print(f"[{i}/{len(labels)}] {video}: {len(frames)} frame, {med:.0f} ms/frame, "
              f"{time.perf_counter() - t0:.0f}s → {out}")


if __name__ == "__main__":
    main()
