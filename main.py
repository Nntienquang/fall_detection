"""Chạy hệ thống phát hiện té ngã từ dòng lệnh.

Ví dụ:
  python main.py --source 0                                   # webcam
  python main.py --source data/videos/V05.mp4 --save-video    # video MP4
  python main.py --source data/videos/V05.mp4 --no-display --max-frames 300   # benchmark
  python main.py --source 0 --method B --imgsz 416 --threads 4
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import cv2

from src.config import load_config
from src.event_logger import EventLogger
from src.fall_detector import FallDetector
from src.visualize import render
from src.video import VideoSource


def parse_args():
    ap = argparse.ArgumentParser(description="Phát hiện người té ngã qua camera (YOLO Pose + ByteTrack)")
    ap.add_argument("--source", default="0", help="0/1 = webcam, hoặc đường dẫn file video")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--weights", help="ghi đè model.weights")
    ap.add_argument("--imgsz", type=int, help="ghi đè model.imgsz")
    ap.add_argument("--threads", type=int, help="ghi đè model.threads")
    ap.add_argument("--method", choices=["A", "B", "C"], help="ghi đè detector.method")
    ap.add_argument("--no-display", action="store_true", help="không mở cửa sổ (benchmark / server)")
    ap.add_argument("--save-video", action="store_true", help="lưu annotated.mp4")
    ap.add_argument("--max-frames", type=int, default=0, help="dừng sau N frame (0 = không giới hạn)")
    ap.add_argument("--out", default="outputs", help="thư mục lưu kết quả")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.weights: cfg.model.weights = args.weights
    if args.imgsz: cfg.model.imgsz = args.imgsz
    if args.threads: cfg.model.threads = args.threads
    if args.method: cfg.detector.method = args.method

    from src.pose_tracker import PoseTracker  # import muộn để --help chạy nhanh

    print(f"[INFO] Model {cfg.model.weights} | imgsz {cfg.model.imgsz} | threads {cfg.model.threads} "
          f"| method {cfg.detector.method}")
    tracker = PoseTracker(cfg.model)
    tracker.warmup()
    detector = FallDetector(cfg.detector)
    source = VideoSource(args.source)
    logger = EventLogger(args.out, meta={"source": args.source, **cfg.to_dict()})
    print(f"[INFO] Kết quả lưu tại: {logger.dir}")

    infer_ms, loop_ms = [], []
    fps_ema = 0.0
    n = 0
    t_start = time.perf_counter()
    try:
        while True:
            t_loop = time.perf_counter()
            ok, frame, t = source.read()
            if not ok:
                break

            people = tracker(frame)
            results, events = detector.update(people, t)

            dt = time.perf_counter() - t_loop
            fps_ema = (1 / dt) if fps_ema == 0 else 0.9 * fps_ema + 0.1 * (1 / max(dt, 1e-6))
            infer_ms.append(tracker.last_infer_ms)
            loop_ms.append(dt * 1000)
            n += 1

            need_img = (not args.no_display) or args.save_video or events
            vis = render(frame, people, results, cfg.detector.kp_conf, fps_ema, tracker.last_infer_ms) if need_img else None

            for ev in events:
                logger.log(ev, vis)
                print(f"[{ev.kind}] t={ev.t:.2f}s  ID={ev.track_id}")

            if args.save_video:
                logger.write_frame(vis, source.fps if not source.is_camera else max(fps_ema, 1.0))
            if not args.no_display:
                cv2.imshow("Fall Detection (Q de thoat)", vis)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        total = time.perf_counter() - t_start
        source.release()
        logger.close()
        cv2.destroyAllWindows()

    if n:
        print("\n========== TỔNG KẾT ==========")
        print(f"Số frame                        : {n}")
        print(f"End-to-end throughput           : {n / total:.2f} FPS")
        print(f"Model + tracking (trung vị)     : {statistics.median(infer_ms):.1f} ms/frame")
        print(f"Toàn bộ vòng lặp (trung vị)     : {statistics.median(loop_ms):.1f} ms/frame")
        print(f"Số cảnh báo té ngã              : {sum(e.kind == 'FALL' for e in logger.events)}")
        print(f"Thư mục kết quả                 : {logger.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
