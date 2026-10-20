"""YOLO Pose + ByteTrack trên CPU (TV1).

Đầu vào : 1 frame BGR (numpy).
Đầu ra  : list[Person] — xem src/types.py.
"""
from __future__ import annotations

import time

import numpy as np

from .config import ModelConfig
from .types import Person


class PoseTracker:
    def __init__(self, cfg: ModelConfig):
        import torch
        from ultralytics import YOLO

        self.cfg = cfg
        if cfg.device == "cpu":
            torch.set_num_threads(max(1, int(cfg.threads)))
        self.model = YOLO(cfg.weights)
        self.last_infer_ms = 0.0
        self._untracked = 0

    def warmup(self, shape=(480, 640, 3), n: int = 2) -> None:
        """Chạy vài frame rỗng để lần đo FPS đầu tiên không bị chậm bất thường."""
        dummy = np.zeros(shape, dtype=np.uint8)
        for _ in range(n):
            self.model.predict(dummy, imgsz=self.cfg.imgsz, device=self.cfg.device, verbose=False)

    def reset(self) -> None:
        """Xoá trạng thái tracker — gọi khi chuyển sang video khác."""
        pred = getattr(self.model, "predictor", None)
        for trk in getattr(pred, "trackers", None) or []:
            try:
                trk.reset()
            except Exception:  # phiên bản ultralytics cũ không có reset()
                pass
        if pred is not None and hasattr(pred, "vid_path"):
            pred.vid_path = [None]

    def __call__(self, frame: np.ndarray) -> list[Person]:
        t0 = time.perf_counter()
        res = self.model.track(
            frame,
            persist=True,
            tracker=self.cfg.tracker,
            device=self.cfg.device,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            classes=[0],
            verbose=False,
        )[0]
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0

        if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
            return []

        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else None
        kps = res.keypoints.xy.cpu().numpy()
        kconf = (res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None
                 else np.ones(kps.shape[:2], dtype=float))

        people = []
        for i in range(len(boxes)):
            if ids is not None:
                tid = int(ids[i])
            else:  # ByteTrack chưa gán ID (frame đầu / độ tin cậy thấp) → ID âm tạm thời
                self._untracked += 1
                tid = -self._untracked
            people.append(Person(tid, boxes[i], float(scores[i]), kps[i], kconf[i]))
        return people
