"""Nguồn video: webcam hoặc file MP4 (TV3).

- Với file MP4: thời gian t = số frame / FPS của video → kết quả đánh giá không phụ
  thuộc tốc độ CPU (quan trọng để so sánh công bằng giữa các máy).
- Với webcam: luôn lấy FRAME MỚI NHẤT bằng một luồng đọc riêng, tránh độ trễ tích
  luỹ khi CPU xử lý chậm hơn tốc độ camera; t = đồng hồ thực.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class _LatestFrameCamera:
    def __init__(self, index: int, width: int, height: int):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Không mở được webcam {index}. Thử --source 1")
        self._frame = None
        self._lock = threading.Lock()
        self._stop = False
        self._new = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
            self._new.set()

    def read(self, timeout: float = 2.0):
        if not self._new.wait(timeout):
            return False, None
        self._new.clear()
        with self._lock:
            return True, self._frame.copy()

    def release(self):
        self._stop = True
        self._thread.join(timeout=1.0)
        self.cap.release()


class VideoSource:
    def __init__(self, source: str | int, width: int = 640, height: int = 480):
        src = str(source)
        self.is_camera = src.isdigit()
        self.frame_idx = 0
        self._t0 = time.monotonic()
        if self.is_camera:
            self._cam = _LatestFrameCamera(int(src), width, height)
            self.fps = 30.0
            self.total_frames = -1
        else:
            self._cap = cv2.VideoCapture(src)
            if not self._cap.isOpened():
                raise RuntimeError(f"Không mở được video: {src}")
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps if fps and fps > 1 else 30.0
            self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        """Trả về (ok, frame, t_giây)."""
        if self.is_camera:
            ok, frame = self._cam.read()
            t = time.monotonic() - self._t0
        else:
            ok, frame = self._cap.read()
            t = self.frame_idx / self.fps
        if ok:
            self.frame_idx += 1
        return ok, frame, t

    def release(self):
        if self.is_camera:
            self._cam.release()
        else:
            self._cap.release()
