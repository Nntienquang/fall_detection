"""Thuật toán phát hiện té ngã theo thời gian (TV2).

Ba phương pháp để so sánh trong báo cáo:
  A — Bounding box : "nằm" khi W/H > ratio_thresh                          (baseline)
  B — Pose tĩnh    : "nằm" khi góc thân > angle_thresh (fallback W/H)       (cải tiến)
  C — Pose + thời gian : như B nhưng chỉ nghi ngờ khi trước đó có CÚ HẠ NHANH
                     của hông (loại trừ chủ động nằm xuống, người nằm sẵn) (đề xuất)

Máy trạng thái cho mỗi track ID:

    NORMAL ──(điều kiện nghi ngờ)──▶ SUSPECTED ──(nằm đủ confirm_time)──▶ FALL  → phát cảnh báo
       ▲                                │                                   │
       └──────(đứng dậy lại)────────────┘◀───(thẳng người đủ recover_time)──┘  → RECOVERED

Bổ sung: nối lại lịch sử khi ByteTrack đổi ID (người ngã thường bị mất dấu vài frame).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import DetectorConfig
from .features import PoseFeatures, compute_features
from .types import FallEvent, Person, State, TrackResult

SPEED_WINDOW = 0.3  # giây — cửa sổ đo vận tốc tức thời của hông


@dataclass
class _Track:
    history: deque = field(default_factory=lambda: deque(maxlen=120))  # (t, hip_y, body_len)
    state: State = State.NORMAL
    state_since: float = 0.0
    seen_upright: bool = False
    upright_since: float | None = None
    last_descent_t: float | None = None
    last_fast_t: float | None = None
    support_hist: deque = field(default_factory=lambda: deque(maxlen=90))  # (t, có chống tay) khi thân ngang
    susp_frames: int = 0
    susp_horiz: int = 0
    last_alert_t: float = -1e9
    last_seen: float = 0.0
    last_center: tuple[float, float] = (0.0, 0.0)
    last_len: float = 1.0
    speed: float = 0.0


class FallDetector:
    def __init__(self, cfg: DetectorConfig):
        if cfg.method not in ("A", "B", "C"):
            raise ValueError("method phải là A, B hoặc C")
        self.cfg = cfg
        self.tracks: dict[int, _Track] = {}

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.tracks.clear()

    def is_horizontal(self, f: PoseFeatures) -> bool:
        c = self.cfg
        if c.method == "A":
            return f.aspect_ratio > c.ratio_thresh
        if f.torso_angle is None:
            # Không thấy vai/hông (vd ngồi sát webcam, chỉ thấy đầu-vai): không đủ thông tin.
            # ratio_fallback=False → coi là KHÔNG nằm, tránh báo nhầm khi khung người bị cắt.
            return c.ratio_fallback and f.aspect_ratio > c.ratio_thresh
        return f.torso_angle > c.angle_thresh

    # ------------------------------------------------------------------ #
    def update(self, people: list[Person], t: float) -> tuple[list[TrackResult], list[FallEvent]]:
        """Cập nhật với toàn bộ người trong 1 frame tại thời điểm t (giây)."""
        results: list[TrackResult] = []
        events: list[FallEvent] = []
        people = self._drop_duplicates(people)
        current_ids = {p.track_id for p in people}

        for p in people:
            f = compute_features(p.bbox, p.keypoints, p.kp_conf, self.cfg.kp_conf)
            if not f.valid:
                continue
            tr = self._get_track(p.track_id, f, t, current_ids)
            res, ev = self._step(p.track_id, tr, f, t)
            results.append(res)
            if ev is not None:
                events.append(ev)

        self._cleanup(t)
        return results, events

    # ------------------------------------------------------------------ #
    @staticmethod
    def _overlap(a: np.ndarray, b: np.ndarray) -> float:
        """Phần giao / diện tích khung nhỏ hơn (1.0 = khung nhỏ nằm trọn trong khung lớn)."""
        iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        area = lambda r: max((r[2] - r[0]) * (r[3] - r[1]), 1e-6)
        return iw * ih / min(area(a), area(b))

    def _drop_duplicates(self, people: list[Person]) -> list[Person]:
        """YOLO đôi khi trả 2 khung cho cùng 1 người đang nằm, ByteTrack gán ID mới cho khung thứ 2.
        Giữ khung thuộc track đã có (rồi đến khung confidence cao hơn), bỏ khung trùng — nhờ vậy
        khi ID cũ biến mất ở frame sau, ID mới được nối lại với lịch sử cũ thay vì bắt đầu từ đầu."""
        if len(people) < 2 or self.cfg.dup_overlap >= 1.0:
            return people
        ranked = sorted(people, key=lambda p: (p.track_id not in self.tracks, -p.score))
        kept: list[Person] = []
        for p in ranked:
            if all(self._overlap(p.bbox, k.bbox) < self.cfg.dup_overlap for k in kept):
                kept.append(p)
        return kept

    def _get_track(self, tid: int, f: PoseFeatures, t: float, current_ids: set[int]) -> _Track:
        if tid in self.tracks:
            return self.tracks[tid]
        # Thử nối với một track vừa mất dấu ở gần vị trí này (ID switch khi ngã)
        best_id, best_d = None, float("inf")
        for old_id, old in self.tracks.items():
            if old_id in current_ids or t - old.last_seen > self.cfg.track_ttl or old.last_seen >= t:
                continue
            d = np.hypot(f.center[0] - old.last_center[0], f.center[1] - old.last_center[1])
            d /= max(old.last_len, f.body_len, 1.0)
            if d < self.cfg.reassoc_dist and d < best_d:
                best_id, best_d = old_id, d
        tr = self.tracks.pop(best_id) if best_id is not None else _Track(state_since=t)
        self.tracks[tid] = tr
        return tr

    def _cleanup(self, t: float) -> None:
        dead = [k for k, v in self.tracks.items() if t - v.last_seen > self.cfg.track_ttl]
        for k in dead:
            del self.tracks[k]

    # ------------------------------------------------------------------ #
    def _descent(self, tr: _Track, t: float) -> tuple[float, float]:
        """Trả về (độ hạ chuẩn hoá trong cửa sổ, vận tốc hạ hiện tại)."""
        win = [h for h in tr.history if t - h[0] <= self.cfg.drop_window]
        if len(win) < 2:
            return 0.0, 0.0
        scale = float(np.median([h[2] for h in win]))
        ys = [h[1] for h in win]
        drop = (ys[-1] - min(ys)) / scale
        # Vận tốc tức thời: độ hạ trong ~0.3 giây gần nhất (ít bị pha loãng hơn trung bình cả cửa sổ)
        recent = [h for h in win if t - h[0] <= SPEED_WINDOW]
        if len(recent) < 2:
            recent = win[-2:]
        dt = max(recent[-1][0] - recent[0][0], 1e-3)
        speed = max((recent[-1][1] - recent[0][1]) / scale / dt, 0.0)
        return drop, speed

    def _set(self, tr: _Track, s: State, t: float) -> None:
        tr.state, tr.state_since = s, t
        tr.susp_frames = tr.susp_horiz = 0

    def _step(self, tid: int, tr: _Track, f: PoseFeatures, t: float):
        c = self.cfg
        tr.history.append((t, f.hip_y, f.body_len))
        tr.last_seen, tr.last_center, tr.last_len = t, f.center, f.body_len

        horizontal = self.is_horizontal(f)
        drop, speed = self._descent(tr, t)
        tr.speed = speed
        if speed >= c.min_descent_speed:
            tr.last_fast_t = t
        # Cú hạ nhanh = hông đã hạ đủ sâu VÀ trong cửa sổ đó từng có lúc hạ đủ nhanh
        if drop >= c.drop_thresh and tr.last_fast_t is not None and t - tr.last_fast_t <= c.drop_window:
            tr.last_descent_t = t

        # Chống tay (chống đẩy, bò, đang chống dậy): thân ngang nhưng tay đỡ người → không tính là "nằm".
        #   - Ngoại lệ: vừa có cú hạ hông NHANH (trong descent_memory giây) → đang ngã sấp chống tay,
        #     vẫn tính là nằm. Chống đẩy thì người hạ xuống từ từ nên không bị ngoại lệ này.
        #   - Pha thấp của nhịp chống đẩy (ngực sát sàn, vai gần ngang cổ tay) vẫn được coi là đang chống
        #     nếu trong support_memory giây gần nhất có >= support_ratio số khung chống tay.
        supported = False
        if c.method != "A" and c.support_drop > 0 and horizontal:
            falling = tr.last_fast_t is not None and t - tr.last_fast_t <= c.descent_memory
            now = f.arm_drop >= c.support_drop and not falling
            tr.support_hist.append((t, now))
            if now:
                supported = True
            elif not falling:
                recent = [(ts, s) for (ts, s) in tr.support_hist if t - ts <= c.support_memory]
                hits = [ts for ts, s in recent if s]
                supported = len(hits) >= 2 and len(hits) / len(recent) >= c.support_ratio
        lying = horizontal and not supported

        if horizontal:
            tr.upright_since = None
        else:
            tr.seen_upright = True
            if tr.upright_since is None:
                tr.upright_since = t
        upright_for = 0.0 if tr.upright_since is None else t - tr.upright_since

        event = None

        if tr.state == State.NORMAL:
            if lying and tr.seen_upright:
                if c.method in ("A", "B"):
                    self._set(tr, State.SUSPECTED, t)
                elif tr.last_descent_t is not None and t - tr.last_descent_t <= c.descent_memory:
                    self._set(tr, State.SUSPECTED, t)

        elif tr.state == State.SUSPECTED:
            tr.susp_frames += 1
            tr.susp_horiz += int(lying)
            if upright_for >= 0.5:  # đứng dậy ngay → huỷ nghi ngờ
                self._set(tr, State.NORMAL, t)
            elif t - tr.state_since >= c.confirm_time:
                if tr.susp_horiz / max(tr.susp_frames, 1) >= c.confirm_ratio:
                    self._set(tr, State.FALL, t)
                    if t - tr.last_alert_t >= c.alert_cooldown:
                        tr.last_alert_t = t
                        event = FallEvent(t, tid, "FALL", f.torso_angle, f.aspect_ratio,
                                          f"method={c.method}")
                else:
                    self._set(tr, State.NORMAL, t)

        elif tr.state == State.FALL:
            if upright_for >= c.recover_time:
                self._set(tr, State.NORMAL, t)
                tr.last_descent_t = None
                event = FallEvent(t, tid, "RECOVERED", f.torso_angle, f.aspect_ratio)

        res = TrackResult(tid, tr.state, lying, f.torso_angle, f.aspect_ratio, speed)
        return res, event


__all__ = ["FallDetector", "State"]
