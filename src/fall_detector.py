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
    stand_ref: tuple[float, float, float] | None = None  # (hip_y, body_len, box_h) lần gần nhất đứng thẳng
    prob: float = 0.0                                     # phương pháp D: xác suất "đang nằm" của khung hiện tại
    motion_hist: deque = field(default_factory=lambda: deque(maxlen=150))  # (t, x vai, y vai, body_len)
    susp_frames: int = 0
    susp_horiz: int = 0
    await_still: bool = False   # đã đủ điều kiện nằm, đang chờ người nằm yên (still_thresh)
    last_alert_t: float = -1e9
    last_seen: float = 0.0
    last_center: tuple[float, float] = (0.0, 0.0)
    last_len: float = 1.0
    speed: float = 0.0


class FallDetector:
    def __init__(self, cfg: DetectorConfig):
        if cfg.method not in ("A", "B", "C", "D"):
            raise ValueError("method phải là A, B, C hoặc D")
        self.cfg = cfg
        self.tracks: dict[int, _Track] = {}
        self.model = None
        if cfg.method == "D":
            from .learned import LearnedModel
            self.model = LearnedModel(cfg.learned_model)
        self.collect: list | None = None   # tools/train_fall_model.py: gom (id, t, body_len, vectơ đặc trưng)

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.tracks.clear()

    def is_horizontal(self, f: PoseFeatures) -> bool | None:
        """True = nằm ngang, False = thẳng người, None = không đủ thông tin để kết luận."""
        c = self.cfg
        if c.edge_guard and f.truncated:
            # Người bị cắt mất nửa dưới ở mép ảnh (vd đứng sát webcam cúi người): khung và góc thân
            # chỉ phản ánh phần nhìn thấy → KHÔNG RÕ: không tính là nằm, cũng không tính là đã đứng dậy.
            return None
        if c.method == "A":
            return f.aspect_ratio > c.ratio_thresh
        if f.torso_angle is None:
            # Không thấy vai/hông (vd ngồi sát webcam, chỉ thấy đầu-vai): không đủ thông tin.
            # ratio_fallback=False → coi là KHÔNG nằm, tránh báo nhầm khi khung người bị cắt.
            return c.ratio_fallback and f.aspect_ratio > c.ratio_thresh
        return f.torso_angle > c.angle_thresh

    # ------------------------------------------------------------------ #
    def update(self, people: list[Person], t: float,
               frame_shape: tuple[int, int] | None = None) -> tuple[list[TrackResult], list[FallEvent]]:
        """Cập nhật với toàn bộ người trong 1 frame tại thời điểm t (giây).
        frame_shape = (cao, rộng) của ảnh — cần cho edge_guard; không truyền thì bỏ qua kiểm tra mép."""
        results: list[TrackResult] = []
        events: list[FallEvent] = []
        people = self._drop_duplicates(people)
        current_ids = {p.track_id for p in people}

        for p in people:
            f = compute_features(p.bbox, p.keypoints, p.kp_conf, self.cfg.kp_conf)
            if frame_shape is not None:
                f.truncated = self._truncated(p, frame_shape)
            if not f.valid:
                continue
            if self.model is not None or self.collect is not None:
                from .learned import static_features
                f.static_x = static_features(p.bbox, p.keypoints, p.kp_conf, self.cfg.kp_conf, f.legs_upright)
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

    def _truncated(self, p: Person, frame_shape) -> bool:
        """Khung người chạm mép ảnh (trong 2%) VÀ không thấy cổ chân nào → bị cắt mất nửa dưới."""
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = p.bbox
        m = 0.02
        touch = x1 <= m * w or y1 <= m * h or x2 >= (1 - m) * w or y2 >= (1 - m) * h
        ankles = max(p.kp_conf[15], p.kp_conf[16]) >= self.cfg.kp_conf
        return bool(touch and not ankles)

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

    def _confirm(self, tr: _Track, tid: int, f: PoseFeatures, t: float) -> FallEvent | None:
        """SUSPECTED → FALL, phát cảnh báo (trừ khi vừa báo cho người này trong alert_cooldown)."""
        self._set(tr, State.FALL, t)
        if t - tr.last_alert_t < self.cfg.alert_cooldown:
            return None
        tr.last_alert_t = t
        return FallEvent(t, tid, "FALL", f.torso_angle, f.aspect_ratio, f"method={self.cfg.method}")

    def _motion(self, tr: _Track, t: float) -> float:
        """Mức cử động trong still_window giây gần nhất: biên độ dịch chuyển lớn nhất của vai
        (max - min theo x hoặc y), chia chiều dài cơ thể. Chưa đủ dữ liệu → vô cùng (coi như còn cử động)."""
        win = [m for m in tr.motion_hist if t - m[0] <= self.cfg.still_window]
        if len(win) < 3 or win[-1][0] - win[0][0] < 0.8 * self.cfg.still_window:
            return float("inf")
        xs, ys = [m[1] for m in win], [m[2] for m in win]
        scale = float(np.median([m[3] for m in win]))
        return max(max(xs) - min(xs), max(ys) - min(ys)) / max(scale, 1.0)

    def _set(self, tr: _Track, s: State, t: float) -> None:
        tr.state, tr.state_since = s, t
        tr.susp_frames = tr.susp_horiz = 0
        tr.await_still = False

    def _step(self, tid: int, tr: _Track, f: PoseFeatures, t: float):
        c = self.cfg
        tr.history.append((t, f.hip_y, f.body_len))
        tr.motion_hist.append((t, f.anchor[0], f.anchor[1], f.body_len))
        tr.last_seen, tr.last_center, tr.last_len = t, f.center, f.body_len

        drop, speed = self._descent(tr, t)
        x = None
        if f.static_x is not None:
            from .learned import temporal_features
            ref = tr.stand_ref
            x = np.concatenate([f.static_x, temporal_features(
                None if ref is None else (f.hip_y - ref[0]) / max(ref[1], 1.0),
                None if ref is None else f.box_h / max(ref[2], 1.0),
                speed, self._motion(tr, t))])
            if self.collect is not None:
                self.collect.append((tid, t, f.body_len, x))
        if self.model is not None and not (c.edge_guard and f.truncated):
            tr.prob = self.model.proba(x)          # D: mạng nơ-ron quyết định "đang nằm"
            h = tr.prob >= c.learned_thresh
        else:
            h = self.is_horizontal(f)
        unknown = h is None
        horizontal = bool(h)
        if h is False and f.torso_angle is not None and f.torso_angle < 30:
            tr.stand_ref = (f.hip_y, f.body_len, f.box_h)
        if horizontal and c.bend_guard and f.legs_upright and tr.stand_ref is not None:
            # Cúi người: thân ngang nhưng chân vẫn dựng đứng và hông chưa hạ so với lúc đứng → coi là đứng.
            if (f.hip_y - tr.stand_ref[0]) / max(tr.stand_ref[1], 1.0) < c.bend_hip_drop:
                horizontal = False
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

        if unknown:
            pass  # không rõ tư thế → giữ nguyên bộ đếm đứng/nằm
        elif horizontal:
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
            if not unknown:
                tr.susp_frames += 1
                tr.susp_horiz += int(lying)
            if upright_for >= 0.5:  # đứng dậy ngay → huỷ nghi ngờ
                self._set(tr, State.NORMAL, t)
            elif tr.await_still:
                # Đã nằm đủ lâu; chỉ còn chờ nằm yên. Chống đẩy, bò: còn cử động đều → không báo.
                if self._motion(tr, t) <= c.still_thresh:
                    event = self._confirm(tr, tid, f, t)
            elif t - tr.state_since >= c.confirm_time:
                if tr.susp_horiz / max(tr.susp_frames, 1) < c.confirm_ratio:
                    self._set(tr, State.NORMAL, t)
                elif c.still_thresh > 0 and self._motion(tr, t) > c.still_thresh:
                    tr.await_still = True
                else:
                    event = self._confirm(tr, tid, f, t)

        elif tr.state == State.FALL:
            if upright_for >= c.recover_time:
                self._set(tr, State.NORMAL, t)
                tr.last_descent_t = None
                event = FallEvent(t, tid, "RECOVERED", f.torso_angle, f.aspect_ratio)

        res = TrackResult(tid, tr.state, lying, f.torso_angle, f.aspect_ratio, speed)
        return res, event


__all__ = ["FallDetector", "State"]
