"""Unit test thuật toán — không cần webcam, không cần YOLO.  Chạy:  python -m pytest -q"""
from dataclasses import replace

import numpy as np
import pytest

from src.config import DetectorConfig
from src.fall_detector import FallDetector
from src.features import compute_features, torso_angle
from src.metrics import match_events
from src.simulate import scenario, skeleton, to_person
from src.types import State

CFG = DetectorConfig()


def run(name, method="C", seed=1, **kw):
    frames, falls = scenario(name, seed=seed, **kw)
    det = FallDetector(replace(CFG, method=method))
    events, states = [], []
    for t, people in frames:
        res, evs = det.update(people, t)
        events += evs
        states.append({r.track_id: r.state for r in res})
    return events, falls, states


def falls_of(events):
    return [e for e in events if e.kind == "FALL"]


# ---------------------------- features ---------------------------- #
def test_torso_angle_upright_and_lying():
    conf = np.ones(17)
    assert torso_angle(skeleton(300, 300, 200, 0), conf, 0.3) == pytest.approx(0, abs=1)
    assert torso_angle(skeleton(300, 300, 200, 90), conf, 0.3) == pytest.approx(90, abs=1)


def test_missing_keypoints_fall_back_to_bbox():
    p = to_person(1, skeleton(300, 300, 200, 0))
    p.kp_conf[:] = 0.0
    f = compute_features(p.bbox, p.keypoints, p.kp_conf, 0.35)
    assert f.torso_angle is None and f.valid


# ---------------------------- detector ---------------------------- #
@pytest.mark.parametrize("method", ["A", "B", "C"])
def test_walking_never_alarms(method):
    events, _, _ = run("walk", method)
    assert falls_of(events) == []


@pytest.mark.parametrize("method", ["A", "B", "C"])
def test_real_fall_alarms_once(method):
    events, falls, _ = run("fall", method)
    alerts = falls_of(events)
    assert len(alerts) == 1
    assert match_events([e.t for e in alerts], falls).tp == 1


def test_method_c_ignores_slow_lie_down_but_b_does_not():
    assert falls_of(run("lie_down", "C")[0]) == []
    assert len(falls_of(run("lie_down", "B")[0])) == 1  # chính là điểm yếu của baseline


def test_method_c_ignores_bending():
    assert falls_of(run("bend", "C")[0]) == []


def test_sitting_never_alarms():
    for m in "ABC":
        assert falls_of(run("sit", m)[0]) == []


def test_recovery_returns_to_normal():
    events, _, states = run("fall_getup", "C")
    assert [e.kind for e in events] == ["FALL", "RECOVERED"]
    assert states[-1][1] == State.NORMAL


def test_person_lying_from_start_is_not_a_new_fall():
    det = FallDetector(CFG)
    events = []
    for i in range(60):
        p = to_person(1, skeleton(300, 400, 200, 88))
        events += det.update([p], i / 15)[1]
    assert events == []


def test_only_the_fallen_person_alarms():
    events, _, _ = run("two_people", "C")
    assert [e.track_id for e in falls_of(events)] == [2]


def test_track_id_switch_keeps_history():
    """ByteTrack đổi ID giữa lúc ngã: thuật toán vẫn phải phát hiện nhờ nối lại track."""
    frames, falls = scenario("fall", seed=2)
    det = FallDetector(CFG)
    alerts = []
    for t, people in frames:
        if t > falls[0][0] + 0.3:   # sau khi bắt đầu ngã 0.3s, ID đổi từ 1 → 7
            for p in people:
                p.track_id = 7
        alerts += [e for e in det.update(people, t)[1] if e.kind == "FALL"]
    assert len(alerts) == 1 and alerts[0].track_id == 7


def test_no_duplicate_alert_within_cooldown():
    det = FallDetector(replace(CFG, recover_time=0.5))
    frames, _ = scenario("fall_getup", seed=4)
    n = sum(e.kind == "FALL" for t, ps in frames for e in det.update(ps, t)[1])
    assert n == 1


# ---------------------------- metrics ---------------------------- #
def test_event_matching():
    m = match_events([3.0, 3.4, 20.0], [(2.5, 5.0), (40.0, 42.0)])
    assert (m.tp, m.fp, m.fn) == (1, 1, 1)
    assert m.latencies == [pytest.approx(0.5)]
    assert m.precision == pytest.approx(0.5) and m.recall == pytest.approx(0.5)


def test_duplicate_box_with_new_id_keeps_history():
    """Lỗi thật trên UR Fall (fall-08): lúc người nằm xuống, YOLO trả 2 khung chồng nhau,
    ByteTrack gán ID mới cho khung thứ 2, rồi ID cũ biến mất. Phải vẫn phát hiện được."""
    frames, falls = scenario("fall", seed=2)
    det = FallDetector(CFG)
    alerts = []
    t_dup = falls[0][0] + 0.4
    for t, people in frames:
        if t >= t_dup:
            p = people[0]
            dup = to_person(9, p.keypoints.copy())
            dup.bbox = p.bbox + np.array([4, 2, -3, 1])   # khung gần trùng, ID mới
            dup.score = 0.5
            people = [p, dup] if t < t_dup + 0.2 else [dup]  # 0.2s có cả hai, sau đó chỉ còn ID mới
        alerts += [e for e in det.update(people, t)[1] if e.kind == "FALL"]
    assert len(alerts) == 1 and alerts[0].track_id == 9


def test_two_separate_people_are_not_merged():
    det = FallDetector(CFG)
    a = to_person(1, skeleton(150, 300, 200, 0))
    b = to_person(2, skeleton(450, 300, 200, 0))
    res, _ = det.update([a, b], 0.0)
    assert {r.track_id for r in res} == {1, 2}


def test_close_up_without_hips_does_not_alarm_when_fallback_off():
    """Ngồi sát webcam: chỉ thấy đầu-vai, khung bè ngang (W/H > 1), không có hông."""
    det = FallDetector(replace(CFG, method="B", ratio_fallback=False, confirm_time=0.5))
    alerts = []
    for i in range(45):          # 1.5 s đứng thẳng, thấy đủ khớp
        alerts += det.update([to_person(1, skeleton(300, 300, 200, 0))], i / 30)[1]
    for i in range(45, 150):     # rồi lại gần camera: mất hông, khung rộng
        p = to_person(1, skeleton(300, 300, 200, 0))
        p.kp_conf[11:] = 0.0
        p.bbox = np.array([50.0, 100.0, 600.0, 480.0])
        alerts += det.update([p], i / 30)[1]
    assert alerts == []


def _pushup_frames(fps=30, floor=400.0, L=200.0, cx=300.0):
    """Đứng 1.5 s → từ từ cúi, đặt 2 tay xuống sàn (1.5 s) → chống đẩy 8 s (nhịp 1.5 s/lần).
    Khi chống đẩy: cổ tay chạm sàn ngay dưới vai; pha thấp gập tay nên vai hạ sát sàn hơn."""
    frames = []
    for i in range(int(11 * fps)):
        t = i / fps
        if t < 1.5:
            a, low = 0.0, 0.0
        elif t < 3.0:
            a, low = 68.0 * (t - 1.5) / 1.5, 0.0
        else:
            low = 0.5 - 0.5 * np.cos(2 * np.pi * (t - 3.0) / 1.5)   # 0 = tay thẳng, 1 = pha thấp
            a = 68.0 + 14.0 * low
        rad = np.radians(a)
        k = skeleton(cx, floor - 0.5 * L * np.cos(rad), L, a)
        if a > 30:  # tay chống xuống sàn, ngay dưới vai
            for s, e, w in ((5, 7, 9), (6, 8, 10)):
                k[w] = [k[s][0], floor]
                k[e] = (k[s] + k[w]) / 2 + [0.08 * L * low, 0]
        frames.append((t, [to_person(1, k)]))
    return frames


def test_pushup_does_not_alarm():
    """Lỗi thật trên webcam 26/10: chống đẩy bị báo té ngã (thân ngang 81°).
    Tay chống thẳng xuống sàn → đang tự đỡ người, không phải nằm."""
    demo = replace(CFG, method="B", angle_thresh=45.0, ratio_thresh=1.0,
                   confirm_time=0.5, ratio_fallback=False, support_drop=0.25)
    for cfg, expect in ((demo, 0), (replace(demo, support_drop=0.0), None)):
        det = FallDetector(cfg)
        alerts = []
        for t, people in _pushup_frames():
            alerts += [e for e in det.update(people, t)[1] if e.kind == "FALL"]
        if expect is None:
            assert alerts, "khi tắt luật chống tay, kịch bản này phải báo nhầm (kiểm tra test có ý nghĩa)"
        else:
            assert alerts == []


def test_hand_catch_fall_still_alarms():
    """Ngã sấp chống tay (ur_fall-05/09): chống tay ~0.3 s ngay sau cú hạ nhanh rồi nằm bẹp → vẫn báo."""
    frames, falls = scenario("fall", seed=3)
    det = FallDetector(replace(CFG, method="B", angle_thresh=45.0, confirm_time=0.5, support_drop=0.25))
    alerts = []
    t_land = falls[0][0] + 0.6
    for t, people in frames:
        p = people[0]
        if t_land <= t < t_land + 0.3:   # 0.3 s chống tay chạm sàn ngay dưới vai
            k = p.keypoints.copy()
            floor = k[:, 1].max() + 0.6 * (k[:, 1].max() - k[:, 1].min() + 40)
            for s, w in ((5, 9), (6, 10)):
                k[w] = [k[s][0], floor]
            p = to_person(p.track_id, k)
        alerts += [e for e in det.update([p], t)[1] if e.kind == "FALL"]
    assert len(alerts) == 1 and match_events([e.t for e in alerts], falls).tp == 1


def test_close_person_cut_at_frame_edge_does_not_alarm():
    """Lỗi thật trên webcam 26/10: người đứng sát webcam cúi ~45°, bị cắt mất nửa dưới ở mép phải/dưới
    ảnh 640x480 → bị báo FALL. edge_guard: chạm mép + không thấy cổ chân → không kết luận 'nằm'."""
    demo = replace(CFG, method="B", angle_thresh=45.0, confirm_time=0.5, ratio_fallback=False)
    results = {}
    for guard in (False, True):
        det = FallDetector(replace(demo, edge_guard=guard))
        alerts = []
        for i in range(105):                       # 1.5 s đứng thẳng rồi 2 s cúi người 50°
            t = i / 30
            p = to_person(1, skeleton(560, 420, 500, 0 if t < 1.5 else 50), L=500)
            p.bbox = np.clip(p.bbox, 0, [639, 479, 639, 479])   # bị cắt ở mép ảnh
            p.kp_conf[15:] = 0.0                                 # cổ chân ngoài khung
            alerts += [e for e in det.update([p], t, (480, 640))[1] if e.kind == "FALL"]
        results[guard] = len(alerts)
    assert results[False] >= 1, "không bật edge_guard thì ca này phải bị báo nhầm (test có ý nghĩa)"
    assert results[True] == 0


def test_lying_person_with_feet_visible_still_alarms_with_edge_guard():
    frames, falls = scenario("fall", seed=1)
    det = FallDetector(replace(CFG, method="B", edge_guard=True))
    alerts = [e for t, ps in frames for e in det.update(ps, t, (480, 640))[1] if e.kind == "FALL"]
    assert len(alerts) == 1


DEMO_STILL = replace(CFG, method="B", angle_thresh=45.0, ratio_thresh=1.0, confirm_time=0.5,
                     ratio_fallback=False, still_thresh=0.12)


def test_pushup_moving_does_not_alarm_with_still_check():
    """Chống đẩy: thân nằm ngang nhưng vai lên xuống liên tục → chờ nằm yên, không báo."""
    for still, expect_alarm in ((0.0, True), (0.12, False)):
        det = FallDetector(replace(DEMO_STILL, still_thresh=still))
        alerts = [e for t, ps in _pushup_frames() for e in det.update(ps, t)[1] if e.kind == "FALL"]
        assert bool(alerts) == expect_alarm


def test_real_fall_still_alarms_with_still_check():
    """Ngã rồi nằm yên → vẫn báo, chậm hơn khoảng 1 s (chờ đủ cửa sổ bất động)."""
    for seed in (1, 2, 3):
        frames, falls = scenario("fall", seed=seed)
        det = FallDetector(DEMO_STILL)
        alerts = [e for t, ps in frames for e in det.update(ps, t)[1] if e.kind == "FALL"]
        assert len(alerts) == 1 and match_events([e.t for e in alerts], falls).tp == 1


def test_still_check_cancels_when_person_gets_up():
    """Đang chờ nằm yên mà người đứng dậy → huỷ, không báo."""
    det = FallDetector(DEMO_STILL)
    alerts = []
    for t, ps in _pushup_frames():
        if t > 6.0:   # chống đẩy xong thì đứng dậy
            ps = [to_person(1, skeleton(300, 300, 200, 0))]
        alerts += [e for e in det.update(ps, t)[1] if e.kind == "FALL"]
    assert alerts == []
