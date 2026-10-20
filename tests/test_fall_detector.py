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
