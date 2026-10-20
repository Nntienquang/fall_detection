"""Giao diện web Streamlit (TV3).

Chạy:  streamlit run app.py
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import replace
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from src.config import load_config
from src.event_logger import EventLogger
from src.fall_detector import FallDetector
from src.types import State
from src.video import VideoSource
from src.visualize import render

st.set_page_config(page_title="Phát hiện té ngã", page_icon="🚨", layout="wide")
CFG_FILE = "config_demo.yaml" if Path("config_demo.yaml").exists() else "config.yaml"
BASE = load_config(CFG_FILE)


def show_image(box, img):
    """Tương thích cả Streamlit cũ (use_container_width) và mới (width='stretch')."""
    try:
        box.image(img, channels="RGB", width="stretch")
    except (TypeError, Exception):
        box.image(img, channels="RGB", use_container_width=True)


def show_table(box, df):
    try:
        box.dataframe(df, hide_index=True, width="stretch")
    except (TypeError, Exception):
        box.dataframe(df, hide_index=True, use_container_width=True)


@st.cache_resource(show_spinner="Đang tải mô hình YOLO Pose…")
def get_tracker(weights: str, imgsz: int, threads: int):
    from src.pose_tracker import PoseTracker
    tr = PoseTracker(replace(BASE.model, weights=weights, imgsz=imgsz, threads=threads))
    tr.warmup()
    return tr


# ----------------------------- Sidebar ----------------------------- #
st.sidebar.header("⚙️ Cấu hình")
src_kind = st.sidebar.radio("Nguồn video", ["Webcam", "Tải video MP4"])
cam_idx = st.sidebar.number_input("Chỉ số webcam", 0, 5, 0) if src_kind == "Webcam" else None
upload = st.sidebar.file_uploader("Chọn video", type=["mp4", "avi", "mov", "mkv"]) if src_kind != "Webcam" else None

st.sidebar.caption(f"Cấu hình: {CFG_FILE}")
st.sidebar.subheader("Mô hình")
weights = st.sidebar.selectbox("Trọng số", ["yolo11n-pose.pt", "yolo26n-pose.pt", "yolo11s-pose.pt"])
imgsz = st.sidebar.select_slider("Kích thước ảnh vào (imgsz)", [256, 320, 416, 480, 640], value=BASE.model.imgsz)
threads = st.sidebar.slider("Số luồng CPU", 1, 8, BASE.model.threads)

st.sidebar.subheader("Thuật toán")
METHODS = ["C", "B", "A"]
method = st.sidebar.radio("Phương pháp", METHODS, index=METHODS.index(BASE.detector.method),
                          format_func={"A": "A — Bounding box W/H", "B": "B — Góc thân người",
                                       "C": "C — Pose + thời gian (đề xuất)"}.get)
d = BASE.detector
angle = st.sidebar.slider("Ngưỡng góc thân (độ)", 30.0, 85.0, d.angle_thresh, 1.0)
ratio = st.sidebar.slider("Ngưỡng W/H", 0.8, 2.0, d.ratio_thresh, 0.05)
drop = st.sidebar.slider("Độ hạ hông tối thiểu", 0.05, 0.6, d.drop_thresh, 0.01, disabled=method != "C")
confirm = st.sidebar.slider("Thời gian xác nhận (giây)", 0.5, 5.0, d.confirm_time, 0.1)
det_cfg = replace(d, method=method, angle_thresh=angle, ratio_thresh=ratio, drop_thresh=drop, confirm_time=confirm)
save = st.sidebar.checkbox("Lưu video kết quả", value=False)

# ----------------------------- Main ----------------------------- #
st.title("🚨 Hệ thống phát hiện người té ngã qua camera")
st.caption("YOLO Pose + ByteTrack + máy trạng thái theo thời gian · chạy trên CPU")

col_vid, col_info = st.columns([3, 1])
frame_box = col_vid.empty()
with col_info:
    status_box = st.empty()
    m1, m2 = st.columns(2)
    fps_box, people_box = m1.empty(), m2.empty()
    st.markdown("**Lịch sử sự kiện**")
    events_box = st.empty()

run = st.toggle("▶️ Bắt đầu / dừng", value=False)

if run:
    if src_kind != "Webcam" and upload is None:
        st.warning("Hãy tải lên một video trước.")
        st.stop()

    source_arg = str(cam_idx)
    if upload is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix="." + upload.name.split(".")[-1])
        tmp.write(upload.read())
        tmp.close()
        source_arg = tmp.name

    tracker = get_tracker(weights, imgsz, threads)
    tracker.reset()
    detector = FallDetector(det_cfg)
    source = VideoSource(source_arg)
    logger = EventLogger("outputs", meta={"source": src_kind, "method": method})
    rows: list[dict] = []
    fps = 0.0
    try:
        while True:
            t0 = time.perf_counter()
            ok, frame, t = source.read()
            if not ok:
                break
            people = tracker(frame)
            results, events = detector.update(people, t)
            dt = time.perf_counter() - t0
            fps = 1 / dt if fps == 0 else 0.9 * fps + 0.1 / max(dt, 1e-6)

            vis = render(frame, people, results, det_cfg.kp_conf, fps, tracker.last_infer_ms)
            for ev in events:
                logger.log(ev, vis)
                rows.insert(0, {"Thời điểm (s)": round(ev.t, 2), "ID": ev.track_id, "Sự kiện": ev.kind})
                if ev.kind == "FALL":
                    st.toast(f"⚠️ Phát hiện té ngã — người ID {ev.track_id}", icon="🚨")
            if save:
                logger.write_frame(vis, source.fps if not source.is_camera else max(fps, 1.0))

            show_image(frame_box, cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            if any(r.state == State.FALL for r in results):
                status_box.error("🚨 PHÁT HIỆN TÉ NGÃ")
            elif any(r.state == State.SUSPECTED for r in results):
                status_box.warning("⚠️ Nghi ngờ té ngã")
            else:
                status_box.success("✅ Bình thường")
            fps_box.metric("FPS", f"{fps:.1f}")
            people_box.metric("Số người", len(people))
            show_table(events_box, pd.DataFrame(rows))
    finally:
        source.release()
        logger.close()
    st.info(f"Đã xử lý xong. Kết quả lưu tại `{logger.dir}`")
else:
    frame_box.info("Chọn nguồn video ở thanh bên trái rồi bật **Bắt đầu**.")
