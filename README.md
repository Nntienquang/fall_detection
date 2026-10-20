# Phát hiện người té ngã qua camera

Đồ án môn Thị giác máy tính — YOLO Pose + ByteTrack + máy trạng thái theo thời gian, chạy trên **CPU**, đầu vào **webcam hoặc video MP4**.

```
Camera / MP4 ─▶ YOLO11n-Pose (17 keypoints) ─▶ ByteTrack (ID) ─▶ Đặc trưng (góc thân, W/H, vận tốc hạ hông)
            ─▶ Máy trạng thái NORMAL → SUSPECTED → FALL ─▶ Cảnh báo + ảnh chụp + events.csv
```

> **Mỗi thành viên làm theo [SETUP.md](SETUP.md)** rồi chạy `python tools/check_setup.py --name "Ho Ten"` để có báo cáo cài đặt.

## 1. Cài đặt (Windows 10/11, Python 3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # bị chặn thì dùng: .\.venv\Scripts\python.exe thay cho python
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # mong đợi: ... False
```

Lần chạy đầu Ultralytics tự tải `yolo11n-pose.pt` (~6 MB) — cần mạng.

## 2. Chạy

| Mục đích | Lệnh |
|---|---|
| Webcam | `python main.py --source 0` (không nhận thì `--source 1`) |
| Video MP4 + lưu video kết quả | `python main.py --source data/videos/V05.mp4 --save-video` |
| Benchmark CPU (không mở cửa sổ) | `python main.py --source data/videos/V05.mp4 --no-display --max-frames 300 --imgsz 320 --threads 4` |
| So sánh phương pháp | thêm `--method A` / `B` / `C` |
| Giao diện web | `streamlit run app.py` |
| Unit test | `python -m pytest -q` |

Kết quả mỗi lần chạy nằm trong `outputs/run_YYYYmmdd_HHMMSS/`: `events.csv`, `fall_*.jpg`, `annotated.mp4`, `run_config.json`.

## 3. Ba phương pháp (phần thí nghiệm của báo cáo)

| | Điều kiện "nằm" | Điều kiện nghi ngờ | Vai trò |
|---|---|---|---|
| **A** | W/H bbox > `ratio_thresh` | nằm | Baseline |
| **B** | góc thân > `angle_thresh` (thiếu vai/hông → dùng W/H) | nằm | Cải tiến |
| **C** | như B | nằm **và** vừa có cú hạ hông nhanh (độ hạ ≥ `drop_thresh`, vận tốc ≥ `min_descent_speed`) | Đề xuất |

Chung cho cả 3: phải nằm liên tục `confirm_time` giây mới báo FALL; người nằm sẵn từ đầu không bị báo; đứng thẳng lại `recover_time` giây → RECOVERED; không báo lặp trong `alert_cooldown`; tự nối lại lịch sử khi ByteTrack đổi ID lúc người ngã.

Mọi ngưỡng nằm trong `config.yaml` và **chỉ là giá trị khởi đầu**.

## 4. Quy trình đánh giá (TV4)

1. Quay video vào `data/videos/`, ghi nhãn vào `data/labels.csv` (mẫu: `data/labels_template.csv`).
   Mỗi cú ngã 1 dòng (`fall_start`, `fall_end` tính bằng giây); video không có cú ngã để trống 2 cột đó.
   Cột `split`: `val` (dùng để chỉnh ngưỡng) hoặc `test` (chỉ dùng để báo cáo). **Không** dùng chung video giữa 2 tập.
2. Chạy YOLO một lần cho tất cả video (chậm, chạy qua đêm cũng được):
   `python tools/cache_poses.py --labels data/labels.csv --videos data/videos`
3. Tìm ngưỡng trên tập val (vài giây):
   `python tools/tune.py --method C --split val` → `config.tuned.yaml`
   (làm tương tự `--method A`, `--method B` nếu muốn so sánh công bằng, lưu ra file khác bằng `--out`)
4. Đánh giá cuối trên tập test:
   `python tools/evaluate.py --config config.tuned.yaml --split test` → `results/metrics.md` (dán vào báo cáo)
5. Biểu đồ đặc trưng cho báo cáo:
   `python tools/plot_features.py --video V05_fall_forward.mp4`

Chỉ số: Precision, Recall, F1 theo **sự kiện**; số cảnh báo sai mỗi giờ; độ trễ phát hiện trung bình; tốc độ YOLO (ms/frame).

**Chạy thử khi chưa có video:** `python tools/simulate.py` tạo dữ liệu người que giả lập, rồi chạy bước 3–4 với
`--labels data/labels_sim.csv --cache cache_sim`. Số liệu giả lập chỉ để kiểm tra công cụ, **không đưa vào báo cáo**.

## 5. Cấu trúc & người phụ trách

```
main.py                 CLI chạy webcam/MP4                                  TV3
app.py                  Giao diện Streamlit                                  TV3
config.yaml             Toàn bộ tham số                                      TV2
src/types.py            Cấu trúc dữ liệu chung (Person, State, FallEvent)    Cả nhóm (chốt ngày 1)
src/pose_tracker.py     YOLO Pose + ByteTrack                                TV1
src/video.py            Đọc webcam (luôn lấy frame mới nhất) / MP4           TV3
src/features.py         Góc thân, W/H, vị trí hông                           TV2
src/fall_detector.py    Máy trạng thái A/B/C, nối ID                         TV2
src/visualize.py        Vẽ skeleton, nhãn, cảnh báo                          TV3
src/event_logger.py     events.csv, ảnh chụp, video                          TV3
src/metrics.py          Ghép sự kiện, P/R/F1, độ trễ                         TV4
src/replay.py           Cache kết quả YOLO, đọc nhãn                         TV4
src/simulate.py         Sinh dữ liệu giả lập                                 TV2
tools/cache_poses.py    Chạy YOLO 1 lần cho mọi video                        TV1/TV4
tools/tune.py           Grid search ngưỡng trên val                          TV2
tools/evaluate.py       So sánh A/B/C trên test                              TV4
tools/plot_features.py  Biểu đồ đặc trưng theo thời gian                     TV2/TV4
tests/                  17 unit test                                         TV2/TV4
```

## 6. Lỗi thường gặp

- **FPS thấp / webcam trễ:** giảm `--imgsz 256`, thử `--threads` = số nhân thật của CPU; `src/video.py` đã bỏ frame cũ nên không bị trễ tích luỹ.
- **`lap` lỗi khi cài:** `pip install lap` riêng, hoặc đổi `tracker: botsort.yaml` trong config.
- **Mất keypoint khi người nằm xa camera:** thử `--imgsz 416`, đặt camera cao 1.5–2.5 m nhìn chéo xuống.
- **Báo nhầm khi nằm/cúi:** dùng phương pháp C và chỉnh ngưỡng bằng `tools/tune.py`.

## 7. Hạn chế đã biết (ghi vào báo cáo)

- Ngưỡng hình học phụ thuộc góc đặt camera; cú ngã về phía camera làm góc thân thay đổi ít.
- Người bị che khuất phần lớn hoặc ngã ra ngoài khung hình có thể bị bỏ sót.
- Nối ID chỉ dựa vào khoảng cách tâm, có thể nối nhầm khi nhiều người đứng sát nhau.
- Mô hình chưa fine-tune trên dữ liệu té ngã; nằm ngang trên sàn đôi khi cho keypoint kém tin cậy.
