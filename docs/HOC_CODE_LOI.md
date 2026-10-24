# Học code lõi — chia cho 4 thành viên

Chỉ học phần **code lõi** liên quan trực tiếp đến môn Thị giác máy tính, khoảng **60–80 dòng mỗi người**. Phần còn lại của repo (công cụ đánh giá, giả lập, giao diện, kiểm tra cài đặt) chỉ cần biết nó dùng để làm gì.

Mỗi phần gồm 4 mục:
1. **Kiến thức môn học**: phần lý thuyết cần ôn lại.
2. **Code**: trích đúng từ repo, kèm giải thích.
3. **Câu hỏi bảo vệ**: các câu giảng viên hay hỏi, có gợi ý trả lời.
4. **Bài tập tay**: tính trên giấy để chắc mình hiểu.

| Người | Phần | File / hàm | Chủ đề môn học |
|---|---|---|---|
| TV1 | Phát hiện & theo dõi người | `src/pose_tracker.py` → `__call__`; `src/fall_detector.py` → `_overlap`, `_drop_duplicates`, `_get_track` | Object detection, IoU, pose estimation, multi-object tracking |
| TV2 | Đặc trưng hình học | `src/features.py` → `_mid`, `torso_angle`, `compute_features`; `src/fall_detector.py` → `is_horizontal` | Hệ tọa độ ảnh, hình học từ keypoint, chuẩn hóa |
| TV3 | Phân tích theo thời gian | `src/fall_detector.py` → `_descent`, `_step` | Xử lý video, thông tin thời gian, máy trạng thái |
| TV4 | Luồng chạy & đánh giá | `main.py` → vòng `while True`; `src/fall_detector.py` → `update`; `src/metrics.py` | Pipeline thị giác, Precision / Recall / F1 |

---

## Phần chung — cả 4 người đều phải nói được

Mỗi khung hình đi qua đúng 5 bước:

```
Camera/MP4 ──► YOLO11n-Pose ──► ByteTrack ──► Đặc trưng ──► Máy trạng thái ──► Cảnh báo
 (khung ảnh)   (khung người +    (ID cho      (góc thân,     (NORMAL →          (ảnh chụp,
               17 keypoint)      từng người)  W/H, hông)     SUSPECTED → FALL)  events.csv)
   TV4             TV1              TV1          TV2               TV3               TV4
```

Câu một dòng: **"Camera → YOLO lấy khung xương → tính góc thân → nếu nằm ngang đủ lâu thì báo ngã."**

Ba phương pháp so sánh trong báo cáo chỉ khác nhau ở **cách quyết định một khung hình là "đang nằm"**:
- **A**: dựa vào tỷ lệ W/H của khung người.
- **B**: dựa vào góc thân người.
- **C**: như B, và trước đó phải có cú hạ hông nhanh.

Kết quả trên tập test: **B tốt nhất, F1 = 0,86**.

---

## TV1 · Phát hiện & theo dõi người

### Kiến thức môn học
- **Phát hiện đối tượng (object detection)**: mô hình trả về bounding box `(x1, y1, x2, y2)` kèm điểm tin cậy cho từng đối tượng.
- **IoU và NMS**: IoU đo độ chồng của hai khung. NMS bỏ các khung trùng nhau, chỉ giữ khung có điểm cao nhất.
- **Ước lượng tư thế (pose estimation)**: YOLO-Pose dự đoán thêm **17 keypoint COCO** cho mỗi người: 0 mũi · 1–4 mắt, tai · 5–6 vai · 7–8 khuỷu · 9–10 cổ tay · 11–12 hông · 13–14 gối · 15–16 cổ chân. Mỗi keypoint có tọa độ `(x, y)` và độ tin cậy.
- **Theo dõi đa đối tượng (tracking)**: ByteTrack ghép khung của khung hình mới với người ở khung hình trước. Nó dùng IoU giữa vị trí dự đoán (bộ lọc Kalman) và vị trí thực tế, nhờ đó **mỗi người giữ một ID** qua các khung hình.

### Code: gọi YOLO + ByteTrack (`PoseTracker.__call__`)
```python
res = self.model.track(
    frame,
    persist=True,                 # ByteTrack nhớ các track giữa các lần gọi → ID ổn định
    tracker=self.cfg.tracker,     # "bytetrack.yaml"
    imgsz=self.cfg.imgsz,         # ảnh được co về 320/416/640 trước khi vào mạng
    conf=self.cfg.conf,           # bỏ khung có độ tin cậy < 0.35
    iou=self.cfg.iou,             # ngưỡng IoU cho NMS
    classes=[0],                  # lớp 0 của COCO = người
    verbose=False,
)[0]

boxes = res.boxes.xyxy.cpu().numpy()        # (N, 4)  khung người
ids   = res.boxes.id...                     # (N,)    ID từ ByteTrack (có thể None)
kps   = res.keypoints.xy.cpu().numpy()      # (N, 17, 2) tọa độ keypoint
kconf = res.keypoints.conf.cpu().numpy()    # (N, 17)  độ tin cậy từng keypoint
```
Sau đó mỗi người được đóng gói thành một `Person(track_id, bbox, score, keypoints, kp_conf)`. Nếu ByteTrack chưa kịp gán ID, người đó nhận một ID âm tạm thời.

### Code: bỏ khung trùng (`_overlap`, `_drop_duplicates`)
```python
def _overlap(a, b):   # phần giao / diện tích khung NHỎ hơn
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih / min(area(a), area(b))
```
Lỗi thật gặp ở video `fall-08`: khi người nằm xuống, YOLO trả **hai khung** cho cùng một người, và khung thứ hai nhận ID mới. Hàm `_drop_duplicates` giữ khung thuộc ID đã có và bỏ khung chồng ≥ 60%.

Ở đây chia cho khung **nhỏ hơn**, không chia cho phần hợp như IoU, vì khung trùng thường nằm **lọt trong** khung kia. Khi đó IoU có thể thấp, nhưng tỷ lệ này vẫn gần 1.

### Code: nối lại ID (`_get_track`)
Khi xuất hiện một ID mới, hàm tìm track vừa mất dấu (trong vòng 3 giây) có tâm khung gần nhất, với khoảng cách chia cho kích thước người. Nếu khoảng cách < 0,6 thì nối ID mới vào lịch sử cũ. Nhờ vậy, ByteTrack đổi ID giữa lúc người đang ngã cũng không làm mất trạng thái "đang nghi ngờ".

### Câu hỏi bảo vệ
| Câu hỏi | Gợi ý trả lời |
|---|---|
| YOLO-Pose khác YOLO thường thế nào? | Ngoài khung và lớp, nó dự đoán thêm 17 keypoint cùng độ tin cậy trong **một lần chạy mạng** (single-stage) |
| Vì sao cần tracking, không chỉ detection? | Thuật toán cần **lịch sử của từng người** qua thời gian. Không có ID thì không biết hông ở khung trước là của ai |
| `imgsz` 320 so với 640? | 640 thấy người nhỏ, người nằm rõ hơn (bắt thêm fall-13, fall-17) nhưng chậm hơn khoảng 2 lần |
| Khi nào ByteTrack đổi ID? | Khi người bị che, ra khỏi khung, hoặc đổi dáng đột ngột (ngã), làm IoU giữa vị trí dự đoán và vị trí thật thấp |

### Bài tập tay
Khung `a = [0, 0, 100, 200]`, khung `b = [10, 20, 60, 200]`. Tính `_overlap(a, b)` và IoU thường. Với ngưỡng 0,6, khung `b` có bị bỏ không?
> Đáp án: giao = 50 × 180 = 9000; khung nhỏ = 9000 nên `_overlap` = **1,0** → bị bỏ. IoU = 9000 / (20000 + 9000 − 9000) = **0,45**, tức là nếu dùng IoU thì đã **không** bắt được khung trùng này.

---

## TV2 · Đặc trưng hình học

### Kiến thức môn học
- **Hệ tọa độ ảnh**: gốc ở **góc trên bên trái**, trục x sang phải, **trục y hướng xuống**. Vì vậy `y` càng lớn thì điểm càng thấp trong ảnh.
- **Hình học từ keypoint**: dùng trung điểm và vector giữa các khớp để mô tả tư thế.
- **Chuẩn hóa**: người ở xa camera trông nhỏ hơn. Muốn ngưỡng dùng được ở mọi khoảng cách thì phải chia cho kích thước người.

### Code: trung điểm hai khớp (`_mid`)
```python
def _mid(kps, conf, a, b, thr):
    ok_a, ok_b = conf[a] >= thr, conf[b] >= thr
    if ok_a and ok_b: return (kps[a] + kps[b]) / 2   # thấy cả hai bên
    if ok_a:          return kps[a]                  # chỉ thấy bên trái
    if ok_b:          return kps[b]                  # chỉ thấy bên phải
    return None                                      # không thấy → không đoán
```
Keypoint có độ tin cậy < 0,35 bị coi là không thấy, vì YOLO vẫn trả tọa độ cho cả khớp bị che.

### Code: góc thân người (`torso_angle`)
```python
sh = _mid(kps, conf, 5, 6, thr)     # trung điểm hai vai
hp = _mid(kps, conf, 11, 12, thr)   # trung điểm hai hông
dx, dy = hp[0] - sh[0], hp[1] - sh[1]
return degrees(atan2(abs(dx), abs(dy)))   # 0° = đứng thẳng, 90° = nằm ngang
```
`atan2(|dx|, |dy|)` là góc giữa trục thân và **phương thẳng đứng**. Lấy trị tuyệt đối nên ngã sang trái hay sang phải đều cho cùng một góc.

### Code: các đặc trưng còn lại (`compute_features`)
- `aspect_ratio = W / H` của khung người: đứng thì khoảng 0,3–0,5, nằm thì > 1.
- `hip_y`: tọa độ y của hông, dùng để đo cú hạ người (phần của TV3).
- `body_len = max(W, H)`: cạnh dài nhất của khung **gần như không đổi** dù đứng hay nằm, nên dùng làm thước đo chuẩn hóa.

### Code: quyết định "đang nằm" (`is_horizontal`)
```python
if method == "A":          return aspect_ratio > ratio_thresh      # phương pháp A
if torso_angle is None:    return ratio_fallback and aspect_ratio > ratio_thresh
return torso_angle > angle_thresh                                  # B và C (demo: 45°)
```
Khi không thấy vai hoặc hông (ví dụ ngồi sát webcam, chỉ thấy đầu và vai), cấu hình demo đặt `ratio_fallback: false` để không đoán bừa. Thay đổi này không làm đổi kết quả trên UR Fall, nhưng tránh được báo nhầm khi demo.

### Câu hỏi bảo vệ
| Câu hỏi | Gợi ý trả lời |
|---|---|
| Vì sao góc thân tốt hơn tỷ lệ W/H? | W/H phụ thuộc hướng nhìn của camera và tư thế tay chân (dang tay cũng làm khung bè ra). Góc vai → hông đo trực tiếp thân người. Kết quả: B 0,86 so với A 0,78 |
| Sao không dùng đầu hoặc cổ chân? | Đầu và cổ chân hay bị che hoặc nhiễu khi ngã. Vai và hông là bốn điểm ổn định nhất |
| Ngã thẳng về phía camera thì sao? | Góc 2D thay đổi ít vì thân đổ theo chiều sâu. Đây là hạn chế đã ghi nhận; cần nhiều camera hoặc dữ liệu độ sâu |

### Bài tập tay
1. Vai ở (100, 100), hông ở (120, 200). Góc thân bằng bao nhiêu? Người đang đứng hay nằm?
2. Vai ở (100, 300), hông ở (200, 310). Góc thân bằng bao nhiêu?
> Đáp án: (1) dx = 20, dy = 100 → atan(0,2) ≈ **11°**, đứng. (2) dx = 100, dy = 10 → atan(10) ≈ **84°**, nằm, vượt ngưỡng 45°.

---

## TV3 · Phân tích theo thời gian

### Kiến thức môn học
- **Video là chuỗi ảnh theo thời gian.** Một khung hình đơn lẻ không phân biệt được *đang ngã* với *đang nằm nghỉ*, mà phải xem **trước và sau**.
- **Lịch sử theo đối tượng**: mỗi ID lưu chuỗi `(thời điểm, hip_y, body_len)` gần nhất.
- **Máy trạng thái hữu hạn (FSM)**: gom nhiều khung hình thành một quyết định ổn định, chống nhiễu từng khung.

### Code: đo cú hạ hông (`_descent`)
```python
win  = [h for h in history if t - h[0] <= 1.0]            # các mẫu trong 1 giây gần nhất
scale = median(body_len trong win)
drop  = (hip_y hiện tại − hip_y cao nhất trong win) / scale  # hạ được bao nhiêu "thân người"
recent = [h for h in win if t - h[0] <= 0.3]               # 0,3 giây gần nhất
speed = (hip_y cuối − hip_y đầu của recent) / scale / Δt     # tốc độ hạ, đơn vị thân/giây
```
Ngã thì hông rơi nhanh (khoảng 0,9–1,3 thân/giây trên dữ liệu thử), còn tự nằm xuống thì chậm (dưới 0,45). Phương pháp C dựa vào sự khác biệt này.

### Code: máy trạng thái (`_step`), phần quan trọng nhất của đề tài
```
NORMAL ──(đang nằm ngang VÀ trước đó từng đứng thẳng)──▶ SUSPECTED
        [C còn đòi: vừa có cú hạ nhanh trong 1,2 s]
SUSPECTED ──(đứng thẳng lại ≥ 0,5 s)──▶ NORMAL                    (chỉ cúi người)
SUSPECTED ──(sau confirm_time, ≥ 70% số khung là nằm ngang)──▶ FALL → phát cảnh báo
FALL ──(đứng thẳng ≥ 2 s)──▶ NORMAL, phát sự kiện RECOVERED
```
Ba chi tiết hay bị hỏi:
- `seen_upright`: người **nằm sẵn** từ đầu video không bị báo, vì chưa từng thấy họ đứng.
- `confirm_ratio = 0.7`: cho phép 30% số khung nhiễu (keypoint lệch) mà vẫn xác nhận được ngã.
- `alert_cooldown = 10 s`: một cú ngã chỉ báo **một lần**.

### Câu hỏi bảo vệ
| Câu hỏi | Gợi ý trả lời |
|---|---|
| Sao không báo ngay khi thấy nằm ngang? | Một khung đơn dễ sai (cúi người, keypoint nhiễu). Chờ `confirm_time` để chắc chắn giúp giảm báo nhầm |
| `confirm_time` lớn hay nhỏ thì sao? | Lớn: ít báo nhầm nhưng báo chậm, dễ sót khi clip ngắn. Nhỏ: ngược lại. Giá trị chọn trên tập val (demo: 0,5 s) |
| Vì sao C (có tốc độ) lại kém B trên test? | C cần thêm điều kiện nên thận trọng hơn: 0 báo nhầm ở ảnh 320 nhưng bỏ sót nhiều hơn trên clip UR Fall bị cắt ngắn |
| Tự nằm xuống giường có bị báo không? | Với B thì có thể. C giảm được nhờ điều kiện tốc độ, nhưng "ngồi phịch xuống rồi nằm" vẫn dễ nhầm (ca adl-10) |

### Bài tập tay
Người có `body_len` = 200 px. Trong 0,3 giây, `hip_y` tăng từ 300 lên 360; trong cả cửa sổ 1 giây, `hip_y` nhỏ nhất là 300. Tính `drop` và `speed`. Với `drop_thresh = 0.22` và `min_descent_speed = 0.6`, đây có phải cú hạ nhanh không?
> Đáp án: drop = 60 / 200 = **0,30**; speed = 60 / 200 / 0,3 = **1,0 thân/giây** → cả hai đều vượt ngưỡng, nên **có**.

---

## TV4 · Luồng chạy & đánh giá

### Kiến thức môn học
- **Pipeline thị giác máy tính**: thu ảnh → tiền xử lý → mô hình → hậu xử lý → quyết định → hiển thị.
- **Tốc độ thời gian thực**: FPS = 1 / thời gian xử lý một khung.
- **Đánh giá**: Precision = TP / (TP + FP); Recall = TP / (TP + FN); F1 = 2PR / (P + R). Chia **val** (để chọn ngưỡng) và **test** (để báo cáo) để kết quả không bị "đẹp giả".

### Code: vòng lặp chính (`main.py`)
```python
while True:
    ok, frame, t = source.read()                  # 1. lấy khung + thời điểm
    if not ok: break
    people = tracker(frame)                       # 2–3. YOLO-Pose + ByteTrack
    results, events = detector.update(people, t)  # 4–5. đặc trưng + máy trạng thái
    dt = time.perf_counter() - t_loop
    fps_ema = 0.9 * fps_ema + 0.1 * (1 / dt)      # FPS làm mượt, không nhảy loạn
    vis = render(frame, people, results, ...)     # vẽ khung xương, nhãn, viền đỏ
    for ev in events: logger.log(ev, vis)         # lưu ảnh + events.csv
    cv2.imshow(...); if phím Q: break
```
Với file MP4, `t = số thứ tự khung / fps`, không phải giờ thật. Nhờ vậy kết quả giống nhau dù máy nhanh hay chậm.

### Code: một khung hình trong thuật toán (`FallDetector.update`)
```python
people = self._drop_duplicates(people)              # bỏ khung trùng
for p in people:
    f  = compute_features(p.bbox, p.keypoints, ...)  # đặc trưng (TV2)
    tr = self._get_track(p.track_id, f, t, ...)      # lịch sử của người đó (TV1)
    res, ev = self._step(p.track_id, tr, f, t)       # máy trạng thái (TV3)
self._cleanup(t)                                     # xóa người đã rời khung > 3 s
```

### Code: chấm điểm theo sự kiện (`match_events`)
```python
for start, end in falls:                       # mỗi cú ngã có nhãn [bắt đầu, kết thúc]
    hits = [cảnh báo a trong [start − 0.5, end + 3.0]]
    if hits: TP += 1; độ trễ = cảnh báo sớm nhất − start
    else:    FN += 1                           # bỏ sót
FP = số cảnh báo không khớp cú ngã nào         # báo nhầm
```
Hệ thống được chấm **theo cú ngã**, không theo từng khung, vì người dùng chỉ quan tâm "có báo đúng cú ngã không". Nhiều cảnh báo cho cùng một cú ngã chỉ tính một lần.

### Câu hỏi bảo vệ
| Câu hỏi | Gợi ý trả lời |
|---|---|
| FPS bao nhiêu, đo thế nào? | ~28 FPS (ảnh 320), ~10–13 FPS (ảnh 640) trên CPU máy nhóm. Đo bằng thời gian một vòng lặp |
| Vì sao chia val / test? | Chọn ngưỡng trên val rồi báo cáo trên test chưa từng thấy. Dùng chung một tập thì con số tốt giả tạo |
| F1 = 0,86 nghĩa là gì? | Trên 10 cú ngã + 10 video sinh hoạt: bắt đúng 9, sót 1, báo nhầm 2 → P = 0,82, R = 0,90 |
| Dữ liệu lấy ở đâu, đủ chưa? | 40 video UR Fall (người thật, camera trong phòng). Chưa đủ để kết luận chắc chắn, vì lệch 1 video là F1 đổi 0,05–0,1; nhóm đang quay thêm |

### Bài tập tay
Cảnh báo lúc **3,0 s; 3,4 s; 20,0 s**. Hai cú ngã có nhãn **[2,5 – 5,0]** và **[40,0 – 42,0]**. Tính TP, FP, FN, Precision, Recall, F1 và độ trễ.
> Đáp án: cảnh báo 3,0 và 3,4 khớp cú ngã thứ nhất → TP = 1, độ trễ = 3,0 − 2,5 = **0,5 s**. Cảnh báo 20,0 không khớp → FP = 1. Cú ngã thứ hai không có cảnh báo → FN = 1. P = R = F1 = **0,5**. (Đây cũng chính là test `test_event_matching`.)

---

## Kế hoạch học (2 buổi)

- [ ] **Buổi 1 (~1,5 giờ)**: mỗi người đọc phần mình, làm bài tập tay, mở code trong VS Code để đối chiếu.
- [ ] **Buổi 2 (~1 giờ, cả nhóm)**: lần lượt từng người trình bày phần mình 5 phút theo đúng thứ tự luồng chạy TV4 → TV1 → TV2 → TV3; 3 bạn còn lại hỏi bằng bảng câu hỏi của người đó.
- [ ] Cả nhóm đọc chéo hàm `_step` (phần TV3), vì đây là phần giảng viên hỏi nhiều nhất.

Khi bị hỏi điều chưa biết, trả lời thẳng "phần này em chưa kiểm chứng" và nói cách sẽ kiểm chứng, đừng đoán.
