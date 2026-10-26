# Học code lõi & ôn bảo vệ — 4 thành viên

Mỗi người học **một phần code lõi** (khoảng 60–80 dòng), gắn với một chủ đề của môn Thị giác máy tính. Phần còn lại của repo (công cụ đánh giá, giả lập, giao diện Streamlit, kiểm tra cài đặt) chỉ cần biết nó dùng để làm gì.

Mỗi phần có 5 mục:
1. **Kiến thức môn học**: lý thuyết cần ôn.
2. **Code**: trích đúng từ repo, kèm giải thích.
3. **Câu hỏi bảo vệ**: các câu giảng viên có thể hỏi, kèm gợi ý trả lời.
4. **Bài tập tay**: tính trên giấy để chắc mình hiểu.
5. **Yêu cầu sửa tại chỗ**: giảng viên bảo "em sửa cho thầy xem…", thì sửa ở đâu, gõ gì, kiểm tra thế nào. Mọi đoạn sửa đã được chạy thử trên repo.

| Người | Phần | File / hàm | Chủ đề môn học |
|---|---|---|---|
| **TV1** | Phát hiện & theo dõi người | `src/pose_tracker.py` → `__call__`; `src/fall_detector.py` → `_overlap`, `_drop_duplicates`, `_get_track` | Object detection, IoU/NMS, pose estimation, tracking |
| **TV2** | Đặc trưng hình học | `src/features.py` → `_mid`, `torso_angle`, `compute_features`; `src/fall_detector.py` → `is_horizontal` | Hệ tọa độ ảnh, vector, góc, chuẩn hóa |
| **TV3** | Phân tích theo thời gian | `src/fall_detector.py` → `_descent`, `_step` | Video = chuỗi ảnh, đặc trưng chuyển động, máy trạng thái |
| **TV4** | Luồng chạy & đánh giá | `main.py` → vòng `while True`; `src/fall_detector.py` → `update`; `src/metrics.py` | Pipeline, thời gian thực, Precision / Recall / F1 |

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

**Phần nào nhóm dùng sẵn, phần nào nhóm tự làm** (chắc chắn bị hỏi):
- Dùng sẵn, không huấn luyện lại: YOLO11n-Pose (bước 2) và ByteTrack (bước 3), qua thư viện Ultralytics.
- Nhóm tự xây dựng: đặc trưng hình học (bước 4), máy trạng thái theo thời gian (bước 5), xử lý khung trùng và nối lại ID, khung đánh giá theo sự kiện, so sánh 3 phương pháp.

Ba phương pháp chỉ khác nhau ở **cách quyết định một khung hình là "đang nằm"**:
- **A**: tỷ lệ W/H của khung người > ngưỡng.
- **B**: góc thân người (vai → hông) > ngưỡng.
- **C**: như B, và trước đó phải có cú hạ hông nhanh.

**Số liệu cần thuộc:**

| | Kết quả |
|---|---|
| Dữ liệu | UR Fall: 40 video người thật. val 10 (chọn ngưỡng), test_cu 10, test 20 (10 ngã + 10 sinh hoạt, chưa dùng để chỉnh) |
| Tập test, ảnh 640 | **B tốt nhất: F1 = 0,86** (bắt 9/10 cú ngã, 2 báo nhầm); A 0,78; C 0,74 |
| Tập test, ảnh 320 | B 0,78; A 0,71; C 0,67 (C không báo nhầm lần nào nhưng sót nhiều) |
| Tốc độ CPU | ảnh 640 ~13 FPS; ảnh 416 ~21 FPS (36 ms/khung) |
| Webcam thật | ngã lên đệm 2/2 lần được báo, đứng dậy thì về bình thường |
| Hạn chế đã biết | giữ yên tư thế quỳ chống tay/chống đẩy lâu vẫn bị báo (đã giảm bằng luật nằm yên); ngã thẳng về phía camera; người bị che; người ra khỏi khung (fall-19) |

---

## TV1 · Phát hiện & theo dõi người

### Kiến thức môn học
- **Phát hiện đối tượng (object detection)**: mô hình trả về bounding box `(x1, y1, x2, y2)` kèm điểm tin cậy cho từng đối tượng.
- **IoU và NMS**: IoU = phần giao / phần hợp của hai khung. NMS bỏ các khung trùng nhau, chỉ giữ khung có điểm cao nhất.
- **Ước lượng tư thế (pose estimation)**: YOLO-Pose dự đoán thêm **17 keypoint COCO** cho mỗi người: 0 mũi · 1–4 mắt, tai · 5–6 vai · 7–8 khuỷu · 9–10 cổ tay · 11–12 hông · 13–14 gối · 15–16 cổ chân. Mỗi keypoint có tọa độ `(x, y)` và độ tin cậy.
- **Theo dõi đa đối tượng (tracking)**: ByteTrack ghép khung của khung hình mới với người ở khung hình trước. Nó dùng IoU giữa vị trí dự đoán (bộ lọc Kalman) và vị trí thực tế, nhờ đó **mỗi người giữ một ID** qua các khung hình. ByteTrack dùng cả các khung có điểm thấp ở vòng ghép thứ hai, nên ít mất dấu người bị che một phần.

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

**imgsz**: YOLO thu nhỏ khung sao cho cạnh dài bằng `imgsz` (giữ tỷ lệ, đệm viền), chạy mạng, rồi quy đổi tọa độ về ảnh gốc. Giá trị nên là bội số của 32, vì mạng giảm kích thước theo bước 32. Ảnh lớn thì chính xác hơn nhưng chậm gần theo bình phương: 640 nặng khoảng 4 lần 320.

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
| YOLO-Pose khác YOLO thường thế nào? | Ngoài khung và lớp, nó dự đoán thêm 17 keypoint cùng độ tin cậy trong **một lần chạy mạng** (single-stage), không cần chạy riêng một mô hình tư thế cho từng người |
| Vì sao chọn bản `n` (nano)? | Nhỏ nhất, chạy được thời gian thực trên CPU. Bản `s`/`m` chính xác hơn nhưng chậm gấp 2–4 lần |
| Nhóm có huấn luyện YOLO không? | Không. Dùng trọng số huấn luyện sẵn trên COCO. Đóng góp của nhóm là bước 4–5 phía sau |
| Vì sao cần tracking, không chỉ detection? | Thuật toán cần **lịch sử của từng người** qua thời gian. Không có ID thì không biết hông ở khung trước là của ai, nhất là khi có nhiều người |
| `imgsz` 320 so với 640? | 640 thấy người nhỏ, người nằm rõ hơn (bắt thêm fall-13, fall-17), F1 0,78 → 0,86, nhưng chậm hơn khoảng 2 lần. Webcam demo dùng 416 để cân bằng |
| Khi nào ByteTrack đổi ID? | Khi người bị che, ra khỏi khung, hoặc đổi dáng đột ngột (ngã), làm IoU giữa vị trí dự đoán và vị trí thật thấp. Nhóm xử lý bằng `_get_track` |
| `conf` 0,35 nghĩa là gì, tăng lên thì sao? | Bỏ khung người có độ tin cậy < 0,35. Tăng lên thì ít khung rác hơn, nhưng dễ mất người đang nằm, vì tư thế nằm hiếm trong dữ liệu huấn luyện nên điểm thấp |
| Keypoint bị che thì YOLO trả gì? | Vẫn trả tọa độ (đoán) nhưng độ tin cậy thấp. Nhóm bỏ keypoint có độ tin cậy < 0,35 (`kp_conf`) |

### Bài tập tay
Khung `a = [0, 0, 100, 200]`, khung `b = [10, 20, 60, 200]`. Tính `_overlap(a, b)` và IoU thường. Với ngưỡng 0,6, khung `b` có bị bỏ không?
> Đáp án: giao = 50 × 180 = 9000; khung nhỏ = 9000 nên `_overlap` = **1,0** → bị bỏ. IoU = 9000 / (20000 + 9000 − 9000) = **0,45**, tức là nếu dùng IoU thì đã **không** bắt được khung trùng này.

### Yêu cầu sửa tại chỗ
**1.1 · "Tăng ngưỡng tin cậy phát hiện người lên 0,5."** Mở `config_demo.yaml`, sửa `conf: 0.35` thành `conf: 0.5`, rồi chạy lại webcam. Giải thích: ít khung rác hơn, nhưng người đang nằm hay bị mất khung.

**1.2 · "Đổi thuật toán tracking khác."** Trong `config_demo.yaml`, sửa `tracker: bytetrack.yaml` thành `tracker: botsort.yaml`. Ultralytics có sẵn BoT-SORT; thuật toán này thêm bù chuyển động camera nên chậm hơn ByteTrack một chút.

**1.3 · "Bỏ qua người ở quá xa, khung nhỏ hơn 60 px."**
Trong `src/config.py`, class `ModelConfig`, thêm một dòng:
```python
    min_person_px: float = 60.0   # bỏ người có cạnh dài khung < 60 px
```
Trong `src/pose_tracker.py`, hàm `__call__`, ngay dưới `for i in range(len(boxes)):` thêm:
```python
            w, h = boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]
            if max(w, h) < self.cfg.min_person_px:   # người quá nhỏ (ở quá xa) → bỏ qua
                continue
```
Dùng `max(w, h)` chứ không dùng chiều cao, vì người **đang nằm** có chiều cao khung rất nhỏ. Nếu lọc theo chiều cao thì sẽ bỏ mất chính người bị ngã.

---

## TV2 · Đặc trưng hình học

### Kiến thức môn học
- **Hệ tọa độ ảnh**: gốc ở **góc trên bên trái**, trục x sang phải, **trục y hướng xuống**. Vì vậy `y` càng lớn thì điểm càng thấp trong ảnh.
- **Hình học từ keypoint**: dùng trung điểm, vector và góc giữa các khớp để mô tả tư thế.
- **Chuẩn hóa**: người ở xa camera trông nhỏ hơn. Muốn ngưỡng dùng được ở mọi khoảng cách thì phải chia cho kích thước người. Góc thì tự nó không phụ thuộc khoảng cách.

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
if edge_guard and truncated: return False                          # người bị cắt ở mép ảnh → không đoán
if method == "A":          return aspect_ratio > ratio_thresh      # phương pháp A
if torso_angle is None:    return ratio_fallback and aspect_ratio > ratio_thresh
return torso_angle > angle_thresh                                  # B và C (demo: 45°)
```
Hai luật chống báo nhầm khi demo webcam, đều phát hiện từ lỗi thật và đều **không đổi kết quả trên UR Fall**:
- `ratio_fallback: false`: không thấy vai hoặc hông (ngồi sát webcam, chỉ thấy đầu và vai) thì không đoán bừa bằng W/H.
- `edge_guard: true`: khung người chạm mép ảnh **và** không thấy cổ chân (người đứng sát webcam cúi xuống, bị cắt mất nửa dưới) thì không kết luận "nằm". Hàm `_truncated` trong `src/fall_detector.py`.

### Câu hỏi bảo vệ
| Câu hỏi | Gợi ý trả lời |
|---|---|
| Vì sao góc thân tốt hơn tỷ lệ W/H? | W/H phụ thuộc hướng nhìn của camera và tư thế tay chân (dang tay cũng làm khung bè ra). Góc vai → hông đo trực tiếp thân người. Kết quả: B 0,86 so với A 0,78 |
| Sao không dùng đầu hoặc cổ chân? | Đầu và cổ chân hay bị che hoặc nhiễu khi ngã. Vai và hông là bốn điểm ổn định nhất. Nếu dùng cổ chân thì ngồi xếp chân cũng làm góc lệch (xem yêu cầu 2.3) |
| Vì sao dùng `atan2` mà không dùng `atan(dx/dy)`? | `atan2` không lỗi khi `dy = 0` (người nằm ngang hẳn) và xử lý đúng mọi góc phần tư |
| Ngưỡng 45° lấy ở đâu? | Grid search trên tập val (thử 45–70°), chọn F1 cao nhất. Không chỉnh trên test |
| Ngã thẳng về phía camera thì sao? | Góc 2D thay đổi ít vì thân đổ theo chiều sâu. Đây là hạn chế đã ghi nhận; cần camera đặt cao, nhiều camera, hoặc dữ liệu độ sâu |
| Chống đẩy sao lại bị báo? | Trong một khung hình, chống đẩy và nằm sau ngã đều có thân nằm ngang: hình học 2D giống nhau, chỉ khác ý định. Nhóm đã thử luật "chống tay" nhưng không đủ trên webcam thật; hướng xử lý là kiểm tra bất động sau ngã hoặc học theo chuỗi thời gian |
| Người đứng sát camera cúi xuống có bị báo không? | Trước đây có (góc 44–47°, chỉ thấy nửa trên). Nay có `edge_guard`: chạm mép ảnh mà không thấy cổ chân thì không kết luận nằm. Đánh đổi: người ngã mà chân ra ngoài khung sẽ không được báo |
| Vì sao chuẩn hóa theo `max(W, H)` mà không theo chiều cao? | Người ngã thì chiều cao khung giảm mạnh, còn cạnh dài nhất gần như giữ nguyên chiều dài cơ thể |

### Bài tập tay
1. Vai ở (100, 100), hông ở (120, 200). Góc thân bằng bao nhiêu? Người đang đứng hay nằm?
2. Vai ở (100, 300), hông ở (200, 310). Góc thân bằng bao nhiêu?
3. Hông (0, 0), gối (0, 100), cổ chân (100, 100). Góc tại đầu gối bằng bao nhiêu (dùng tích vô hướng)?
> Đáp án: (1) dx = 20, dy = 100 → atan(0,2) ≈ **11°**, đứng. (2) dx = 100, dy = 10 → atan(10) ≈ **84°**, nằm, vượt ngưỡng 45°. (3) v1 = gối→hông = (0, −100), v2 = gối→cổ chân = (100, 0); tích vô hướng = 0 → cos = 0 → **90°** (đầu gối gập vuông, như đang ngồi).

### Yêu cầu sửa tại chỗ
**2.1 · "Đổi ngưỡng góc 45° thành 60°, kết quả thay đổi thế nào?"** Sửa `angle_thresh: 45.0` trong `config_demo.yaml`, rồi chạy:
```
python tools/evaluate.py --config config_demo.yaml --split test --methods B --hold-last 2
```
Kết quả đã chạy thử (ảnh 320): 45° → bắt 7, sót 3, nhầm 2, F1 0,74. 60° → bắt 6, sót 4, nhầm 1, F1 0,71. Ngưỡng cao hơn thì ít báo nhầm nhưng sót nhiều hơn. Thêm `--cache cache_640` nếu muốn chạy trên dữ liệu ảnh 640.

**2.2 · "Tính thêm góc đầu gối và hiển thị lên màn hình."** Đây là bài hình học hay bị yêu cầu.
Trong `src/features.py`, thêm hàm này ngay **trên** `def compute_features(`:
```python
def joint_angle(kps, conf, a: int, b: int, c: int, thr: float):
    """Góc tại khớp b (độ) tạo bởi 3 điểm a-b-c, vd hông-gối-cổ chân. 180 = duỗi thẳng."""
    if min(conf[a], conf[b], conf[c]) < thr:
        return None
    v1, v2 = kps[a] - kps[b], kps[c] - kps[b]
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))
```
Trong `src/visualize.py`, sửa dòng import thành `from .features import SKELETON, joint_angle`. Trong `draw_person`, ngay **dưới** dòng `label = f"ID {p.track_id} {r.state.value} | goc ...` thêm:
```python
        knee = joint_angle(p.keypoints, p.kp_conf, 11, 13, 15, kp_thr)   # hông trái - gối trái - cổ chân trái
        if knee is not None:
            label += f" | goi {knee:.0f}"
```
Chạy webcam: đứng thì khoảng 170–180°, ngồi khoảng 90°. Giải thích: `cos θ = (v1 · v2) / (|v1| |v2|)`; kẹp giá trị vào [−1, 1] để `acos` không lỗi do sai số làm tròn.

**2.3 · "Đo góc thân từ vai đến cổ chân thay vì đến hông."** Trong `torso_angle`, đổi `L_HIP, R_HIP` thành `L_ANKLE, R_ANKLE`. Kết quả đã thử trên khung xương mô phỏng: đứng 0°, nằm 90° (không đổi), nhưng **ngồi** ra 24° thay vì 0°, vì chân gập làm lệch trục. Đây là lý do nhóm chọn hông.

---

## TV3 · Phân tích theo thời gian

### Kiến thức môn học
- **Video là chuỗi ảnh theo thời gian.** Một khung hình đơn lẻ không phân biệt được *đang ngã* với *đang nằm nghỉ*, mà phải xem **trước và sau**.
- **Lịch sử theo đối tượng**: mỗi ID lưu chuỗi `(thời điểm, hip_y, body_len)` gần nhất.
- **Đặc trưng chuyển động**: độ dịch chuyển và vận tốc tính từ hiệu tọa độ giữa các khung, chia cho thời gian.
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
        [demo webcam: còn đòi vai NẰM YÊN — dịch < 12% thân người trong 1 s (`_motion`)]
FALL ──(đứng thẳng ≥ 2 s)──▶ NORMAL, phát sự kiện RECOVERED
```
Đo mức cử động (`_motion`): biên độ dịch chuyển lớn nhất của trung điểm vai trong 1 s gần nhất (max − min theo x hoặc y), chia chiều dài người. Nằm yên thật chỉ rung 0,04–0,08 (nhiễu keypoint); chống đẩy khoảng 0,3. Đây là cách xử lý chống đẩy, bò.

Bốn chi tiết hay bị hỏi:
- `seen_upright`: người **nằm sẵn** từ đầu video không bị báo, vì chưa từng thấy họ đứng.
- `confirm_ratio = 0.7`: cho phép 30% số khung nhiễu (keypoint lệch) mà vẫn xác nhận được ngã.
- `alert_cooldown = 10 s`: một cú ngã chỉ báo **một lần**.
- Tư thế "không rõ" (người bị cắt ở mép ảnh): không tính là nằm, cũng không tính là đã đứng dậy, để không huỷ nhầm.

### Câu hỏi bảo vệ
| Câu hỏi | Gợi ý trả lời |
|---|---|
| Sao không báo ngay khi thấy nằm ngang? | Một khung đơn dễ sai (cúi người, keypoint nhiễu). Chờ `confirm_time` để chắc chắn giúp giảm báo nhầm |
| `confirm_time` lớn hay nhỏ thì sao? | Lớn: ít báo nhầm nhưng báo chậm, dễ sót khi clip ngắn. Nhỏ: ngược lại. Đã thử trên tập test (ảnh 320): 0,5 s → F1 0,74; 1,0 s → 0,71; 2,0 s → 0,53 |
| Vì sao chia cho `body_len` khi tính tốc độ? | Người ở xa camera di chuyển ít pixel hơn. Đổi sang đơn vị "thân người/giây" thì ngưỡng dùng được ở mọi khoảng cách |
| Vì sao đo tốc độ trong 0,3 s mà không phải cả 1 s? | Trung bình cả 1 s bị pha loãng bởi lúc đứng yên trước khi ngã. 0,3 s bắt được đúng lúc rơi nhanh nhất |
| Vì sao C (có tốc độ) lại kém B trên test? | C cần thêm điều kiện nên thận trọng hơn: không báo nhầm lần nào ở ảnh 320 nhưng bỏ sót nhiều hơn, vì clip UR Fall bị cắt ngay sau khi chạm sàn |
| Tự nằm xuống giường có bị báo không? | Với B thì có thể. C giảm được nhờ điều kiện tốc độ, nhưng "ngồi phịch xuống rồi nằm" vẫn dễ nhầm (ca adl-10) |
| Người nằm sẵn trên sàn từ đầu có bị báo không? | Không, nhờ `seen_upright`. Bỏ điều kiện này thì phương pháp B báo ngay sau 0,5 s (yêu cầu 3.3) |
| Chống đẩy, bò có bị báo không? | Bản demo chờ người **nằm yên** 1 s mới báo; chống đẩy thì vai lên xuống liên tục nên không báo. Đánh đổi: báo chậm hơn ~1 s, và giữ yên tư thế quỳ/chống đẩy quá lâu vẫn bị báo |
| Sao luật nằm yên không bật trong thí nghiệm chính? | Clip UR Fall hết 0,3–1 s sau khi chạm sàn, chưa đủ 1 s nằm yên nên đánh giá không công bằng (xem PHAN_TICH_KET_QUA.md). Cần video tự quay để đo |
| Người ngã rồi nằm mãi không dậy thì sao? | Hiện chỉ báo một lần. Có thể thêm cảnh báo mức 2 khi nằm quá N giây (yêu cầu 3.2) |

### Bài tập tay
Người có `body_len` = 200 px. Trong 0,3 giây, `hip_y` tăng từ 300 lên 360; trong cả cửa sổ 1 giây, `hip_y` nhỏ nhất là 300. Tính `drop` và `speed`. Với `drop_thresh = 0.22` và `min_descent_speed = 0.6`, đây có phải cú hạ nhanh không?
> Đáp án: drop = 60 / 200 = **0,30**; speed = 60 / 200 / 0,3 = **1,0 thân/giây** → cả hai đều vượt ngưỡng, nên **có**.

### Yêu cầu sửa tại chỗ
**3.1 · "Cho hệ thống chờ lâu hơn rồi mới báo."** Sửa `confirm_time: 0.5` thành `1.0` trong `config_demo.yaml`. Giải thích đánh đổi bằng số liệu trong bảng câu hỏi ở trên.

**3.2 · "Nếu người ngã nằm quá 5 giây chưa dậy thì báo thêm một mức nguy hiểm hơn."** Bài sửa máy trạng thái hay gặp nhất.
- Trong `src/config.py`, class `DetectorConfig`, thêm dòng `long_lie_time: float = 5.0`.
- Trong `src/fall_detector.py`, class `_Track`, thêm dòng `long_alerted: bool = False`.
- Trong `_step`, ngay dưới `self._set(tr, State.FALL, t)` thêm `tr.long_alerted = False`.
- Trong nhánh `elif tr.state == State.FALL:`, sau khối `if upright_for >= c.recover_time: ...` thêm:
```python
            elif not tr.long_alerted and t - tr.state_since >= c.long_lie_time:
                tr.long_alerted = True
                event = FallEvent(t, tid, "LONG_LIE", f.torso_angle, f.aspect_ratio, "nam lau chua day")
```
Đã chạy thử trên cú ngã mô phỏng: FALL lúc 4,6 s, LONG_LIE lúc 9,6 s (đúng 5 s sau). Sự kiện được in ra màn hình và ghi vào `events.csv`. Phần đánh giá chỉ đếm sự kiện `FALL` nên không bị ảnh hưởng.

**3.3 · "Bỏ điều kiện đã từng đứng thẳng thì sao?"** Trong `_step`, đổi `if lying and tr.seen_upright:` thành `if lying:`. Chạy thử với phương pháp B: người nằm sẵn từ đầu bị báo FALL ngay sau 0,5 s. Sửa xong nhớ hoàn tác (xem mục cuối).

---

## TV4 · Luồng chạy & đánh giá

### Kiến thức môn học
- **Pipeline thị giác máy tính**: thu ảnh → tiền xử lý → mô hình → hậu xử lý → quyết định → hiển thị.
- **Tốc độ thời gian thực**: FPS = 1 / thời gian xử lý một khung. Camera thường 30 FPS; hệ thống chỉ cần ≥ 10 FPS để không bỏ lỡ cú ngã (ngã kéo dài khoảng 0,5–1 s).
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
- Với file MP4, `t = số thứ tự khung / fps`, không phải giờ thật. Nhờ vậy kết quả giống nhau dù máy nhanh hay chậm.
- Với webcam, một luồng riêng đọc camera và **chỉ giữ khung mới nhất** (`src/video.py`). Nếu máy xử lý chậm, khung cũ bị bỏ, nên hình không bị trễ dần.

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
| FPS bao nhiêu, đo thế nào? | Webcam máy nhóm: ~21 FPS ở ảnh 416, ~13 FPS ở ảnh 640, trên CPU. Đo bằng thời gian một vòng lặp, làm mượt bằng trung bình trượt |
| Vì sao chia val / test? | Chọn ngưỡng trên val rồi báo cáo trên test chưa từng thấy. Dùng chung một tập thì con số tốt giả tạo |
| F1 = 0,86 nghĩa là gì? | Trên 10 cú ngã + 10 video sinh hoạt: bắt đúng 9, sót 1, báo nhầm 2 → P = 0,82, R = 0,90 |
| Vì sao không dùng accuracy? | Dữ liệu thực tế lệch: phần lớn thời gian không có ai ngã. Hệ thống "không bao giờ báo" vẫn có accuracy rất cao nhưng vô dụng. Precision/Recall đo đúng điều quan tâm |
| Bỏ sót và báo nhầm, cái nào nguy hiểm hơn? | Bỏ sót nguy hiểm hơn (người ngã không được giúp), nên ưu tiên Recall. Nhưng báo nhầm nhiều thì người dùng sẽ tắt hệ thống, nên vẫn phải giữ Precision |
| "Giữ frame cuối 2 s" là gì, có gian lận không? | Clip UR Fall bị cắt ngay sau khi chạm sàn, chưa đủ thời gian để xác nhận. Lặp lại khung cuối 2 s để mô phỏng người vẫn nằm; áp dụng **giống nhau** cho mọi video và mọi phương pháp, và có báo cả kết quả không giữ |
| Dữ liệu lấy ở đâu, đủ chưa? | 40 video UR Fall (người thật, camera trong phòng, giấy phép CC BY-NC-SA 4.0). Chưa đủ để kết luận chắc chắn, vì lệch 1 video là F1 đổi 0,05–0,1 |
| Triển khai thực tế cần thêm gì? | Gửi thông báo (Telegram/SMS), camera đặt cao bao quát phòng, chạy liên tục, xử lý ánh sáng yếu, bảo vệ quyền riêng tư (chỉ lưu khung xương, không lưu hình) |

### Bài tập tay
Cảnh báo lúc **3,0 s; 3,4 s; 20,0 s**. Hai cú ngã có nhãn **[2,5 – 5,0]** và **[40,0 – 42,0]**. Tính TP, FP, FN, Precision, Recall, F1 và độ trễ.
> Đáp án: cảnh báo 3,0 và 3,4 khớp cú ngã thứ nhất → TP = 1, độ trễ = 3,0 − 2,5 = **0,5 s**. Cảnh báo 20,0 không khớp → FP = 1. Cú ngã thứ hai không có cảnh báo → FN = 1. P = R = F1 = **0,5**. (Đây cũng chính là test `test_event_matching`.)

### Yêu cầu sửa tại chỗ
**4.1 · "Phát âm thanh khi phát hiện ngã."** Trong `main.py`, ngay dưới dòng `import cv2` thêm:
```python
try:
    import winsound          # chỉ có trên Windows
except ImportError:
    winsound = None
```
Trong vòng `for ev in events:`, sau dòng `print(...)` thêm:
```python
                if ev.kind == "FALL" and winsound:
                    winsound.MessageBeep(winsound.MB_ICONHAND)   # phát tiếng báo, không làm đứng chương trình
```
Dùng `MessageBeep` chứ không dùng `Beep(…)`, vì `Beep` chặn chương trình trong lúc kêu, làm video bị khựng.

**4.2 · "Hiện tổng số lần ngã lên màn hình."** Trong `main.py`, ngay **sau** vòng `for ev in events:` (cùng mức thụt lề với `for`) thêm:
```python
            if vis is not None:
                n_falls = sum(e.kind == "FALL" for e in logger.events)
                cv2.putText(vis, f"So lan nga: {n_falls}", (8, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2, cv2.LINE_AA)
```
Màu trong OpenCV là **BGR**, nên `(0, 0, 255)` là màu đỏ. Viết không dấu vì `putText` không vẽ được chữ tiếng Việt.

**4.3 · "Nới hoặc siết cửa sổ chấm điểm."**
```
python tools/evaluate.py --config config_demo.yaml --split test --methods B --hold-last 2 --post-tol 1.0
```
Cảnh báo phải đến trong 1 s sau khi kết thúc cú ngã mới tính là đúng. Các cảnh báo trễ sẽ chuyển thành vừa bỏ sót vừa báo nhầm. Giải thích vì sao nhóm chọn 3 s: đó là thời gian chấp nhận được để gọi người giúp.

---

## Câu hỏi chung — ai cũng có thể bị hỏi

| Câu hỏi | Gợi ý trả lời |
|---|---|
| Tóm tắt đề tài trong 30 giây? | Dùng YOLO-Pose lấy khung xương người từ camera, tính góc thân và chuyển động của hông, rồi dùng máy trạng thái để xác nhận người nằm ngang đủ lâu thì báo ngã. Chạy thời gian thực trên CPU, F1 0,86 trên dữ liệu UR Fall |
| Hệ thống có chính xác 100% không? | Không. Hình ảnh 2D không phân biệt được ý định (chống đẩy, tập yoga so với ngã). Nhóm đo bằng số liệu, phân tích từng ca sai, và nêu rõ hạn chế |
| So với vòng đeo tay hay cảm biến? | Camera không cần người dùng đeo hay sạc pin và theo dõi được nhiều người, nhưng chỉ thấy trong vùng camera và có vấn đề riêng tư |
| Quyền riêng tư thì sao? | Hệ thống chỉ cần khung xương. Có thể chỉ lưu keypoint, không lưu hình, và chỉ lưu ảnh khi có cảnh báo |
| Hướng phát triển? | (1) kiểm tra bất động sau ngã để giảm báo nhầm; (2) học phân loại từ chuỗi keypoint (LSTM, ST-GCN) với dữ liệu tự quay; (3) gửi thông báo về điện thoại; (4) nhiều camera |
| Vì sao không tự huấn luyện mô hình? | Dữ liệu ngã ít (40 video), huấn luyện từ đầu dễ quá khớp. Dùng mô hình tư thế có sẵn rồi xây luật hình học thì giải thích được, ít dữ liệu vẫn chạy |
| Em đã đóng góp phần nào? | Mỗi người nói đúng phần mình trong bảng ở đầu tài liệu |

---

## Quy trình sửa code trước mặt giảng viên (an toàn)

1. **Trước buổi báo cáo**, đảm bảo code sạch: chạy `git status` phải báo `nothing to commit`.
2. Mở đúng file bằng **Ctrl + P**, gõ tên file. Tìm hàm bằng **Ctrl + Shift + O**.
3. Sửa, lưu (**Ctrl + S**), rồi kiểm tra theo thứ tự:
   ```
   python -m pytest -q
   python main.py --source data/videos/ur_fall-11.mp4 --config config_demo.yaml
   python main.py --source 0 --config config_demo.yaml --imgsz 416
   ```
   Lệnh đầu chạy test thuật toán (không cần camera). Lệnh hai chạy trên video có sẵn để chắc có kết quả. Lệnh ba chạy webcam nếu cần.
4. Nếu báo lỗi, đọc **dòng cuối** của thông báo lỗi: nó ghi tên file, số dòng và loại lỗi. Lỗi hay gặp nhất là thụt lề sai (`IndentationError`): Python bắt buộc thụt lề đúng 4 dấu cách.
5. **Hoàn tác sau khi trình diễn:** `git checkout -- .` đưa mọi file về như trước khi sửa. `git diff` để xem mình đã sửa những gì.

Khi bị hỏi điều chưa biết, trả lời thẳng "phần này em chưa kiểm chứng", rồi nói cách sẽ kiểm chứng, đừng đoán.

---

## Kế hoạch học

- [ ] **Buổi 1 (~1,5 giờ, từng người)**: đọc phần mình, làm bài tập tay, mở code trong VS Code để đối chiếu từng đoạn.
- [ ] **Buổi 2 (~1 giờ, từng người)**: tự làm **cả 3 yêu cầu sửa tại chỗ** của phần mình trên máy, chạy kiểm tra, rồi `git checkout -- .` để hoàn tác.
- [ ] **Buổi 3 (~1 giờ, cả nhóm)**: lần lượt từng người trình bày phần mình 5 phút theo thứ tự luồng chạy TV4 → TV1 → TV2 → TV3. Ba bạn còn lại đóng vai giảng viên: hỏi bằng bảng câu hỏi và ra một yêu cầu sửa bất kỳ.
- [ ] Cả nhóm đọc chéo hàm `_step` (phần TV3) và bảng câu hỏi chung, vì đây là phần giảng viên hỏi nhiều nhất.
