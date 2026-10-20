# Phân tích kết quả thực nghiệm — UR Fall Detection Dataset

*Dữ liệu chạy ngày 20/10/2026 trên máy nhóm (CPU, YOLO11n-Pose, imgsz 320, ~28 ms/frame ≈ 35 FPS cho riêng mô hình).*

## 1. Bộ dữ liệu và cách chia

| Tập | Video | Cú ngã | Video không ngã | Dùng để |
|---|---|---|---|---|
| `val` | fall-01…05, adl-01…05 | 5 | 5 | Chọn ngưỡng cho từng phương pháp |
| `test_cu` | fall-06…10, adl-06…10 | 5 | 5 | Đã xem lỗi để sửa code → **không dùng làm kết quả chính** |
| `test` | fall-11…20, adl-11…20 | 10 | 10 | **Kết quả chính** — chưa từng dùng để chỉnh |

Video lấy từ camera 0 của UR Fall (Kwolek & Kepski, 2014), đã cắt lấy nửa ảnh màu 320×240, 30 fps. Nhãn thời điểm ngã do nhóm ghi tay.

## 2. Kết quả chính trên tập `test`

Ngưỡng của mỗi phương pháp chọn riêng trên `val`. Clip UR Fall bị cắt ngay sau khi chạm sàn nên báo cả hai chế độ.

**Giữ khung cuối 2 s** (giả định người vẫn nằm sau khi ngã):

| Phương pháp | TP | FP | FN | Precision | Recall | F1 | Độ trễ TB (s) |
|---|---|---|---|---|---|---|---|
| A — Bounding box (W/H) | 6 | 1 | 4 | 0.86 | 0.60 | 0.71 | 1.09 |
| B — Góc thân người | 7 | 1 | 3 | 0.88 | 0.70 | **0.78** | 0.70 |
| C — Pose + thời gian | 5 | **0** | 5 | **1.00** | 0.50 | 0.67 | 1.57 |

**Không giữ khung cuối** (đúng độ dài clip gốc):

| Phương pháp | TP | FP | FN | Precision | Recall | F1 | Độ trễ TB (s) |
|---|---|---|---|---|---|---|---|
| A — Bounding box (W/H) | 5 | 1 | 5 | 0.83 | 0.50 | 0.62 | 1.07 |
| B — Góc thân người | 7 | 2 | 3 | 0.78 | 0.70 | **0.74** | 0.64 |
| C — Pose + thời gian | 3 | **0** | 7 | **1.00** | 0.30 | 0.46 | 1.46 |

## 3. Nhận xét (imgsz 320)

1. **B có F1 cao nhất** trên tập test. Giả thuyết ban đầu "C tốt nhất" **không được xác nhận** trên dữ liệu này.
2. **C không báo nhầm lần nào** (0 FP trên 10 video sinh hoạt) nhưng bỏ sót nhiều nhất. Điều kiện "phải có cú hạ hông nhanh" làm C thận trọng: đúng mục tiêu giảm báo nhầm, nhưng trả giá bằng recall.
3. **C cần nhiều thời gian quan sát hơn** (độ trễ 1,5 s so với 0,7 s của B). Với clip bị cắt ngắn, điều này trực tiếp thành bỏ sót: khi bỏ giữ khung cuối, recall của C tụt từ 0,50 xuống 0,30.
4. Trên `test_cu` (giữ khung cuối) A và C cùng F1 = 0,91, B = 0,83. Kết quả đổi thứ hạng giữa hai tập 10 video cho thấy **số liệu chưa đủ để kết luận chắc chắn** phương pháp nào hơn: 1 video thay đổi F1 khoảng 0,05–0,1.

## 4. Phân tích từng ca sai trên tập `test` (imgsz 320)

| Video | Kết quả | Nguyên nhân |
|---|---|---|
| fall-13 | Cả 3 bỏ sót | Người ở xa, nhỏ; khi nằm trên sàn YOLO gần như không phát hiện được |
| fall-17 | Cả 3 bỏ sót | Ngã sát mép dưới khung hình, YOLO mất người từ 2,3 s đến hết clip |
| fall-19 | Cả 3 bỏ sót | YOLO chỉ thấy người ở 49/100 frame, mất hoàn toàn sau khi ngã |
| fall-15 | C bỏ sót | Clip hết 0,4 s sau khi chạm sàn; khung cuối có khung xương lệch (góc thân 29°) nên C coi là đã ngồi dậy |
| fall-18 | A, C bỏ sót | Ngã từ ghế ra mép khung, YOLO mất người từ 1,3 s; B bắt được vì báo ngay khi người chúi về phía trước |
| adl-11 | A báo nhầm | Tư thế cúi/ngồi làm khung người bè ngang (W/H vượt 0,9) |
| adl-17 | B báo nhầm | Thân nghiêng quá ngưỡng góc trong một động tác sinh hoạt |

**3/10 cú ngã bị cả ba phương pháp bỏ sót do YOLO không phát hiện được người đang nằm** (người nhỏ, ở mép khung hoặc ra khỏi khung). Đây là giới hạn của bước nhận diện, không phải của thuật toán phát hiện ngã.

Ca sai đã phân tích trên `test_cu` (trước khi có tập `test`):
- fall-08: YOLO trả 2 khung chồng nhau cho cùng 1 người, ByteTrack gán ID mới làm mất lịch sử → đã sửa (`dup_overlap`, có unit test). Sau khi sửa, ca này vẫn sót do người ra mép khung.
- adl-10: ngồi phịch xuống giường rồi nằm ra; với keypoint 2D không phân biệt được nằm trên giường và nằm trên sàn.

## 5. Ảnh hưởng của kích thước ảnh vào YOLO (imgsz 320 → 640)

Giả thuyết từ mục 4: ảnh vào lớn hơn giúp YOLO nhận ra người nhỏ và người đang nằm. Chạy lại toàn bộ với `imgsz 640` (cùng video, cùng nhãn, ngưỡng chọn lại trên `val`). Tập `test`, giữ khung cuối 2 s:

| Phương pháp | F1 (imgsz 320) | F1 (imgsz 640) | TP / FP / FN ở 640 |
|---|---|---|---|
| A — Bounding box (W/H) | 0.71 | 0.78 | 7 / 1 / 3 |
| B — Góc thân người | 0.78 | **0.86** | 9 / 2 / 1 |
| C — Pose + thời gian | 0.67 | 0.74 | 7 / 2 / 3 |

| | imgsz 320 | imgsz 640 |
|---|---|---|
| Thời gian YOLO + tracking (CPU máy nhóm) | ~28 ms/frame (~35 FPS) | ~58–78 ms/frame (~15 FPS) |

Thay đổi theo từng video:
- **fall-13, fall-17 chuyển từ bỏ sót sang phát hiện được với cả 3 phương pháp** → xác nhận giả thuyết: lỗi do bước nhận diện người, ảnh vào lớn hơn khắc phục được.
- **fall-19 vẫn bị cả 3 bỏ sót**: người ra khỏi khung hình, không kích thước ảnh nào cứu được.
- **Đổi lại, báo nhầm tăng**: adl-11 (B, C) và adl-17 (C) giờ bị báo nhầm. YOLO thấy người rõ hơn trong cả các động tác sinh hoạt (cúi, nằm), nên các tư thế giống ngã cũng hiện ra nhiều hơn.
- Trên `val`, F1 của cả 3 phương pháp ở 320 và 640 **bằng nhau**, nên việc chọn 640 dựa trên lập luận về nhận diện người (mục 4), không phải dựa trên số liệu test.

**Kết luận:** imgsz 640 tăng recall rõ rệt, tốc độ vẫn ~15 FPS, đủ cho giám sát thời gian thực. B (góc thân người) là phương pháp tốt nhất trên tập test ở cả hai kích thước ảnh. C luôn thận trọng hơn ở imgsz 320 (0 báo nhầm), nhưng lợi thế này mất đi ở 640.

## 6. Hạn chế và hướng cải tiến

- **Nhận diện người nằm:** đã thử imgsz 640 (mục 5); có thể thử thêm `yolo11s-pose` (chọn trên `val`, áp dụng cho cả 3 phương pháp).
- **Góc đặt camera:** đặt cao 2–2,5 m, nhìn chéo xuống, để người ngã không rơi ra mép khung.
- **Nằm trên giường vs. trên sàn:** cần ước lượng mặt sàn (hiệu chỉnh camera) hoặc thông tin độ sâu.
- **Dữ liệu:** 20 cú ngã là ít; bổ sung video nhóm tự quay và fall-21…30 của UR Fall.
- **Clip bị cắt:** UR Fall không có đoạn sau khi ngã đủ dài; video tự quay nên ghi tiếp ≥ 5 s sau khi chạm sàn.
