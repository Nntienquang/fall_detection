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

## Bổ sung 26/10 — Báo nhầm khi chống đẩy (phát hiện khi test webcam)

**Hiện tượng.** Chống đẩy trước webcam → cảnh báo FALL (góc thân 81–90°). Phương pháp B chỉ nhìn tư thế thân
trong từng khung; chống đẩy và nằm sau ngã đều có thân nằm ngang → báo nhầm là đúng với thiết kế của B.

**Đã thử: luật "chống tay"** (`arm_drop` trong src/features.py, `support_*` trong config): khi cổ tay chạm sàn
ngay dưới vai và thấp hơn vai rõ rệt thì coi là đang tự đỡ người, không phải nằm. Ngoại lệ: vừa có cú hạ hông nhanh
→ coi là ngã sấp chống tay (để không mất các cú ngã như ur_fall-05, 09).

- Trên UR Fall (cấu hình demo, ảnh 320): không mất cú ngã nào, không thêm báo nhầm, cảnh báo chậm hơn ~0,3 s.
- Hai phiên bản đầu bị loại vì làm mất cú ngã thật (F1 tập test tụt 0,74 → 0,57).
- **Trên webcam thật: KHÔNG chặn được** — 3/3 lần chống đẩy vẫn bị báo (run_20261026_101941, ảnh 320).
  Nguyên nhân: (1) chủ động nằm xuống chống đẩy cũng làm hông hạ nhanh → rơi vào ngoại lệ "đang ngã";
  (2) ở ảnh 320, YOLO bắt cổ tay kém khi người sát sàn; (3) chống đẩy quay mặt vào camera: thân bị co ngắn, tay chồng lên thân.
  Bài kiểm thử mô phỏng đã cho người hạ xuống chậm nên không phát hiện ra vấn đề này.

**Kết luận.** Luật đã tắt (`support_drop: 0`). Đây là **hạn chế của phương pháp dựa trên tư thế 2D**: chống đẩy,
ngã sấp chống tay, nằm tập thể dục có hình học gần giống nhau, chỉ khác ở ý định của người.
Hướng giải quyết: (1) chỉ cảnh báo khi người nằm **bất động** thêm 2–3 s sau khi ngã (người ngã thường nằm yên,
chống đẩy thì chuyển động liên tục) — đổi lại cảnh báo chậm hơn; (2) học phân loại từ chuỗi keypoint (LSTM, ST-GCN)
với dữ liệu tự quay có các động tác dễ nhầm.

## Bổ sung 26/10 (2) — Kết quả webcam run_20261026_104803 (ảnh 416, cấu hình demo B)

4 cảnh báo, chia 2 loại:

| Thời điểm | Tình huống | Góc thân | Loại |
|---|---|---|---|
| 16,8 s; 28,0 s | Quỳ chống tay (tư thế bò) | 79–81° | Hạn chế của phương pháp — giống ca chống đẩy |
| 52,6 s; 64,7 s | Người đứng sát webcam cúi người, bị cắt mất nửa dưới ở mép phải/dưới ảnh | 44–47° | **Sửa được** |

**Sửa: `edge_guard`.** Nếu khung người chạm mép ảnh (trong 2%) **và** không thấy cổ chân nào → người bị cắt mất nửa
dưới, khung xương chỉ phản ánh phần nhìn thấy → không kết luận "nằm". Bật trong `config_demo.yaml`, tắt trong thí nghiệm chính.

- Trên UR Fall (khung 320×240 thật): chỉ 11/996 khung thân-ngang bị ảnh hưởng; số cú ngã bắt được, báo nhầm và
  độ trễ **giữ nguyên** trên cả val, test_cu, test.
- Kiểm thử: `test_close_person_cut_at_frame_edge_does_not_alarm` (tắt luật thì báo nhầm, bật thì không),
  `test_lying_person_with_feet_visible_still_alarms_with_edge_guard`.
- Đánh đổi: người ngã mà chân nằm ngoài khung hình sẽ không được báo — đặt camera bao quát cả phòng.

**Quỳ chống tay / chống đẩy** vẫn bị báo. Không xử lý bằng thêm luật, vì ngã sấp chống tay trông giống hệt
(xem mục trên). Ghi vào báo cáo như hạn chế; hướng giải quyết: kiểm tra bất động sau ngã hoặc học theo chuỗi keypoint.

## Bổ sung 26/10 (3) — Luật "nằm yên mới báo" cho chống đẩy, bò

**Ý tưởng.** Người ngã thường nằm yên; chống đẩy, bò thì vai lên xuống liên tục. Sau khi đủ điều kiện nằm
(`confirm_time`, `confirm_ratio`), hệ thống **chờ đến khi vai dịch chuyển < 12% chiều dài người trong 1 s** rồi mới
báo (`still_thresh: 0.12`). Đang chờ mà đứng dậy → huỷ.

**Chọn ngưỡng.** Độ rung khung xương khi nằm yên thật (fall-06, 11, 12 — các clip có đủ thời gian nằm): 0,04–0,08.
Chống đẩy thật: vai dịch khoảng 0,3 chiều dài người. Chọn 0,12 để có khoảng an toàn với độ rung.

**Hai lỗi phát hiện và sửa khi làm luật này:**
1. Trong lúc chờ nằm yên, tỉ lệ khung nằm ngang bị tính lại liên tục → tụt dưới 70% do góc thân nhiễu lúc đang
   rơi → huỷ nghi ngờ sớm. Sửa: chỉ kiểm tra tỉ lệ một lần ở mốc `confirm_time`, sau đó chỉ chờ nằm yên.
2. `edge_guard` coi người bị cắt ở mép ảnh là "đứng thẳng" → có thể huỷ nghi ngờ của người nằm sát mép.
   Sửa: coi là "không rõ" — không tính là nằm, cũng không tính là đã đứng dậy.

**Kết quả trên UR Fall (cấu hình demo, ảnh 320, có `edge_guard`), TP/FP/FN:**

| Cách giữ khung cuối | Luật | val | test_cu | test | Độ trễ test |
|---|---|---|---|---|---|
| Khung cuối 2 s (cách đánh giá chính thức) | không | 5/3/0 | 5/2/0 | 7/2/3 | 0,64 s |
| | nằm yên 0,12 | 4/1/1 | 3/0/2 | 5/0/5 | 1,95 s |
| Khung cuối **có người** 2 s | không | 5/3/0 | 5/2/0 | 8/2/2 | 0,68 s |
| | nằm yên 0,12 | 5/1/0 | 4/1/1 | 7/1/3 | 1,80 s |

- Báo nhầm giảm rõ (tổng 7 → 3 hoặc 1). Chống đẩy mô phỏng không còn bị báo.
- Đổi lại sót thêm cú ngã và báo chậm hơn ~1,2 s. Phần lớn số sót là do **clip UR bị cắt quá sớm**: luật cần ≥ 1 s
  thấy người nằm yên, trong khi nhiều clip hết 0,3–1 s sau khi chạm sàn, khung cuối không còn ai hoặc khung xương sai
  (fall-07 mất góc thân, fall-15 góc 29°). UR Fall vì vậy **không đánh giá công bằng được luật này** — cần video tự quay
  có người nằm lại vài giây sau ngã.
- Vì thế luật chỉ bật trong `config_demo.yaml` (webcam), thí nghiệm chính giữ nguyên (`still_thresh: 0`).

**Hạn chế còn lại:** giữ yên tư thế quỳ chống tay hoặc pha thấp chống đẩy quá ~1,5 s vẫn bị báo; người ngã còn
giãy giụa sẽ được báo chậm hơn. Kiểm thử: `test_pushup_moving_does_not_alarm_with_still_check`,
`test_real_fall_still_alarms_with_still_check`, `test_still_check_cancels_when_person_gets_up` (27/27 passed).

## Bổ sung 26/10 (4) — Cúi người giữ yên bị báo ngã

**Hiện tượng (webcam).** Đứng cúi người nhặt đồ, góc thân 53°, giữ yên → FALL. Luật nằm yên không giúp vì người đứng yên.

**Đặc trưng phân biệt.** Người cúi: **chân dựng đứng** (trục hông → cổ chân lệch ≤ 25° so với phương thẳng đứng,
cổ chân thấp hơn hông ≥ 35% chiều dài người) **và hông chưa hạ** (< 15% chiều dài người so với lần đứng thẳng gần nhất).
Người ngã: chân nằm ngang hoặc hông rơi xuống sàn. Luật `bend_guard` coi trường hợp này là đứng.

**Đã thử và loại:** chỉ dùng góc chân, hoặc chỉ dùng độ cao hông. Người ngã **hướng chân về camera** (fall-13, 17, 18)
trên ảnh 2D có chân dựng đứng (1–13°) và hông gần như không hạ — giống hệt người cúi.

**Kết quả UR Fall (cấu hình demo, ảnh 320, TP/FP/FN, giữ khung cuối 2 s chính thức):**

| | val | test_cu | test | Báo nhầm ở video |
|---|---|---|---|---|
| Không `bend_guard` | 5/3/0 | 5/2/0 | 7/2/3 | adl-01, 04, 05, 06, 10, 11, 17 |
| Có `bend_guard` | 4/0/1 | 5/1/0 | 6/0/4 | adl-10 |

Báo nhầm tổng 7 → 1 (các ca cúi người adl-04, 05, 06, 17 hết hẳn); sót thêm fall-05 và fall-18 (ngã hướng về camera).
Khi kết hợp với luật nằm yên: không còn báo nhầm nào trên cả 3 tập, nhưng sót nhiều hơn do clip UR ngắn (mục (3)).

**Áp dụng.** Bật trong `config_demo.yaml`; khi demo, ngã **ngang qua khung hình**, không ngã thẳng về phía camera.
Thí nghiệm chính giữ nguyên. Kiểm thử: `test_bending_and_holding_does_not_alarm_with_bend_guard`,
`test_real_fall_still_alarms_with_all_webcam_rules` (29/29 passed).

## Bổ sung 26/10 (5) — Phương pháp D: học máy trên dữ liệu công khai GMDCSA24

**Dữ liệu thêm.** GMDCSA24 (Alam et al., 2024, giấy phép MIT, github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos):
4 người, 79 video ngã (22 ngã ngửa, 17 ngã sấp, nhiều cú ngã xuống giường) + 81 video sinh hoạt (ngủ trên giường, ngồi, cúi, tập thể dục).
Khung xương trích bằng YOLO11n-Pose ảnh 416 (`tools/cache_poses.py`), nhãn từ file CSV gốc (`tools/prepare_gmdcsa24.py`).

**Mô hình.** MLP 2 lớp ẩn (64, 32) trên 62 đặc trưng mỗi khung: 17 khớp chuẩn hoá theo khung người, góc thân, W/H,
độ co ngắn thân/chân so với bề rộng vai, hông hạ và khung người thấp đi so với lần đứng gần nhất, tốc độ hạ, mức cử động.
Tăng dữ liệu bằng lật ngang. Huấn luyện bằng scikit-learn, xuất trọng số ra `models/fall_mlp.npz` (52 KB), chạy bằng numpy.
D thay bước "khung này có đang nằm không" của B; máy trạng thái giữ nguyên.

**Chia dữ liệu (theo người, không trộn):** train = GMDCSA24 người 1–3 + UR val, test_cu · test = GMDCSA24 người 4 + UR test.
Ngưỡng 0,2 và tổ hợp luật chọn bằng **kiểm tra chéo trên train** (bỏ ra từng người), không nhìn tập test.

**Kiểm tra chéo trên train (72 cú ngã):** B demo 52/11/20 (F1 0,77) · B không luật 59/38/13 (0,70) · **D + mép ảnh 55/4/17 (0,84)**.

**Test — chạy một lần (TP/FP/FN):**

| Cấu hình | GMDCSA24 người 4 | UR test | Tổng P / R / F1 |
|---|---|---|---|
| B · demo (cúi + mép + nằm yên) | 10/5/7 | 8/1/2 | 0,75 / 0,67 / 0,71 |
| B · cúi + mép ảnh | 14/6/3 | 8/1/2 | 0,76 / 0,81 / **0,79** |
| D · mép ảnh | 11/2/6 | 7/1/3 | **0,86** / 0,67 / 0,75 |
| D · mép ảnh + nằm yên | 10/2/7 | 6/0/4 | 0,89 / 0,59 / 0,71 |

Riêng **ngã xuống sàn** (bỏ 11 video ngã xuống giường; 16 cú ngã, 30 video sinh hoạt): B demo 13/6/3 (P 0,68, R 0,81, F1 0,74) ·
**D + mép ảnh 12/3/4 (P 0,80, R 0,75, F1 0,77)**.

**Nhận xét.**
- D **giảm báo nhầm khoảng một nửa** (nhất là nằm ngủ trên giường, nhặt đồ dưới sàn), đổi lại sót nhiều hơn B một chút.
- D sót gần hết các cú **ngã xuống giường**, vì dữ liệu gán "ngã xuống giường" là ngã nhưng "nằm ngủ trên giường" là không ngã —
  với khung xương hai việc gần như giống nhau. D học cách coi nằm trên giường là an toàn.
- Luật "nằm yên mới báo" làm giảm recall trên cả hai bộ dữ liệu vì clip hết ngay sau khi ngã; trên webcam (người nằm lâu) nó chỉ làm báo chậm hơn.
- Cả B và D vẫn báo nhầm video chống đẩy (GMDCSA24 người 4, ADL 06): dữ liệu có quá ít video chống đẩy để D học được.
- Chênh lệch giữa các cấu hình chỉ 1–4 video trên 27 cú ngã, nên **chưa thể kết luận D tốt hơn B**; ưu điểm rõ ràng của D là precision.

**Áp dụng.** Cấu hình demo giữ phương pháp B (đã kiểm chứng trên webcam). D chạy bằng `--method D` để so sánh trực tiếp.
Tái lập: `python tools/train_fall_model.py --train cache_gmd:data/labels_gmd.csv:train_gmd cache_416:data/labels.csv:val,test_cu --test cache_gmd:data/labels_gmd.csv:test_gmd cache_416:data/labels.csv:test`
