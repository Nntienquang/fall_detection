# Kịch bản video giới thiệu (8–10 phút)

**Yêu cầu của cô:** video chứng minh cả nhóm cùng làm. **Đủ 4 thành viên, hiện mặt khi nói.** Ai không xuất hiện thì không có điểm.

## Cách quay

- **Cách dễ nhất:** họp Google Meet hoặc Zoom, ai nói thì bật camera và chia sẻ màn hình (slide hoặc VS Code), bấm ghi hình. Camera người nói luôn hiện ở góc.
- **Phần demo (cảnh 6):** quay trực tiếp tại phòng bằng điện thoại đặt cạnh laptop, vừa thấy người thật vừa thấy màn hình hệ thống. Hoặc ghi màn hình laptop, vì người ngã đã hiện trong khung webcam.
- Mỗi người nói phần mình **bằng lời của mình**, nhìn ý chính dưới đây chứ đừng đọc nguyên văn.
- Up YouTube ở chế độ **"Không công khai"** (đừng chọn "Riêng tư", cô sẽ không xem được). Mở link bằng tab ẩn danh để kiểm tra.
- Thay `[Tên TVx]` bằng tên thật, chia vai theo bảng phân công trong `HOC_CODE_LOI.md`.

## Phân cảnh

| # | Cảnh | Ai | Thời lượng | Trên màn hình |
|---|---|---|---|---|
| 1 | Mở đầu | Cả 4, cùng khung | 0:45 | Mặt cả nhóm, rồi slide bìa |
| 2 | Bài toán & kiến trúc | TV4 | 1:15 | Slide Bài toán, Kiến trúc |
| 3 | Phát hiện & theo dõi người | TV1 | 1:30 | Slide + VS Code `pose_tracker.py` |
| 4 | Đặc trưng hình học | TV2 | 1:30 | Slide Đặc trưng + VS Code `features.py` |
| 5 | Phân tích theo thời gian | TV3 | 1:30 | Slide Ba phương pháp + VS Code `_step` |
| 6 | Demo trực tiếp | Cả 4 | 2:00 | Webcam: lần lượt từng người ngồi, cúi, ngã |
| 7 | Kết quả & hạn chế | TV4 | 1:15 | Slide Kết quả, Phân tích lỗi, Camera trực tiếp |
| 8 | Kết luận | Cả 4 | 0:30 | Slide Kết luận, mặt cả nhóm |

## Ý chính từng cảnh

**Cảnh 1 – Mở đầu (cả 4)**
- Mỗi người: "Em là [tên], phụ trách [phần]." (TV1 phát hiện & theo dõi người, TV2 đặc trưng hình học, TV3 phân tích thời gian, TV4 luồng chạy & đánh giá.)
- Một người: "Đề tài của nhóm là phát hiện người té ngã qua camera, chạy thời gian thực trên CPU."

**Cảnh 2 – Bài toán & kiến trúc (TV4)**
- Người cao tuổi ngã một mình cần được phát hiện nhanh; camera không cần đeo thiết bị.
- Mỗi khung hình qua 5 bước: camera → YOLO-Pose lấy 17 khớp → ByteTrack giữ ID → tính đặc trưng → máy trạng thái quyết định.
- Bước 2–3 dùng mô hình có sẵn; nhóm tự xây bước 4–5 và toàn bộ phần đánh giá.

**Cảnh 3 – Phát hiện & theo dõi (TV1)**
- Mở `pose_tracker.py`, chỉ vào `model.track(...)`: một lần chạy mạng trả về khung người, 17 keypoint và độ tin cậy.
- Vì sao cần tracking: phải biết lịch sử của từng người qua thời gian.
- Lỗi thật đã sửa: video fall-08, YOLO trả 2 khung cho cùng một người đang nằm; nhóm bỏ khung trùng bằng tỷ lệ chồng trên khung nhỏ.

**Cảnh 4 – Đặc trưng hình học (TV2)**
- Mở `features.py`, hàm `torso_angle`: góc giữa trục vai → hông và phương thẳng đứng, dùng `atan2`. Đứng ≈ 0°, nằm ≈ 90°.
- Vì sao góc thân tốt hơn tỷ lệ W/H: không phụ thuộc tay chân dang ra hay hướng camera.
- Chuẩn hoá theo `max(W, H)` để ngưỡng dùng được ở mọi khoảng cách.

**Cảnh 5 – Phân tích theo thời gian (TV3)**
- Một khung hình không phân biệt được "đang ngã" với "đang nằm nghỉ" → cần máy trạng thái NORMAL → SUSPECTED → FALL.
- Chỉ báo khi nằm ngang đủ lâu (0,5 s, ≥ 70% số khung), và người phải từng đứng trước đó.
- Ba phương pháp: A dùng W/H, B dùng góc thân, C thêm tốc độ hạ hông.
- Bản demo chờ người nằm yên mới báo, để chống đẩy, bò không bị báo nhầm.

**Cảnh 6 – Demo (cả 4)**
- Theo `KICH_BAN_DEMO.md`, bước 1–6. Ngã ngang qua khung hình, không ngã thẳng về phía camera. **Mỗi người xuất hiện ít nhất một lần trước webcam**, ví dụ: TV1 đi lại, TV2 ngồi ghế, TV3 cúi nhặt đồ, TV4 ngã xuống đệm và nằm yên. Ai đứng ngoài thì nói giải thích.
- Kết thúc bằng mở `outputs/`: ảnh chụp lúc ngã và `events.csv`.

**Cảnh 7 – Kết quả & hạn chế (TV4)**
- 40 video người thật UR Fall; chọn ngưỡng trên val, báo cáo trên test chưa từng dùng.
- B tốt nhất: F1 0,86, bắt 9/10 cú ngã, 2 báo nhầm, trễ trung bình 0,7 s. Giả thuyết "C tốt nhất" không được xác nhận, nhóm báo cáo trung thực.
- Webcam: 13–21 FPS trên CPU; đã xử lý 4 kiểu báo nhầm gặp khi thử (ngồi sát webcam, bị cắt mép ảnh, cúi người lâu, chống đẩy). Hạn chế còn lại: giữ yên tư thế quỳ chống tay, ngã thẳng về phía camera.

**Cảnh 8 – Kết luận (cả 4)**
- Hướng phát triển: quay thêm dữ liệu, học từ chuỗi khung xương, gửi thông báo về điện thoại.
- "Cảm ơn cô đã theo dõi." Cả nhóm cùng khung hình.

## Kiểm tra trước khi nộp

- [ ] Đủ 4 người, ai cũng có mặt và có lời nói.
- [ ] Nghe rõ tiếng, chữ trên màn hình đọc được (phóng to VS Code: Ctrl + =).
- [ ] Link YouTube "Không công khai", mở được bằng tab ẩn danh.
- [ ] Dán link vào form trước 23h00 ngày 07/11.
