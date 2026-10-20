# Kịch bản demo — buổi báo cáo

Tập trước ít nhất 1 lần trên đúng máy và đúng phòng sẽ demo.

## Trước giờ báo cáo (30 phút trước)

1. Mở **Anaconda Prompt**:
   ```
   cd /d "D:\learning\Nam 4\Ki1\TGMT\fall_detection\fall_detection"
   conda activate falldet
   python -m pytest -q
   ```
   Phải ra `19 passed`.
2. Thử webcam (phương pháp B, ảnh 640):
   ```
   python main.py --source 0 --config config_demo.yaml --save-video
   ```
   Thấy khung xương + `ID 1 NORMAL` là được. Nhấn **Q** để thoát.
   - Không mở được webcam: đóng Zoom/Meet/Teams; thử `--source 1`.
   - Quá chậm (FPS < 8): thêm `--imgsz 320`.
3. Mở sẵn giao diện web (để trong tab trình duyệt):
   ```
   streamlit run app.py
   ```
   Thanh bên trái phải ghi **Cấu hình: config_demo.yaml**.
4. Đặt camera cao 1,5–2 m, thấy toàn thân người cách 2–3 m; trải nệm/chăn dày trong khung hình.

## Trong buổi báo cáo (3 phút)

| Bước | Làm gì | Nói gì |
|---|---|---|
| 1 | App: chọn **Webcam**, bật **Bắt đầu**. Một bạn đi lại trước camera | Khung xanh, ID, góc thân, FPS hiển thị trực tiếp |
| 2 | Bạn đó **ngồi xuống ghế** rồi đứng lên | Ngồi không bị báo ngã |
| 3 | Bạn đó **ngã mô phỏng xuống nệm**, nằm yên 3 giây | Vàng (nghi ngờ) → đỏ (té ngã), thông báo bật lên, sự kiện vào bảng lịch sử |
| 4 | Đứng dậy, chờ 2 giây | Hệ thống ghi RECOVERED |
| 5 | Tắt webcam, chọn **Tải video MP4** → `data/videos/ur_fall-11.mp4` | Cùng hệ thống trên video người thật của bộ dữ liệu UR Fall |
| 6 | Mở thư mục `outputs/` mới nhất | Có `events.csv` và ảnh chụp lúc ngã |

**An toàn:** chỉ ngã có kiểm soát xuống nệm dày, có người đứng cạnh. Không ngã thật.

## Phương án dự phòng

- Webcam lỗi hoặc phòng quá tối → bỏ bước 1–4, demo bằng video `ur_fall-11.mp4` (bước 5) và `ur_adl-01.mp4` (không báo nhầm).
- Streamlit lỗi → dùng `python main.py --source data/videos/ur_fall-11.mp4 --config config_demo.yaml`.
- Mất mạng: không ảnh hưởng (mô hình `yolo11n-pose.pt` đã có sẵn trong thư mục). Font chữ của slide cần mạng; xuất slide ra PDF trước.

## Câu hỏi giảng viên có thể hỏi

- **Vì sao không tự huấn luyện YOLO?** Bài toán chính là logic phát hiện ngã; dùng mô hình pose có sẵn và đánh giá giới hạn của nó (người nằm ở xa/mép khung).
- **Vì sao C (đề xuất) không tốt nhất?** C thận trọng hơn, cần quan sát lâu hơn; trên clip UR Fall bị cắt ngắn nên bỏ sót nhiều. Trên test: B F1 0,86, C 0,74. Nhóm báo cáo trung thực, có phân tích từng ca.
- **Chạy được bao nhiêu FPS?** ~28 FPS (ảnh 320), ~15 FPS (ảnh 640) trên CPU máy nhóm.
- **Nằm trên giường thì sao?** Đó là hạn chế đã ghi nhận (adl-10): keypoint 2D không phân biệt được giường và sàn; hướng cải tiến là ước lượng mặt sàn.
