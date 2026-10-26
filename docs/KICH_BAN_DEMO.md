# Kịch bản demo hệ thống

Dùng cho cả **buổi báo cáo** và **phần demo trong video**. Tập trước ít nhất 1 lần trên đúng máy, đúng phòng.

## Chuẩn bị (30 phút trước)

1. Mở VS Code ở thư mục project, terminal **Command Prompt**:
   ```
   conda activate falldet
   git pull
   python -m pytest -q
   ```
   Phải ra **27 passed**.
2. Thử webcam:
   ```
   python main.py --source 0 --config config_demo.yaml --imgsz 416
   ```
   Thấy khung xương và `ID 1 NORMAL` là được. Phím **F** = toàn màn hình, **Q** = thoát.
   - Không mở được webcam: đóng Zoom/Meet/Teams, hoặc thử `--source 1`.
   - FPS < 8: dùng `--imgsz 320`.
3. **Đặt camera:** cao 1–1,5 m, cách chỗ ngã 2,5–3,5 m, **thấy trọn người từ đầu đến chân** cả khi đứng lẫn khi nằm. Trải đệm hoặc chăn dày trong khung hình. Không ai đứng sát camera.
4. Nếu dùng giao diện web: `streamlit run app.py`, thanh bên trái phải ghi **Cấu hình: config_demo.yaml**.

## Hệ thống sẽ phản ứng thế nào (để nói đúng khi demo)

| Màu khung | Trạng thái | Khi nào |
|---|---|---|
| Xanh | NORMAL | Đi, đứng, ngồi, cúi |
| Vàng | SUSPECTED | Thân nghiêng quá 45°, đang theo dõi |
| Đỏ + viền đỏ + dòng cảnh báo | FALL | Nằm ngang ≥ 0,5 s **và đã nằm yên** khoảng 1 s |

Bản demo chờ người **nằm yên** mới báo, nên vàng sẽ kéo dài thêm khoảng 1–1,5 s so với video UR Fall trên slide. Đây là chủ ý: để chống đẩy, bò không bị báo nhầm.

## Trình tự demo (3–4 phút)

| Bước | Làm gì | Nói gì |
|---|---|---|
| 1 | Một bạn đi lại trước camera | Khung xương 17 điểm, ID, góc thân, FPS hiển thị trực tiếp |
| 2 | **Ngồi xuống ghế**, rồi đứng lên | Ngồi: góc thân nhỏ, vẫn xanh, không báo |
| 3 | **Cúi nhặt đồ**, đứng lên | Có thể vàng thoáng qua, đứng dậy thì huỷ, không báo |
| 4 | **Ngã có kiểm soát xuống đệm**, **nằm yên 3 giây** | Vàng → đỏ, cảnh báo bật lên, lưu ảnh |
| 5 | Đứng dậy, chờ 2 giây | Ghi RECOVERED, khung về xanh |
| 6 | Mở thư mục `outputs/` mới nhất | `events.csv` (thời điểm, góc thân) và ảnh chụp lúc ngã |
| 7 | `python main.py --source data/videos/ur_fall-11.mp4 --config config_demo.yaml` | Cùng hệ thống trên video người thật của bộ dữ liệu UR Fall |

**Tuỳ chọn** (chỉ làm nếu đã thử thành công trước đó): chống đẩy liên tục vài nhịp → vẫn vàng, không lên đỏ, vì vai còn chuyển động. Nói rõ: giữ yên tư thế quỳ chống tay thì vẫn bị báo, đó là hạn chế.

**An toàn:** chỉ ngã có kiểm soát xuống đệm dày, ngã từ tư thế quỳ hoặc ngồi xổm, có người đứng cạnh. Không ngã từ tư thế đứng thẳng.

## Phương án dự phòng

- Webcam lỗi hoặc phòng quá tối → bỏ bước 1–6, demo bằng `ur_fall-11.mp4` (có ngã) và `ur_adl-01.mp4` (không ngã, không báo).
- Streamlit lỗi → dùng `main.py` như bước 7.
- Mất mạng: không ảnh hưởng, vì mô hình `yolo11n-pose.pt` đã có trong thư mục. Xuất slide ra PDF trước, phòng trường hợp font chữ cần mạng.
