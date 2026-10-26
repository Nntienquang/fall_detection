# Hướng dẫn cài đặt cho từng thành viên

**Yêu cầu: cả 4 người đều phải tự cài và chạy được trên máy mình.** Làm 6 bước dưới đây (khoảng 20–30 phút), xong thì gửi file `setup_reports/<ten>.md` vào nhóm.

Nhóm dùng **VS Code** + **Anaconda**. Chưa có thì cài trước: [VS Code](https://code.visualstudio.com/), [Anaconda](https://www.anaconda.com/download), [Git](https://git-scm.com/download/win).

## Bước 1 — Lấy code

Nhận lời mời vào repo GitHub của nhóm (email), rồi mở **Command Prompt**:
```
cd /d D:\
git clone https://github.com/<tai-khoan-nhom>/fall-detection.git
```
Lần đầu Git sẽ mở cửa sổ đăng nhập GitHub.

## Bước 2 — Tạo môi trường Python (Anaconda Prompt, làm 1 lần)

Mở **Anaconda Prompt** (bấm phím Windows, gõ "Anaconda Prompt"):
```
cd /d D:\fall-detection
conda create -n falldet python=3.11 -y
conda activate falldet
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
Mất khoảng 5–15 phút (PyTorch ~200 MB). **Phải chạy đủ cả hai lệnh `pip install`.** Thiếu lệnh thứ hai sẽ báo `No module named 'cv2'`.

## Bước 3 — Mở dự án trong VS Code

1. VS Code → **File → Open Folder** → chọn `D:\fall-detection`.
2. Nếu trên cùng hiện **Restricted Mode**: bấm **Manage** → **Trust**. Không làm bước này thì extension Python bị tắt.
3. Cài extension **Python** (của Microsoft) nếu chưa có: `Ctrl+Shift+X` → tìm "Python".
4. `Ctrl+Shift+P` → gõ đủ `Python: Select Interpreter` → chọn dòng có **falldet** (`...\anaconda3\envs\falldet\python.exe`). **Không** chọn dòng "Recommended" hay "base".

## Bước 4 — Đặt terminal là Command Prompt

PowerShell trên nhiều máy không nhận lệnh `conda`, nên cả nhóm dùng **cmd**:

1. `Ctrl+Shift+P` → `Terminal: Select Default Profile` → **Command Prompt**.
2. Đóng terminal cũ (biểu tượng 🗑), mở terminal mới bằng ``Ctrl+` ``.
3. Đầu dòng phải có `(falldet)`. Chưa có thì gõ:
   ```
   C:\Users\<ten-may>\anaconda3\Scripts\activate falldet
   ```

## Bước 5 — Kiểm tra và tạo báo cáo cài đặt (bắt buộc)

Trong terminal VS Code:
```
python -m pytest -q
python tools/check_setup.py --name "Ho Ten Cua Ban" --webcam
```
- Lệnh 1 phải ra **29 passed**.
- Lệnh 2 kiểm tra thư viện, YOLO trên video test, webcam; in **ĐẠT / CHƯA ĐẠT** và lưu `setup_reports/<ten>.md`. Gửi file này vào nhóm.

## Bước 6 — Chạy thử hệ thống

| Việc | Lệnh |
|---|---|
| Webcam trực tiếp (nhấn **Q** để thoát) | `python main.py --source 0 --config config_demo.yaml --imgsz 416` |
| Video có sẵn | `python main.py --source data/videos/ur_fall-11.mp4 --config config_demo.yaml` |
| Giao diện web (dừng bằng **Ctrl+C**) | `streamlit run app.py` |

Webcam: ngồi xa 2–3 m để camera thấy **toàn thân**; ngồi sát chỉ thấy đầu-vai thì không phát hiện ngã được.

**Không** bấm nút ▶ Run góc trên phải VS Code: nút đó chạy không kèm tham số. Luôn chạy bằng lệnh trong terminal.

## Làm việc với Git hằng ngày

```
git pull                          # đầu buổi: lấy code mới nhất
git add <file đã sửa>
git commit -m "Mô tả ngắn việc đã làm"
git push                          # cuối buổi: đẩy lên
```
Mỗi người chủ yếu sửa file mình phụ trách (xem tài liệu Hướng dẫn đọc code) để tránh xung đột. Không commit `outputs/`, `cache/`, `results/`, `*.pt` (đã chặn sẵn trong `.gitignore`).

## Khi cả 4 người đã có báo cáo cài đặt

Người tổng hợp (TV4) chép 4 file `.json` trong `setup_reports/` vào cùng một thư mục rồi chạy:
```
python tools/merge_setup_reports.py
```
→ `setup_reports/TONG_HOP.md`: bảng so sánh 4 máy, dán vào mục "Môi trường thực nghiệm" của báo cáo.

## Lỗi hay gặp

| Lỗi | Cách sửa |
|---|---|
| `'conda' is not recognized` | Terminal đang là PowerShell → làm lại Bước 4, hoặc chạy lệnh trong Anaconda Prompt |
| `conda init powershell` báo `Operation failed` | Bỏ qua, dùng cmd (Bước 4) |
| Không thấy `Python: Select Interpreter` | VS Code đang ở Restricted Mode → Trust (Bước 3) |
| `No module named 'cv2'` / `ultralytics` | Chưa kích hoạt `falldet` hoặc chưa chạy `pip install -r requirements.txt` |
| Cài `lap` báo lỗi build | `pip install lap==0.5.12`; vẫn lỗi thì đổi `tracker: botsort.yaml` trong `config.yaml` |
| Tải `yolo11n-pose.pt` thất bại | Lần chạy đầu cần mạng; hoặc xin file `.pt` từ bạn cùng nhóm, đặt vào thư mục dự án |
| Webcam không mở | Đóng Zoom/Meet/Teams; thử `--source 1`; kiểm tra quyền camera của Windows |
| Cảnh báo `LF will be replaced by CRLF` khi `git add` | Bình thường, bỏ qua |
| `torch` cài bản rất nặng (có CUDA) | Gỡ rồi cài lại đúng lệnh có `--index-url .../cpu` ở Bước 2 |
