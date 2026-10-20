# Hướng dẫn cài đặt cho từng thành viên

**Yêu cầu: cả 4 người đều phải tự cài và chạy được trên máy mình.** Mỗi người làm 4 bước dưới đây, xong thì gửi file `setup_reports/<ten>.md` vào nhóm.

## Bước 1 — Lấy code

Lấy bản mới nhất từ nhóm (GitHub hoặc file zip), giải nén, nhớ có thư mục `data/videos/` chứa `video_test_goc.mp4`.

## Bước 2 — Tạo môi trường Python

Chọn **một** trong ba cách, tuỳ máy.

**Cách A: Windows, có Anaconda (Anaconda Prompt)**
```
cd /d "<đường dẫn tới thư mục dự án>"
conda create -n falldet python=3.11 -y
conda activate falldet
```

**Cách B: Windows, không có Anaconda (PowerShell, cần Python 3.11 từ python.org)**
```
cd "<đường dẫn tới thư mục dự án>"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
Nếu PowerShell báo bị chặn script: chạy `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` một lần rồi thử lại.

**Cách C: Linux / Kali / macOS (Terminal)**
```
cd "<đường dẫn tới thư mục dự án>"
python3 -m venv .venv
source .venv/bin/activate
```
Kali/Ubuntu thiếu venv thì cài trước: `sudo apt install python3-venv python3-pip`.

## Bước 3 — Cài thư viện (giống nhau cho mọi máy)
```
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
Mất khoảng 5–15 phút tuỳ mạng (PyTorch ~200 MB).

## Bước 4 — Chạy kiểm tra (bắt buộc, có webcam thì thêm `--webcam`)
```
python tools/check_setup.py --name "Ho Ten Cua Ban" --webcam
```
Công cụ tự kiểm tra: phiên bản Python và thư viện, 17 unit test, tải YOLO Pose, chạy YOLO + ByteTrack trên video test và đo FPS, mở webcam. Cuối cùng in **ĐẠT** hoặc **CHƯA ĐẠT** và lưu `setup_reports/<ten>.md`.

Sau đó tự chạy thử giao diện để quen:
```
python main.py --source 0
streamlit run app.py
```

## Khi cả 4 người đã có báo cáo

Người tổng hợp (TV4) chép 4 file `.json` trong `setup_reports/` vào cùng một thư mục rồi chạy:
```
python tools/merge_setup_reports.py
```
→ `setup_reports/TONG_HOP.md`: bảng so sánh 4 máy, dán vào mục "Môi trường thực nghiệm" của báo cáo.

## Lỗi hay gặp

| Lỗi | Cách sửa |
|---|---|
| `No module named ...` | Chưa kích hoạt môi trường (`conda activate falldet` hoặc `activate` của `.venv`), hoặc chưa chạy Bước 3 |
| Cài `lap` báo lỗi build | `pip install lap==0.5.12`; vẫn lỗi thì đổi `tracker: botsort.yaml` trong `config.yaml` |
| Tải `yolo11n-pose.pt` thất bại | Cần mạng ở lần chạy đầu; hoặc xin file `.pt` từ bạn cùng nhóm, đặt vào thư mục dự án |
| Webcam không mở | Đóng Zoom/Meet/Teams; thử `--source 1`; kiểm tra quyền camera của Windows |
| `torch` cài bản quá nặng (có CUDA) | Gỡ rồi cài lại đúng lệnh có `--index-url .../cpu` ở Bước 3 |
