"""Kiểm tra cài đặt cho TỪNG thành viên và xuất báo cáo làm bằng chứng.

  python tools/check_setup.py --name "Nguyen Van A"
  python tools/check_setup.py --name "Nguyen Van A" --webcam      # kiểm tra thêm webcam

Kết quả: setup_reports/<ten>.md và setup_reports/<ten>.json
Gộp 4 người thành bảng cho báo cáo:  python tools/merge_setup_reports.py
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VIDEO = ROOT / "data" / "videos" / "video_test_goc.mp4"


def cpu_name() -> str:
    try:
        if sys.platform == "win32":
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        if sys.platform.startswith("linux"):
            for line in open("/proc/cpuinfo", encoding="utf-8"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    return platform.processor() or "không rõ"


def ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:
        return None


def physical_cores():
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except Exception:
        return None


class Report:
    def __init__(self, name):
        self.data = {"name": name, "time": datetime.now().isoformat(timespec="seconds"), "checks": [], "system": {}, "bench": {}}

    def check(self, title, ok, detail=""):
        self.data["checks"].append({"title": title, "ok": bool(ok), "detail": str(detail)})
        mark = "OK  " if ok else "LỖI"
        print(f"  [{mark}] {title}" + (f" — {detail}" if detail else ""))
        return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Họ tên thành viên")
    ap.add_argument("--webcam", action="store_true", help="kiểm tra thêm webcam số 0")
    ap.add_argument("--frames", type=int, default=150, help="số frame dùng để đo tốc độ")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--threads", type=int, default=0, help="0 = số nhân thật của CPU")
    args = ap.parse_args()

    R = Report(args.name)
    print(f"\n=== Kiểm tra cài đặt — {args.name} ===\n")

    # 1. Hệ thống
    sysinfo = {"os": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
               "cpu": cpu_name(), "cores_physical": physical_cores(), "cores_logical": os.cpu_count(),
               "ram_gb": ram_gb(), "machine": platform.node()}
    R.data["system"] = sysinfo
    print(f"  Máy: {sysinfo['os']} | {sysinfo['cpu']} | {sysinfo['cores_physical']} nhân / "
          f"{sysinfo['cores_logical']} luồng | RAM {sysinfo['ram_gb']} GB | Python {sysinfo['python']}\n")
    R.check("Python 3.9 trở lên", sys.version_info >= (3, 9), sysinfo["python"])

    # 2. Thư viện
    versions = {}
    for mod, label in [("torch", "PyTorch"), ("ultralytics", "Ultralytics"), ("cv2", "OpenCV"),
                       ("numpy", "NumPy"), ("yaml", "PyYAML"), ("lap", "lap (ByteTrack)"),
                       ("streamlit", "Streamlit"), ("pandas", "pandas"), ("matplotlib", "matplotlib"),
                       ("pytest", "pytest")]:
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "?")
            R.check(f"Cài {label}", True, versions[mod])
        except Exception as e:
            versions[mod] = None
            R.check(f"Cài {label}", False, f"{type(e).__name__}: {e}  →  pip install -r requirements.txt")
    R.data["versions"] = versions
    try:
        import torch
        R.check("PyTorch chạy CPU", True, "CUDA có sẵn" if torch.cuda.is_available() else "không có CUDA (bình thường)")
    except Exception:
        pass

    # 3. Unit test
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, capture_output=True, text=True, timeout=300)
        last = (p.stdout.strip().splitlines() or ["(không có kết quả)"])[-1]
        m = re.search(r"(\d+) passed", last)
        R.check("Unit test thuật toán", p.returncode == 0, last)
        R.data["tests_passed"] = int(m.group(1)) if m else 0
    except Exception as e:
        R.check("Unit test thuật toán", False, e)

    # 4. YOLO Pose + ByteTrack trên video test
    if not VIDEO.exists():
        R.check("Có video test", False, f"thiếu {VIDEO.relative_to(ROOT)}")
    else:
        try:
            import cv2
            from src.config import load_config
            from src.fall_detector import FallDetector
            from src.pose_tracker import PoseTracker

            cfg = load_config(ROOT / "config.yaml")
            cfg.model.imgsz = args.imgsz
            cfg.model.threads = args.threads or physical_cores() or 4
            t0 = time.perf_counter()
            tracker = PoseTracker(cfg.model)
            tracker.warmup()
            R.check("Tải mô hình YOLO Pose", True, f"{cfg.model.weights}, {time.perf_counter() - t0:.1f}s")

            cap = cv2.VideoCapture(str(VIDEO))
            det = FallDetector(cfg.detector)
            infer, loop, n_people, falls, n = [], [], 0, 0, 0
            fps_v = cap.get(cv2.CAP_PROP_FPS) or 15
            t_all = time.perf_counter()
            while n < args.frames:
                ok, frame = cap.read()
                if not ok:
                    break
                tl = time.perf_counter()
                people = tracker(frame)
                _, evs = det.update(people, n / fps_v)
                loop.append((time.perf_counter() - tl) * 1000)
                infer.append(tracker.last_infer_ms)
                n_people += bool(people)
                falls += sum(e.kind == "FALL" for e in evs)
                n += 1
            cap.release()
            total = time.perf_counter() - t_all
            bench = {"imgsz": cfg.model.imgsz, "threads": cfg.model.threads, "frames": n,
                     "model_ms_median": round(statistics.median(infer), 1) if infer else None,
                     "loop_ms_median": round(statistics.median(loop), 1) if loop else None,
                     "fps": round(n / total, 2) if total else None,
                     "frames_with_person_pct": round(100 * n_people / max(n, 1), 1)}
            R.data["bench"] = bench
            R.check("YOLO + tracking chạy trên video", n > 0, f"{n} frame, {bench['fps']} FPS, "
                    f"model {bench['model_ms_median']} ms/frame")
            R.check("YOLO nhận ra người", n_people > 0,
                    f"{bench['frames_with_person_pct']}% frame có người (video là hình dựng, không cần 100%)")
        except Exception as e:
            R.check("YOLO + tracking chạy trên video", False, f"{type(e).__name__}: {e}")

    # 5. Webcam (tuỳ chọn)
    if args.webcam:
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            cap.release()
            R.check("Webcam số 0", ok, f"{frame.shape[1]}x{frame.shape[0]}" if ok else "không mở được, thử camera khác")
        except Exception as e:
            R.check("Webcam số 0", False, e)

    # 6. Xuất báo cáo
    out = ROOT / "setup_reports"
    out.mkdir(exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", args.name.strip()).strip("_") or "member"
    (out / f"{slug}.json").write_text(json.dumps(R.data, indent=2, ensure_ascii=False), encoding="utf-8")
    ok_all = all(c["ok"] for c in R.data["checks"])
    md = [f"# Báo cáo cài đặt — {args.name}", "",
          f"- Thời gian: {R.data['time']}",
          f"- Máy: {sysinfo['os']}, {sysinfo['cpu']}, {sysinfo['cores_physical']} nhân / {sysinfo['cores_logical']} luồng, RAM {sysinfo['ram_gb']} GB",
          f"- Python {sysinfo['python']}, PyTorch {versions.get('torch')}, Ultralytics {versions.get('ultralytics')}, OpenCV {versions.get('cv2')}",
          f"- Kết luận: **{'ĐẠT' if ok_all else 'CHƯA ĐẠT'}**", "", "| Hạng mục | Kết quả | Chi tiết |", "|---|---|---|"]
    md += [f"| {c['title']} | {'Đạt' if c['ok'] else 'Lỗi'} | {c['detail']} |" for c in R.data["checks"]]
    (out / f"{slug}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\n{'ĐẠT — đã cài đặt và chạy được.' if ok_all else 'CHƯA ĐẠT — sửa các mục [LỖI] ở trên rồi chạy lại.'}")
    print(f"Báo cáo: setup_reports/{slug}.md")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
