"""Gộp báo cáo cài đặt của cả nhóm thành 1 bảng (dán vào báo cáo, mục môi trường thực nghiệm).

  python tools/merge_setup_reports.py   →  setup_reports/TONG_HOP.md
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
d = ROOT / "setup_reports"
rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))]
if not rows:
    raise SystemExit("Chưa có báo cáo nào trong setup_reports/. Mỗi người chạy tools/check_setup.py trước.")

lines = ["# Tổng hợp cài đặt và tốc độ chạy của nhóm", "",
         "| Thành viên | Kết quả | Hệ điều hành | CPU | Nhân/luồng | RAM (GB) | imgsz | Model (ms/frame) | FPS |",
         "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    s, b = r["system"], r.get("bench", {})
    ok = "Đạt" if all(c["ok"] for c in r["checks"]) else "Chưa đạt"
    lines.append(f"| {r['name']} | {ok} | {s['os']} | {s['cpu']} | {s['cores_physical']}/{s['cores_logical']} | "
                 f"{s['ram_gb']} | {b.get('imgsz', '')} | {b.get('model_ms_median', '')} | {b.get('fps', '')} |")
lines += ["", f"Đo trên `data/videos/video_test_goc.mp4`, YOLO11n-Pose, chỉ CPU, {rows[0].get('bench', {}).get('frames', '')} frame mỗi máy."]
(d / "TONG_HOP.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"\nĐã lưu {d / 'TONG_HOP.md'}")
