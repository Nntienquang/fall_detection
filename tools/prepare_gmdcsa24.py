"""Tạo data/labels_gmd.csv từ nhãn gốc của bộ dữ liệu GMDCSA24.

GMDCSA24 (Alam et al., 2024, giấy phép MIT): 4 người, 79 video ngã + 81 video sinh hoạt,
mỗi người một thư mục "Subject N/" gồm Fall/, ADL/, Fall.csv, ADL.csv.
Cột "Classes" ghi đoạn thời gian, vd "Falling (SW)[1.8 to 5]; Walking[0 to 1.8]".

Chia tập theo NGƯỜI (không trộn khung hình của cùng một người vào cả train và test):
  Subject 1–3 → train_gmd (huấn luyện)   ·   Subject 4 → test_gmd (kiểm tra)

  python tools/prepare_gmdcsa24.py --root data/gmdcsa24 --out data/labels_gmd.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

FALL_RE = re.compile(r"Fall\w*\s*\(?[A-Za-z]*\)?\s*\[\s*([\d.]+)\s*(?:to)?\s*([\d.]+)\s*\]")
SEG_RE = re.compile(r"\[\s*([\d.]+)\s*(?:to)?\s*([\d.]+)\s*\]")


def fall_interval(classes: str, length: float) -> tuple[float, float]:
    """Đoạn ngã (bắt đầu, kết thúc). Nhiều đoạn ngã liền nhau → gộp.
    Không ghi thời điểm (vd "Falling (FW); Sitting[0 to 1.7]") → từ cuối đoạn khác đến hết clip."""
    m = FALL_RE.findall(classes)
    if m:
        return min(float(a) for a, _ in m), max(float(b) for _, b in m)
    others = [float(b) for _, b in SEG_RE.findall(classes)]
    return (max(others) if others else 0.0), length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/gmdcsa24")
    ap.add_argument("--out", default="data/labels_gmd.csv")
    ap.add_argument("--test-subjects", nargs="+", default=["4"])
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for sub in sorted(root.glob("Subject *")):
        sid = sub.name.split()[-1]
        split = "test_gmd" if sid in args.test_subjects else "train_gmd"
        for kind in ("Fall", "ADL"):
            meta = sub / f"{kind}.csv"
            if not meta.exists():
                continue
            with open(meta, newline="", encoding="utf-8-sig") as fh:
                for r in csv.reader(fh):
                    if not r or not r[0].strip().lower().endswith(".mp4"):
                        continue
                    name = r[0].strip()
                    length = float(re.sub(r"[^\d.]", "", r[1]) or 0)
                    classes = ",".join(r[5:])
                    desc = r[4].strip().replace(",", ";") if len(r) > 4 else ""
                    video = f"{sub.name}/{kind}/{name}"
                    if kind == "Fall":
                        s, e = fall_interval(classes, length)
                        rows.append([video, split, f"{s:.2f}", f"{e:.2f}", desc])
                    else:
                        rows.append([video, split, "", "", desc])
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "split", "fall_start", "fall_end", "note"])
        w.writerows(rows)
    n_fall = sum(1 for r in rows if r[2])
    print(f"Đã ghi {args.out}: {len(rows)} video ({n_fall} ngã, {len(rows) - n_fall} sinh hoạt)")


if __name__ == "__main__":
    main()
