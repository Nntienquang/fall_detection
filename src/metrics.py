"""Đánh giá ở mức SỰ KIỆN (TV4).

Một cú ngã thật (khoảng [start, end] trong nhãn) được tính là phát hiện đúng (TP) nếu có
ít nhất 1 cảnh báo FALL trong khoảng [start - pre_tol, end + post_tol].
Cảnh báo không khớp với cú ngã nào → cảnh báo sai (FP).
Cú ngã không có cảnh báo nào → bỏ sót (FN).
Độ trễ = thời điểm cảnh báo đầu tiên khớp − start.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MatchResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    latencies: list[float] = field(default_factory=list)

    def __iadd__(self, o: "MatchResult"):
        self.tp += o.tp
        self.fp += o.fp
        self.fn += o.fn
        self.latencies += o.latencies
        return self

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def mean_latency(self) -> float | None:
        return sum(self.latencies) / len(self.latencies) if self.latencies else None


def match_events(alerts: list[float], falls: list[tuple[float, float]],
                 pre_tol: float = 0.5, post_tol: float = 3.0) -> MatchResult:
    res = MatchResult()
    used = set()
    for start, end in sorted(falls):
        hits = [i for i, a in enumerate(alerts)
                if i not in used and start - pre_tol <= a <= end + post_tol]
        if hits:
            first = min(hits, key=lambda i: alerts[i])
            res.tp += 1
            res.latencies.append(alerts[first] - start)
            used.update(hits)  # nhiều cảnh báo cho cùng 1 cú ngã chỉ tính 1 lần
        else:
            res.fn += 1
    res.fp = len(alerts) - len(used)
    return res
