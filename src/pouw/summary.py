"""Summarize a verifier log: verdicts by check, and the work the accepted proofs imply."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

TERA = 1e12


def summarize(log: Path) -> str:
    rows = log.read_text().splitlines() if log.exists() else []
    verdicts = [json.loads(row) for row in rows if row.strip()]
    if not verdicts:
        return "no submissions"
    accepted = [v for v in verdicts if v["accepted"]]
    rejected = Counter(v["check"] for v in verdicts if not v["accepted"])
    lines = [f"submissions: {len(verdicts)}  accepted: {len(accepted)}  rejected: {sum(rejected.values())}"]
    lines += [f"  failed {check}: {n}" for check, n in rejected.most_common()]
    if accepted:
        lines += _work(accepted, verdicts)
    return "\n".join(lines)


def _work(accepted: list[dict], verdicts: list[dict]) -> list[str]:
    tiles = sum(1 / v["win_probability"] for v in accepted)
    macs = sum(v["work"] for v in accepted)
    by_projection = Counter(v["weight"].split(".")[-2] for v in accepted)
    lines = [
        f"distinct weight matrices proven: {len({v['weight'] for v in accepted})}",
        "accepted by projection: " + ", ".join(f"{p} {n}" for p, n in sorted(by_projection.items())),
        f"activation rows (m) seen: {sorted({v['m'] for v in accepted})}",
        f"estimated tiles: {tiles:.3g}  estimated multiply-adds: {macs:.3g}",
    ]
    span = max(v["time"] for v in verdicts) - min(v["time"] for v in verdicts)
    if span > 0:
        lines.append(f"over {span:.0f} s between first and last submission: {macs / span / TERA:.1f} T multiply-adds/s")
    if any(v.get("gpu_id") for v in accepted):
        lines += ["by GPU ID (self-reported):", *_per_gpu(accepted)]
    return lines


def _per_gpu(accepted: list[dict]) -> list[str]:
    totals: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    for v in accepted:
        gpu = totals[v.get("gpu_id") or "no GPU ID"]
        gpu[0] += 1
        gpu[1] += v["work"]
    return [f"  {gpu}: accepted {n}  estimated multiply-adds: {macs:.3g}" for gpu, (n, macs) in sorted(totals.items())]
