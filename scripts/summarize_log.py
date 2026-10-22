"""Summarize a verifier log: what was accepted, what was rejected, and the work it implies.

Each accepted proof stands for about 1 / win_probability tile hashes. One tile
is 2 rows by 64 columns of output over the full inner dimension k, so it's
128 * k multiply-adds.

If any accepted proof names a GPU ID (the verifier ran with --attribute-gpus),
it also prints the accepted proofs and estimated multiply-adds per GPU ID.
Proofs from the shared job appear under "no GPU ID".

Example:
    python scripts/summarize_log.py runs/gpu-verifier.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

TILE_OUTPUTS = 2 * 64


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("log", type=Path)
    args = p.parse_args()

    rows = [json.loads(line) for line in args.log.read_text().splitlines() if line.strip()]
    accepted = [r for r in rows if r["accepted"]]
    rejected = Counter(r["reason"].split(":")[0] for r in rows if not r["accepted"])

    tiles = sum(1 / r["win_probability"] for r in accepted)
    macs = sum(TILE_OUTPUTS * r["k"] / r["win_probability"] for r in accepted)
    span = (max(r["time"] for r in rows) - min(r["time"] for r in rows)) if len(rows) > 1 else 0.0

    print(f"submissions: {len(rows)}  accepted: {len(accepted)}  rejected: {sum(rejected.values())}")
    for reason, n in rejected.most_common():
        print(f"  rejected, {reason}: {n}")
    print(f"distinct weight matrices proven: {len({r['weight'] for r in accepted})}")
    by_kind = Counter(r["weight"].split(".")[-2] for r in accepted)
    print("accepted by projection: " + ", ".join(f"{k} {n}" for k, n in sorted(by_kind.items())))
    print(f"activation rows (m) seen: {sorted({r['m'] for r in accepted})}")
    print(f"estimated tiles: {tiles:.3g}  estimated multiply-adds: {macs:.3g}")
    if span > 0:
        print(f"over {span:.0f} s between first and last submission: {macs / span / 1e12:.1f} T multiply-adds/s")

    if any(r.get("gpu_id") for r in accepted):
        per_gpu: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
        for r in accepted:
            totals = per_gpu[r.get("gpu_id") or "no GPU ID"]
            totals[0] += 1
            totals[1] += TILE_OUTPUTS * r["k"] / r["win_probability"]
        print("by GPU ID (self-reported, not verified):")
        for gpu, (n, gpu_macs) in sorted(per_gpu.items()):
            print(f"  {gpu}: accepted {n}  estimated multiply-adds: {gpu_macs:.3g}")


if __name__ == "__main__":
    main()
