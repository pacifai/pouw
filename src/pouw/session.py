"""Write a session file: recorded runs, checked again, in the one file the demo site reads.

Each run directory comes from scripts/gpu/run_e2e.sh with RECORD=1. It holds
run.json, recording.jsonl, and serving_since, and may hold cheats.jsonl (from
the cheat relay) and gemm_shapes.jsonl (from patches/0003). Every recorded
proof is checked again by a verifier rebuilt from the recording, at its
recorded arrival time.

Times in the session file are seconds since vLLM started serving. Proofs with
a negative time came from vLLM's warm-up on dummy inputs.

Example:
    python -m pouw.session runs/honest runs/relay runs/changed-model --out session.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .cheat_relay import CHEATS, submission_hash
from .recording import load
from .verifier import CHECKS, Verifier
from .weights import DeclaredWeights

FORMAT = "pouw.session.v1"
CHANGED_MODEL = "Serves a copy of the declared model with one weight changed in every int8 matrix"


def proven_share(shapes: Path, since: float) -> float | None:
    """Share of int7 multiply-adds after `since` that ran the noisy multiply, from patches/0003's log."""
    if not shapes.exists():
        return None
    noisy = total = 0
    for line in shapes.read_text().splitlines():
        r = json.loads(line)
        if r["time"] >= since:
            work = r["m"] * r["n"] * r["k"]
            total += work
            noisy += work if r["noisy"] else 0
    return noisy / total if total else None


def cheat_marks(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    marks = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {m["hash"]: m for m in marks}


def export_run(run_dir: Path, weights: DeclaredWeights) -> dict:
    meta = json.loads((run_dir / "run.json").read_text())
    serving = float((run_dir / "serving_since").read_text())
    recording = load(run_dir / "recording.jsonl")
    verifier = Verifier.from_recording(recording, weights)
    changed_model = meta["model"] != meta["declared_model"]
    marks = cheat_marks(run_dir / "cheats.jsonl")
    seen: set[str] = set()
    proofs = []
    for arrival, params in recording.submissions:
        v = verifier.check(params["plain_proof"], params["mining_job"], now=arrival)
        h = submission_hash(params)
        mark = marks.get(h)
        if mark and mark["repeat"] and h not in seen:
            mark = None
        seen.add(h)
        proofs.append(
            {
                "t": round(arrival - serving, 3),
                "accepted": v.accepted,
                "check": v.check,
                "reason": v.reason,
                "cheat": "changed_model" if changed_model else mark and mark["cheat"],
                "changes": mark["changes"] if mark else [],
                "weight": v.weight,
                "m": v.m,
                "n": v.n,
                "k": v.k,
                "row": v.a_rows[0] if v.a_rows else None,
                "col": v.b_cols[0] if v.b_cols else None,
                "gpu_id": v.gpu_id,
                "work": round(v.work),
            }
        )
    s = recording.settings
    return {
        "name": run_dir.name,
        "description": meta["description"],
        "model": meta["model"],
        "declared_model": meta["declared_model"],
        "settings": {k: s.get(k) for k in ("win_probability", "interval_s", "max_age_s")},
        "duration_s": round(max((p["t"] for p in proofs), default=0) + 5, 1),
        "proven_share": proven_share(run_dir / "gemm_shapes.jsonl", serving),
        "proofs": proofs,
    }


def export(run_dirs: list[Path], gpu: str | None, weights: list[Path] | None = None) -> dict:
    loaded: dict[str, DeclaredWeights] = {}
    runs = []
    for run_dir in run_dirs:
        declared = json.loads((run_dir / "run.json").read_text())["declared_model"]
        if declared not in loaded:
            loaded[declared] = DeclaredWeights.from_safetensors(weights or sorted(Path(declared).glob("*.safetensors")))
        runs.append(export_run(run_dir, loaded[declared]))
    return {
        "format": FORMAT,
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gpu": gpu,
        "checks": CHECKS,
        "cheats": {**CHEATS, "changed_model": CHANGED_MODEL},
        "runs": runs,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Check recorded runs again and write them into one session file.")
    p.add_argument("runs", nargs="+", type=Path, help="run directories written by scripts/gpu/run_e2e.sh with RECORD=1")
    p.add_argument("--out", type=Path, required=True, help="session file to write")
    p.add_argument("--gpu", help="the GPU's name, as nvidia-smi reports it")
    p.add_argument("--weights", nargs="+", type=Path, help="declared weights, if not at the path each run.json names")
    args = p.parse_args()
    session = export(args.runs, args.gpu, args.weights)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(session, separators=(",", ":")))
    for run in session["runs"]:
        accepted = sum(p["accepted"] for p in run["proofs"])
        print(f"{run['name']}: {len(run['proofs'])} proofs, {accepted} accepted")


if __name__ == "__main__":
    main()
