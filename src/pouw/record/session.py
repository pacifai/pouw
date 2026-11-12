"""The session file: recorded runs, every proof checked again, in the one file the demo site reads.

Each run directory comes from scripts/run_e2e.sh with RECORD=1. It holds
run.json, recording.jsonl, and serving_since, and may hold cheats.jsonl from
the relay and gemm_shapes.jsonl from patches/0003. Times in the session file
are seconds since vLLM started serving, so vLLM's warm-up proofs have negative
times.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..cheats import CHEATS
from ..cheats.relay import submission_hash
from ..verifier import CHECKS, Verdict
from ..weights import DeclaredWeights
from .recorder import load, replay_verifier

FORMAT = "pouw.session.v1"
CHANGED_MODEL = "changed_model"
CHANGED_MODEL_DESCRIPTION = "Serves a copy of the declared model with one weight changed in every int8 matrix"
TIME_DECIMALS = 3
# How long the demo's timeline runs past a run's last proof.
TIMELINE_TAIL_S = 5.0


def write(run_dirs: list[Path], out: Path, gpu: str | None, weights: list[Path] | None = None) -> dict:
    session = build(run_dirs, gpu, weights)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(session, separators=(",", ":")))
    return session


def build(run_dirs: list[Path], gpu: str | None, weights: list[Path] | None = None) -> dict:
    declared: dict[str, DeclaredWeights] = {}
    runs = []
    for run_dir in run_dirs:
        model = _read_json(run_dir / "run.json")["declared_model"]
        if model not in declared:
            declared[model] = DeclaredWeights.from_safetensors(weights or sorted(Path(model).glob("*.safetensors")))
        runs.append(build_run(run_dir, declared[model]))
    return {
        "format": FORMAT,
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gpu": gpu,
        "checks": CHECKS,
        "cheats": {**CHEATS, CHANGED_MODEL: CHANGED_MODEL_DESCRIPTION},
        "runs": runs,
    }


def build_run(run_dir: Path, weights: DeclaredWeights) -> dict:
    meta = _read_json(run_dir / "run.json")
    serving_since = float((run_dir / "serving_since").read_text())
    recording = load(run_dir / "recording.jsonl")
    changed_model = meta["model"] != meta["declared_model"]
    proofs = _check_again(run_dir, recording, weights, serving_since, changed_model)
    settings = recording.settings
    return {
        "name": run_dir.name,
        "description": meta["description"],
        "model": meta["model"],
        "declared_model": meta["declared_model"],
        "settings": {key: settings.get(key) for key in ("win_probability", "interval_s", "max_age_s")},
        "duration_s": round(max((p["t"] for p in proofs), default=0) + TIMELINE_TAIL_S, 1),
        "proven_share": proven_share(run_dir / "gemm_shapes.jsonl", serving_since),
        "proofs": proofs,
    }


def _check_again(run_dir, recording, weights, serving_since: float, changed_model: bool) -> list[dict]:
    verifier = replay_verifier(recording, weights)
    cheats = CheatLabels(run_dir / "cheats.jsonl")
    proofs = []
    for arrival, submission in recording.submissions:
        verdict = verifier.check(submission["plain_proof"], submission["mining_job"], now=arrival)
        mark = cheats.label(submission)
        cheat = CHANGED_MODEL if changed_model else mark and mark["cheat"]
        proofs.append(_proof(round(arrival - serving_since, TIME_DECIMALS), verdict, cheat, mark))
    return proofs


def _proof(t: float, v: Verdict, cheat: str | None, mark: dict | None) -> dict:
    return {
        "t": t,
        "accepted": v.accepted,
        "check": v.check,
        "reason": v.reason,
        "cheat": cheat,
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


class CheatLabels:
    """The relay's log, keyed by submission. A cheat that resends a proof unchanged marks only its second arrival."""

    def __init__(self, path: Path):
        lines = path.read_text().splitlines() if path.exists() else []
        self._marks = {m["hash"]: m for m in map(json.loads, filter(str.strip, lines))}
        self._seen: set[str] = set()

    def label(self, submission: dict) -> dict | None:
        h = submission_hash(submission)
        mark = self._marks.get(h)
        first_arrival = h not in self._seen
        self._seen.add(h)
        if mark and mark["repeat"] and first_arrival:
            return None
        return mark


def proven_share(shapes: Path, since: float) -> float | None:
    """Share of int7 multiply-adds after since that ran the noisy multiply, from patches/0003's log."""
    if not shapes.exists():
        return None
    noisy = total = 0
    for line in shapes.read_text().splitlines():
        gemm = json.loads(line)
        if gemm["time"] >= since:
            work = gemm["m"] * gemm["n"] * gemm["k"]
            total += work
            noisy += work if gemm["noisy"] else 0
    return noisy / total if total else None


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())
