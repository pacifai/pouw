"""Replay a recorded H100 run through a real verifier, and run attacks against it.

The replay clock runs from just before vLLM started serving to the last
recorded proof. Every recorded proof goes through `Verifier.check` with `now`
set to its recorded arrival time, so each verdict is a fresh check, not a
copy of the live verdict. Proofs that arrived before vLLM started serving
came from its warm-up passes on dummy inputs, and are labeled so.

Each attack builds a submission and sends it through the same verifier at the
current replay time. Some reuse a recorded H100 proof (replay, precomputation,
difficulty tampering, forged proof, proof theft). Others need a fresh proof,
which the laptop builds on the CPU with `pouw.cpu_prover` (model substitution,
synthetic inputs, GPU ID spoofing). A CPU can't find wins at the H100's
target, so those laptop jobs use an easier target of 1 in 50 tiles. The work
estimate accounts for each job's own target, so laptop proofs add little.

A run recorded through scripts/gpu/cheat_relay.py also has cheats.jsonl,
which names the cheat behind each submission the relay changed. Replay labels
those submissions with it.

LiveEngine takes proofs as the prover sends them instead of from a recording,
and records how long each one waited for its verdict.
"""

from __future__ import annotations

import base64
import hashlib
import json
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from pearl_mining import PlainProof

from ..cpu_prover import prove, random_activations
from ..jobs import JobIssuer, new_job, nbits_for_win_probability
from ..recording import Recorder, load, replay_verifier
from ..verifier import Verdict, Verifier
from ..weights import DeclaredWeights

TILE_OUTPUTS = 2 * 64
LAPTOP_WIN = 1 / 50
LAPTOP_M = 128
# Byte offset of the first opened row of A in a serialized proof: four 8-byte
# sizes (m, n, k, noise rank), then A's leaf count and leaf length, 8 bytes
# each. The leaves hold the opened rows' bytes in order.
A_LEAF_OFFSET = 48

# The verifier's checks, in order, and the reason prefixes each one produces.
CHECKS = [
    ("Job binding", ("unknown job", "job fields differ")),
    ("Freshness", ("stale job",)),
    ("Proof verification", ("malformed proof", "MoE proofs", "proof verification failed")),
    ("Weight commitment", ("weight matrix is not",)),
    ("Replay protection", ("duplicate",)),
]

ATTACKS = {
    # Detected
    "replay": "Replay",
    "precompute": "Precomputation",
    "difficulty": "Difficulty tampering",
    "forged": "Forged proof",
    "model": "Model substitution",
    "theft": "Proof theft",
    # Not detected
    "synthetic": "Synthetic inputs",
    "spoof": "GPU ID spoofing",
}


@dataclass
class Entry:
    id: int
    time: float
    source: str
    accepted: bool
    reason: str
    gpu_id: str | None
    weight: str | None
    m: int | None
    k: int | None
    win_probability: float | None
    attack: str | None = None
    changes: list[dict] = field(default_factory=list)
    # Seconds from arrival to verdict. Only live mode measures it.
    delay_s: float | None = None

    @property
    def macs(self) -> float:
        if not self.accepted or not self.win_probability or not self.k:
            return 0.0
        return TILE_OUTPUTS * self.k / self.win_probability


def checks_for(accepted: bool, reason: str) -> list[dict]:
    """Each check's outcome: pass, fail, or not reached."""
    failed = None if accepted else next((i for i, (_, prefixes) in enumerate(CHECKS) if reason.startswith(prefixes)), None)
    out = []
    for i, (name, _) in enumerate(CHECKS):
        status = "pass" if failed is None or i < failed else ("fail" if i == failed else "not reached")
        out.append({"name": name, "status": status})
    return out


def submission_hash(params: dict) -> str:
    """Identifies a submission: the same function the cheat relay uses."""
    job = json.dumps(params["mining_job"], sort_keys=True)
    return hashlib.sha256((params["plain_proof"] + "|" + job).encode()).hexdigest()


class PassGrouper:
    """Group the plugin's per-GEMM shape log into forward passes.

    Consecutive int7 GEMMs with the same m belong to one pass, up to 80 of
    them (the model's 80 int7 GEMMs per pass). Each pass records its time, m,
    whether it used the noisy multiply, and its multiply-adds.
    """

    def __init__(self) -> None:
        self.passes: list[dict] = []
        self._current: dict | None = None

    def add(self, r: dict) -> None:
        c = self._current
        if c is None or r["m"] != c["m"] or c["gemms"] >= 80:
            c = self._current = {"time": r["time"], "m": r["m"], "noisy": r["noisy"], "macs": 0, "gemms": 0}
            self.passes.append(c)
        c["macs"] += r["m"] * r["n"] * r["k"]
        c["gemms"] += 1
        c["noisy"] = c["noisy"] and r["noisy"]


def load_passes(path: Path) -> list[dict]:
    if not path.exists():
        return []
    g = PassGrouper()
    for line in path.read_text().splitlines():
        if line.strip():
            g.add(json.loads(line))
    return g.passes


def load_cheats(path: Path) -> dict[str, dict]:
    """The relay's log of changed submissions, by submission hash."""
    if not path.exists():
        return {}
    marks = {}
    for line in path.read_text().splitlines():
        if line.strip():
            mark = json.loads(line)
            marks[mark["hash"]] = mark
    return marks


def load_timing(runs: Path) -> dict | None:
    """Median benchmark duration for the plain, proofs-off, and proofs-on runs."""

    def durations(prefix: str) -> list[float]:
        out = []
        for d in sorted(runs.glob(f"{prefix}-*")):
            for f in d.glob("openai-*.json"):
                out.append(json.loads(f.read_text())["duration"])
        return out

    # demo_session2.sh names its proofs-on runs ab-patched-* and ab-unpatched-*.
    plain, nosubmit = durations("plain"), durations("nosubmit")
    noisy = durations("noisy") or durations("ab-patched")
    unpatched = durations("ab-unpatched")
    if not plain or not noisy:
        return None
    p, n = statistics.median(plain), statistics.median(noisy)
    return {
        "plain_s": p,
        "proofs_off_s": statistics.median(nosubmit) if nosubmit else None,
        "proofs_on_s": n,
        "proofs_on_unpatched_s": statistics.median(unpatched) if unpatched else None,
        "overhead": n / p - 1,
        "runs": {"plain": len(plain), "proofs_off": len(nosubmit), "proofs_on": len(noisy), "unpatched": len(unpatched)},
    }


class Engine:
    mode = "replay"

    def __init__(self, run_dir: Path, weights: DeclaredWeights, speed: float = 1.0, timing_dir: Path | None = None):
        self.label = run_dir.name
        description = run_dir / "description.txt"
        self.description = description.read_text().strip() if description.exists() else ""
        self.cheat_marks = load_cheats(run_dir / "cheats.jsonl")
        self.weights = weights
        self.speed = speed
        self.recording = load(run_dir / "recording.jsonl")
        self.submissions = sorted(self.recording.submissions, key=lambda s: s[0])
        serving = run_dir / "serving_since"
        self.serving_since = float(serving.read_text()) if serving.exists() else None
        first = self.submissions[0][0] if self.submissions else time.time()
        self.t_start = (self.serving_since - 2.0) if self.serving_since else first
        self.t_end = max([s[0] for s in self.submissions] + [self.t_start]) + 5.0
        # Passes before vLLM started serving are its warm-up on dummy inputs.
        self.passes = [p for p in load_passes(run_dir / "gemm_shapes.jsonl") if p["time"] >= (self.serving_since or 0)]
        self.timing = load_timing(timing_dir) if timing_dir else None
        self.gpu_ids = sorted({j.gpu_id for j in self.recording.jobs if j.gpu_id})
        self.laptop_matrix = next(name for name in sorted(weights.tensors) if name.endswith("o_proj.weight"))
        self._lock = threading.RLock()
        self.ready = False
        self.reset()

    def prepare(self) -> None:
        """Hash the declared weights under every recorded job once, so replay doesn't stall.

        Checks each recorded proof with a throwaway verifier. The weight roots
        it computes are cached in the shared DeclaredWeights.
        """
        scratch = replay_verifier(self.recording, self.weights)
        for arrival, params in self.submissions:
            scratch.check(params["plain_proof"], params["mining_job"], now=arrival)
        self.ready = True

    # The replay clock

    def reset(self) -> None:
        with self._lock:
            self.verifier = replay_verifier(self.recording, self.weights)
            self.entries: list[Entry] = []
            self.params: dict[int, dict] = {}  # recorded params, by entry id
            self.next_sub = 0
            self.next_pass = 0
            self._seen_hashes: set[str] = set()
            self.started_wall: float | None = None
            self.clock = self.t_start

    def start(self) -> None:
        with self._lock:
            if self.started_wall is None and self.ready:
                self.started_wall = time.time()

    def tick(self) -> None:
        with self._lock:
            if self.started_wall is None:
                return
            self.clock = min(self.t_start + (time.time() - self.started_wall) * self.speed, self.t_end)
            due = []
            while self.next_sub < len(self.submissions) and (
                self.submissions[self.next_sub][0] <= self.clock or self.submissions[self.next_sub][0] < self.t_start
            ):
                due.append(self.submissions[self.next_sub])
                self.next_sub += 1
        for arrival, params in due:  # check outside the lock: the first proof of a job hashes every weight
            verdict = self.verifier.check(params["plain_proof"], params["mining_job"], now=arrival)
            warmup = self.serving_since is not None and arrival < self.serving_since
            mark = None
            if self.cheat_marks:
                h = submission_hash(params)
                mark = self.cheat_marks.get(h)
                if mark and mark.get("repeat") and h not in self._seen_hashes:
                    mark = None  # the honest first copy
                self._seen_hashes.add(h)
            source = "H100 warm-up" if warmup else ("H100 relay" if mark else "H100")
            entry = self._add(arrival, source, verdict, mark and mark["cheat"], mark and mark["changes"])
            self.params[entry.id] = {"time": arrival, **params}

    def _add(self, t: float, source: str, verdict: Verdict, attack: str | None = None, changes=None, delay_s=None) -> Entry:
        with self._lock:
            entry = Entry(
                id=len(self.entries),
                time=t,
                source=source,
                accepted=verdict.accepted,
                reason=verdict.reason,
                gpu_id=verdict.gpu_id,
                weight=verdict.weight,
                m=verdict.m,
                k=verdict.k,
                win_probability=verdict.win_probability,
                attack=attack,
                changes=changes or [],
                delay_s=delay_s,
            )
            self.entries.append(entry)
            return entry

    # What the page shows

    def state(self) -> dict:
        with self._lock:
            clock = self.clock
            entries = list(self.entries)
            passes = [p for p in self.passes if p["time"] <= clock]
        accepted = [e for e in entries if e.accepted]
        delays = [e.delay_s for e in entries if e.delay_s is not None and e.time >= clock - 60]
        noisy_macs = sum(p["macs"] for p in passes if p["noisy"])
        all_macs = sum(p["macs"] for p in passes)
        per_gpu: dict[str, dict] = {}
        for e in accepted:
            row = per_gpu.setdefault(e.gpu_id or "no GPU ID", {"proofs": 0, "macs": 0.0, "sources": set()})
            row["proofs"] += 1
            row["macs"] += e.macs
            row["sources"].add("laptop" if e.source == "laptop" else "GPU")
        bins: dict[int, list[int]] = {}
        for p in passes:
            b = int(p["time"] - self.t_start)
            if b < 0:
                continue
            slot = bins.setdefault(b, [0, 0])
            slot[0 if p["noisy"] else 1] += p["m"]
        return {
            "mode": self.mode,
            "label": self.label,
            "description": self.description,
            "ready": self.ready,
            "started": self.started_wall is not None,
            "finished": self.t_end is not None and clock >= self.t_end,
            "elapsed_s": clock - self.t_start,
            "duration_s": None if self.t_end is None else self.t_end - self.t_start,
            "delay": {"max_s": max(delays), "median_s": statistics.median(delays)} if delays else None,
            "gpu_ids": self.gpu_ids,
            "totals": {
                "tokens": sum(p["m"] for p in passes),
                "accepted": len(accepted),
                "rejected": len(entries) - len(accepted),
                "warmup_duplicates": sum(1 for e in entries if e.source == "H100 warm-up" and e.reason.startswith("duplicate")),
                "proven_macs": sum(e.macs for e in accepted),
                "proven_share": noisy_macs / all_macs if all_macs else None,
            },
            "timing": self.timing,
            "throughput": [
                {"t": b, "noisy": v[0], "plain": v[1]} for b, v in sorted(bins.items()) if b >= (clock - self.t_start) - 120
            ],
            "batches": [{"m": p["m"], "noisy": p["noisy"]} for p in passes[-160:]],
            "log": [self._row(e) for e in reversed(entries[-200:])],
            "per_gpu": [
                {"gpu_id": g, "proofs": r["proofs"], "macs": r["macs"], "sources": sorted(r["sources"])}
                for g, r in sorted(per_gpu.items())
            ],
        }

    def _row(self, e: Entry) -> dict:
        row = asdict(e)
        row["t"] = e.time - self.t_start
        row["macs"] = e.macs
        return row

    def entry(self, entry_id: int) -> dict | None:
        with self._lock:
            if not 0 <= entry_id < len(self.entries):
                return None
            e = self.entries[entry_id]
        return {**self._row(e), "checks": checks_for(e.accepted, e.reason)}

    # Attacks

    def attack(self, name: str) -> int:
        if name not in ATTACKS:
            raise ValueError(f"unknown attack: {name}")
        return getattr(self, f"_attack_{name}")()

    def _recent_h100_proof(self) -> tuple[Entry, dict]:
        with self._lock:
            for e in reversed(self.entries):
                if e.source == "H100" and e.accepted and e.attack is None and e.id in self.params:
                    return e, self.params[e.id]
        raise ValueError("no accepted H100 proof yet: start the replay first")

    def _attack_replay(self) -> int:
        e, p = self._recent_h100_proof()
        v = self.verifier.check(p["plain_proof"], p["mining_job"], now=p["time"] + 1.0)
        changes = [{"field": "submitted", "original": f"t = {e.time - self.t_start:.1f} s", "submitted": "1.0 s later, unchanged"}]
        return self._add(self.clock, "H100", v, ATTACKS["replay"], changes).id

    def _attack_precompute(self) -> int:
        e, p = self._recent_h100_proof()
        job = self.verifier.issuer.lookup(base64.b64decode(p["mining_job"]["incomplete_header_bytes"]))
        held = job.issued_at + self.verifier.issuer.max_age_s + 1.0
        v = self.verifier.check(p["plain_proof"], p["mining_job"], now=held)
        changes = [
            {
                "field": "submitted",
                "original": f"{p['time'] - job.issued_at:.1f} s after the job was issued",
                "submitted": f"{held - job.issued_at:.1f} s after (limit {self.verifier.issuer.max_age_s:.0f} s)",
            }
        ]
        return self._add(self.clock, "H100", v, ATTACKS["precompute"], changes).id

    def _attack_difficulty(self) -> int:
        e, p = self._recent_h100_proof()
        job = dict(p["mining_job"])
        job["target"] = job["target"] * 256
        v = self.verifier.check(p["plain_proof"], job, now=p["time"] + 0.5)
        changes = [{"field": "job target", "original": f"{p['mining_job']['target']:.3e}", "submitted": f"{job['target']:.3e} (256 times easier)"}]
        return self._add(self.clock, "H100", v, ATTACKS["difficulty"], changes).id

    def _attack_forged(self) -> int:
        e, p = self._recent_h100_proof()
        raw = bytearray(base64.b64decode(p["plain_proof"]))
        row = list(PlainProof.from_base64(p["plain_proof"]).a.row_indices)[0]
        i = A_LEAF_OFFSET + 100
        before = (raw[i] + 128) % 256 - 128  # the byte as a signed int8
        raw[i] = (raw[i] + 1) % 256
        after = (raw[i] + 128) % 256 - 128
        v = self.verifier.check(base64.b64encode(bytes(raw)).decode("ascii"), p["mining_job"], now=p["time"] + 0.5)
        changes = [{"field": f"activation A[{row}, 100]", "original": str(before), "submitted": str(after)}]
        return self._add(self.clock, "H100", v, ATTACKS["forged"], changes).id

    def _attack_theft(self) -> int:
        e, p = self._recent_h100_proof()
        with self._lock:
            job = new_job(self.verifier.issuer.nbits, now=self.clock, gpu_id="GPU-B (laptop)")
            self.verifier.issuer._jobs[job.header_bytes] = job
        v = self.verifier.check(p["plain_proof"], job.to_dict(), now=self.clock + 0.5)
        changes = [{"field": "job (GPU ID)", "original": e.gpu_id or "none", "submitted": "GPU-B (laptop)"}]
        return self._add(self.clock, "laptop", v, ATTACKS["theft"], changes).id

    def _laptop_proof(self, gpu_id: str | None, Bt: np.ndarray, seed: int) -> Verdict:
        k = Bt.shape[1]
        with self._lock:
            job = new_job(nbits_for_win_probability(LAPTOP_WIN, k), now=self.clock, gpu_id=gpu_id)
            self.verifier.issuer._jobs[job.header_bytes] = job
        proof, _ = prove(job.header_bytes, job.cert_version, random_activations(LAPTOP_M, k, seed=seed), Bt, seed=seed)
        if proof is None:
            raise RuntimeError("the laptop prover found no winning tile")
        return self.verifier.check(proof.to_base64(), job.to_dict(), now=job.issued_at + 1.0)

    def _attack_model(self) -> int:
        Bt = self.weights.tensors[self.laptop_matrix].copy()
        before = int(Bt[100, 200])
        Bt[100, 200] = before + 1 if before < 63 else before - 1
        v = self._laptop_proof(None, Bt, seed=len(self.entries))
        changes = [{"field": f"{self.laptop_matrix} [100, 200]", "original": str(before), "submitted": str(int(Bt[100, 200]))}]
        return self._add(self.clock, "laptop", v, ATTACKS["model"], changes).id

    def _attack_synthetic(self) -> int:
        v = self._laptop_proof(None, self.weights.tensors[self.laptop_matrix], seed=len(self.entries))
        changes = [{"field": "activations A", "original": "from requests", "submitted": f"random, {LAPTOP_M} rows"}]
        return self._add(self.clock, "laptop", v, ATTACKS["synthetic"], changes).id

    def _attack_spoof(self) -> int:
        target = self.gpu_ids[0] if self.gpu_ids else "GPU-00000000"
        v = self._laptop_proof(target, self.weights.tensors[self.laptop_matrix], seed=len(self.entries))
        changes = [{"field": "GPU ID sent", "original": "laptop (no GPU)", "submitted": target}]
        return self._add(self.clock, "laptop", v, ATTACKS["spoof"], changes).id


class LiveEngine(Engine):
    """Proofs arrive from a running prover. The server hands each one to `submit`."""

    mode = "live"

    def __init__(
        self,
        weights: DeclaredWeights,
        issuer: JobIssuer,
        attribute_gpus: bool = False,
        shapes_path: Path | None = None,
        serving_since_path: Path | None = None,
        log_path: Path | None = None,
        recorder: Recorder | None = None,
    ):
        self.label = "live"
        self.description = ""
        self.cheat_marks = {}
        self.weights = weights
        self.speed = 1.0
        self.issuer = issuer
        self.attribute_gpus = attribute_gpus
        self.shapes_path = shapes_path
        self._shapes_pos = 0
        self._grouper = PassGrouper()
        self.serving_since_path = serving_since_path
        self.serving_since: float | None = None
        self.log_path = log_path
        self.recorder = recorder
        self.timing = None
        self.gpu_ids: list[str] = []
        self.laptop_matrix = next(name for name in sorted(weights.tensors) if name.endswith("o_proj.weight"))
        self._lock = threading.RLock()
        self.ready = True
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.verifier = Verifier(self.issuer, self.weights, attribute_gpus=self.attribute_gpus)
            self.entries = []
            self.params = {}
            self.started_wall = time.time()
            self.t_start = time.time()
            self.t_end = None
            self.clock = self.t_start

    @property
    def passes(self) -> list[dict]:
        return [p for p in self._grouper.passes if p["time"] >= (self.serving_since or 0)]

    def prepare(self) -> None:
        pass

    def start(self) -> None:
        pass

    def tick(self) -> None:
        with self._lock:
            self.clock = time.time()
            if self.serving_since is None and self.serving_since_path and self.serving_since_path.exists():
                text = self.serving_since_path.read_text().strip()
                self.serving_since = float(text) if text else None
            if self.shapes_path and self.shapes_path.exists():
                with self.shapes_path.open() as f:
                    f.seek(self._shapes_pos)
                    chunk = f.read()
                # Keep a partial last line for the next read.
                complete = chunk[: chunk.rfind("\n") + 1]
                self._shapes_pos += len(complete.encode())
                for line in complete.splitlines():
                    if line.strip():
                        self._grouper.add(json.loads(line))

    def submit(self, params: dict) -> Verdict:
        arrival = time.time()
        if self.recorder is not None:
            self.recorder.submission(arrival, params)
        verdict = self.verifier.check(params["plain_proof"], params["mining_job"])
        delay = time.time() - arrival
        warmup = self.serving_since_path is not None and self.serving_since is None
        entry = self._add(arrival, "H100 warm-up" if warmup else "H100", verdict, delay_s=delay)
        with self._lock:
            self.params[entry.id] = {"time": arrival, **params}
            if verdict.gpu_id and verdict.gpu_id not in self.gpu_ids:
                self.gpu_ids.append(verdict.gpu_id)
        if self.log_path is not None:
            with self.log_path.open("a") as f:
                f.write(json.dumps({"time": time.time(), **asdict(verdict)}) + "\n")
        return verdict
