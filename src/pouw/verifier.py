"""The verifier: issues jobs to the prover and checks every proof it sends back.

It answers the two JSON-RPC calls upstream's plugin makes to its gateway:
getMiningInfo for the current job, and submitPlainProof for the proof of one
winning tile. A proof counts only if it passes every check in CHECKS, in order.
"""

from __future__ import annotations

import base64
import json
import socketserver
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Callable

from pearl_mining import IncompleteBlockHeader, PlainProof, verify_plain_proof_for_cert_version

from .jobs import MAX_GPU_ID_LEN, Job, JobIssuer
from .proof import TILE_OUTPUTS, config_from_indices, job_key
from .protocol import answer, start_server
from .target import win_probability
from .weights import DeclaredWeights

CHECKS = {
    "issued_job": "The proof's job was issued by this verifier, with its target and version unchanged",
    "on_time": "The job is no older than the maximum age",
    "valid_proof": "Upstream's verify function recomputes the tile from the opened rows and columns, and it wins",
    "declared_weights": "The weight matrix's Merkle root matches one of the declared model's matrices",
    "new_tile": "Nobody submitted this tile before",
}


@dataclass
class Verdict:
    accepted: bool
    reason: str
    check: str | None = None
    job_id: str | None = None
    weight: str | None = None
    m: int | None = None
    n: int | None = None
    k: int | None = None
    a_rows: list[int] | None = None
    b_cols: list[int] | None = None
    win_probability: float | None = None
    # Multiply-adds the proof stands for: the tile's work divided by its chance to win.
    work: float = 0.0
    job_age_s: float | None = None
    gpu_id: str | None = None

    def reject(self, check: str, reason: str) -> Verdict:
        self.check, self.reason = check, reason
        return self


class Verifier:
    def __init__(self, issuer: JobIssuer, weights: DeclaredWeights, attribute_gpus: bool = False):
        self.issuer = issuer
        self.weights = weights
        self.attribute_gpus = attribute_gpus
        self._seen_tiles: set[tuple] = set()
        self._lock = threading.Lock()

    def issue(self, params: dict | None = None, now: float | None = None) -> Job:
        """The job for a getMiningInfo request: the requesting GPU's own, or the shared one."""
        return self.issuer.current(now, self._gpu_id(params))

    def _gpu_id(self, params: dict | None) -> str | None:
        if not self.attribute_gpus or not isinstance(params, dict):
            return None
        gpu_id = params.get("gpu_id")
        if gpu_id is not None and not (isinstance(gpu_id, str) and 0 < len(gpu_id) <= MAX_GPU_ID_LEN):
            raise ValueError(f"gpu_id must be a string of 1 to {MAX_GPU_ID_LEN} characters")
        return gpu_id

    def check(self, plain_proof_b64: str, mining_job: dict, now: float | None = None) -> Verdict:
        now = time.time() if now is None else now
        header_bytes = base64.b64decode(mining_job["incomplete_header_bytes"])

        job = self.issuer.lookup(header_bytes)
        if job is None:
            return Verdict(False, "").reject("issued_job", "unknown job: not issued by this verifier")
        v = Verdict(False, "", job_id=job.job_id, job_age_s=now - job.issued_at, gpu_id=job.gpu_id)
        if mining_job.get("target") != job.target or mining_job.get("cert_version") != job.cert_version:
            return v.reject("issued_job", "job fields differ from what was issued")
        if now - job.issued_at > self.issuer.max_age_s:
            return v.reject("on_time", f"stale job: issued {now - job.issued_at:.1f}s ago")

        try:
            proof = PlainProof.from_base64(plain_proof_b64)
        except ValueError as e:
            return v.reject("valid_proof", f"malformed proof: {e}")
        if proof.moe is not None:
            return v.reject("valid_proof", "MoE proofs are not supported")
        v.m, v.n, v.k = proof.m, proof.n, proof.k
        v.a_rows, v.b_cols = list(proof.a.row_indices), list(proof.bt.row_indices)

        header = IncompleteBlockHeader.from_bytes(header_bytes)
        ok, message = verify_plain_proof_for_cert_version(job.cert_version, header, proof)
        if not ok:
            return v.reject("valid_proof", f"proof verification failed: {message}")

        config = config_from_indices(proof.k, proof.noise_rank, v.a_rows, v.b_cols)
        v.weight = self.weights.match(job_key(header_bytes, config), proof.n, proof.k, bytes(proof.bt.root))
        if v.weight is None:
            return v.reject("declared_weights", "weight matrix is not one of the declared model's matrices")

        tile = (header_bytes, bytes(proof.a.root), bytes(proof.bt.root), tuple(v.a_rows), tuple(v.b_cols))
        with self._lock:
            if tile in self._seen_tiles:
                return v.reject("new_tile", "duplicate: this tile was already submitted")
            self._seen_tiles.add(tile)

        v.win_probability = win_probability(job.target, config)
        v.work = TILE_OUTPUTS * proof.k / v.win_probability
        v.accepted, v.reason = True, "ok"
        return v


class VerifierService:
    """Answers the plugin's JSON-RPC calls with a Verifier, and logs one JSON line per verdict."""

    def __init__(
        self,
        verifier: Verifier,
        log_path: Path | None = None,
        on_submission: Callable[[float, dict], None] | None = None,
    ):
        self.verifier = verifier
        self.log_path = log_path
        self.on_submission = on_submission

    def start(self, socket_path: str | None = None, tcp: tuple[str, int] | None = None) -> socketserver.BaseServer:
        return start_server(self.handle, socket_path, tcp)

    def handle(self, rfile: BinaryIO, wfile: BinaryIO) -> None:
        answer(rfile, wfile, self.dispatch)

    def dispatch(self, method: str, params: dict) -> dict:
        if method == "getMiningInfo":
            return self.verifier.issue(params).to_dict()
        if method == "submitPlainProof":
            if self.on_submission is not None:
                self.on_submission(time.time(), params)
            verdict = self.verifier.check(params["plain_proof"], params["mining_job"])
            self._log(verdict)
            return {"accepted": verdict.accepted, "reason": verdict.reason}
        raise ValueError(f"Method {method} not found")

    def _log(self, verdict: Verdict) -> None:
        line = json.dumps({"time": time.time(), **asdict(verdict)})
        print(line, flush=True)
        if self.log_path is not None:
            with self.log_path.open("a") as f:
                f.write(line + "\n")
