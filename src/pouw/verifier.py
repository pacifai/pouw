"""The verifier: stands in for the upstream gateway and checks every proof the prover sends.

The upstream vLLM plugin asks a gateway for jobs and sends it proofs, over a
local socket, with two JSON-RPC calls:

- getMiningInfo: the prover asks for the current job, a block header and a target.
- submitPlainProof: the prover sends the proof of one winning tile.

The verifier answers both. A proof counts only if it passes every check in
CHECKS, in order.

With attribute_gpus on, getMiningInfo accepts {"gpu_id": "<id>"}, and each
GPU ID gets its own jobs. The prover reports its own ID, and nothing signs it.
"""

from __future__ import annotations

import argparse
import base64
import json
import socketserver
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from pearl_mining import IncompleteBlockHeader, PlainProof, verify_plain_proof_for_cert_version

from .jobs import MAX_GPU_ID_LEN, TILE_OUTPUTS, Job, JobIssuer, nbits_for_win_probability, win_probability
from .recording import Recorder, Recording
from .rpc import answer
from .weights import DeclaredWeights, config_from_indices, job_key

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
    # The check that rejected the proof, a key of CHECKS. None when accepted.
    check: str | None = None
    job_id: str | None = None
    weight: str | None = None
    m: int | None = None
    n: int | None = None
    k: int | None = None
    a_rows: list[int] | None = None
    b_cols: list[int] | None = None
    win_probability: float | None = None
    # Multiply-adds this proof stands for: the tile's work divided by its chance to win.
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
        self._seen: set[tuple] = set()
        self._lock = threading.Lock()

    @classmethod
    def from_recording(cls, recording: Recording, weights: DeclaredWeights) -> Verifier:
        """A verifier with the recorded settings that knows every recorded job."""
        s = recording.settings
        issuer = JobIssuer(
            nbits=s.get("nbits", recording.jobs[0].nbits if recording.jobs else 0),
            interval_s=s.get("interval_s", 5.0),
            max_age_s=s.get("max_age_s", 15.0),
            forget_old=False,
        )
        for job in recording.jobs:
            issuer.remember(job)
        return cls(issuer, weights, attribute_gpus=s.get("attribute_gpus", False))

    def issue(self, params: dict | None = None, now: float | None = None) -> Job:
        """The job for a getMiningInfo request: the requesting GPU's own, or the shared one."""
        gpu_id = None
        if self.attribute_gpus and isinstance(params, dict):
            gpu_id = params.get("gpu_id")
            if gpu_id is not None and not (isinstance(gpu_id, str) and 0 < len(gpu_id) <= MAX_GPU_ID_LEN):
                raise ValueError(f"gpu_id must be a string of 1 to {MAX_GPU_ID_LEN} characters")
        return self.issuer.current(now, gpu_id)

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

        ok, message = verify_plain_proof_for_cert_version(
            job.cert_version, IncompleteBlockHeader.from_bytes(header_bytes), proof
        )
        if not ok:
            return v.reject("valid_proof", f"proof verification failed: {message}")

        config = config_from_indices(proof.k, proof.noise_rank, v.a_rows, v.b_cols)
        v.weight = self.weights.match(job_key(header_bytes, config), proof.n, proof.k, bytes(proof.bt.root))
        if v.weight is None:
            return v.reject("declared_weights", "weight matrix is not one of the declared model's matrices")

        tile = (header_bytes, bytes(proof.a.root), bytes(proof.bt.root), tuple(v.a_rows), tuple(v.b_cols))
        with self._lock:
            if tile in self._seen:
                return v.reject("new_tile", "duplicate: this tile was already submitted")
            self._seen.add(tile)

        v.win_probability = win_probability(job.target, config)
        v.work = TILE_OUTPUTS * proof.k / v.win_probability
        v.accepted, v.reason = True, "ok"
        return v


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        answer(self.rfile, self.wfile, self.server.dispatch)  # type: ignore[attr-defined]


class _Server:
    def setup_verifier(self, verifier: Verifier, log_path: Path | None, recorder: Recorder | None) -> None:
        self.verifier, self.log_path, self.recorder = verifier, log_path, recorder

    def dispatch(self, method: str, params: dict) -> dict:
        if method == "getMiningInfo":
            return self.verifier.issue(params).to_dict()
        if method == "submitPlainProof":
            if self.recorder is not None:
                self.recorder.submission(time.time(), params)
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


class UnixVerifierServer(_Server, socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(
        self, socket_path: str, verifier: Verifier, log_path: Path | None = None, recorder: Recorder | None = None
    ):
        Path(socket_path).unlink(missing_ok=True)
        super().__init__(socket_path, _Handler)
        self.setup_verifier(verifier, log_path, recorder)


class TcpVerifierServer(_Server, socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self, host: str, port: int, verifier: Verifier, log_path: Path | None = None, recorder: Recorder | None = None
    ):
        super().__init__((host, port), _Handler)
        self.setup_verifier(verifier, log_path, recorder)


def main() -> None:
    p = argparse.ArgumentParser(description="Stand in for the upstream gateway and check every proof.")
    p.add_argument(
        "--weights", nargs="+", type=Path, required=True, help="safetensors files with the declared int8 weights"
    )
    p.add_argument(
        "--socket", default="/tmp/pearlgw.sock", help="Unix socket the plugin connects to (upstream's default)"
    )
    p.add_argument("--tcp", metavar="HOST:PORT", help="listen on TCP instead of a Unix socket")
    p.add_argument("--win-probability", type=float, default=1e-4, help="chance that one tile wins")
    p.add_argument("--k", type=int, default=4096, help="inner dimension the target is set for")
    p.add_argument("--interval", type=float, default=5.0, help="seconds between new jobs")
    p.add_argument("--max-age", type=float, default=15.0, help="oldest job a proof may use, in seconds")
    p.add_argument("--log", type=Path, help="append one JSON line per verdict here")
    p.add_argument("--record", type=Path, help="append every job and every submitted proof here, for replay")
    p.add_argument(
        "--attribute-gpus", action="store_true", help="give each GPU ID its own jobs (the ID is self-reported)"
    )
    args = p.parse_args()

    for path in (args.log, args.record):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
    weights = DeclaredWeights.from_safetensors(args.weights)
    nbits = nbits_for_win_probability(args.win_probability, args.k)
    issuer = JobIssuer(nbits=nbits, interval_s=args.interval, max_age_s=args.max_age)
    recorder = None
    if args.record:
        recorder = Recorder(args.record)
        recorder.settings(issuer, args.attribute_gpus, args.win_probability)
        issuer.on_new_job = recorder.job
    verifier = Verifier(issuer, weights, attribute_gpus=args.attribute_gpus)
    attribution = "on" if args.attribute_gpus else "off"
    print(f"declared matrices: {len(weights.tensors)}; nbits={nbits:#010x}; GPU attribution {attribution}", flush=True)

    if args.tcp:
        host, port = args.tcp.rsplit(":", 1)
        server: _Server = TcpVerifierServer(host, int(port), verifier, args.log, recorder)
    else:
        server = UnixVerifierServer(args.socket, verifier, args.log, recorder)
    print(f"listening on {args.tcp or args.socket}", flush=True)
    server.serve_forever()  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
