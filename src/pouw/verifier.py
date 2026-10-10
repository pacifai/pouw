"""The verifier: stands in for the upstream gateway and checks every winning tile.

The prover, the upstream vLLM plugin, talks to a gateway over a local socket
with two JSON-RPC calls, named by upstream:

- getMiningInfo: the prover asks for the current job (header and target).
- submitPlainProof: the prover sends a winning tile's proof.

This module answers both. Each job carries a fresh random value. Each proof
must pass four checks before it counts:

1. Its job was issued here, recently, with the same target and version.
2. The library's own verify function accepts it: the tile, recomputed from the
   opened rows and the noise, hashes below the target.
3. Its weight matrix is one of the declared model's matrices.
4. It hasn't been submitted before.
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

from .jobs import JobIssuer, nbits_for_win_probability, win_probability
from .weights import DeclaredWeights, config_from_indices, job_key


@dataclass
class Verdict:
    accepted: bool
    reason: str
    job_id: str | None = None
    weight: str | None = None
    m: int | None = None
    n: int | None = None
    k: int | None = None
    a_rows: list[int] | None = None
    b_cols: list[int] | None = None
    # Chance that a single tile hash wins under this job. Summing 1/p over
    # accepted proofs estimates how many tiles the GPU computed.
    win_probability: float | None = None
    job_age_s: float | None = None


class Verifier:
    def __init__(self, issuer: JobIssuer, weights: DeclaredWeights):
        self.issuer = issuer
        self.weights = weights
        self._seen: set[tuple] = set()
        self._lock = threading.Lock()

    def check(self, plain_proof_b64: str, mining_job: dict, now: float | None = None) -> Verdict:
        now = time.time() if now is None else now
        header_bytes = base64.b64decode(mining_job["incomplete_header_bytes"])

        job = self.issuer.lookup(header_bytes)
        if job is None:
            return Verdict(False, "unknown job: not issued by this verifier")
        v = Verdict(False, "", job_id=job.job_id, job_age_s=now - job.issued_at)
        if mining_job.get("target") != job.target or mining_job.get("cert_version") != job.cert_version:
            v.reason = "job fields differ from what was issued"
            return v
        if now - job.issued_at > self.issuer.max_age_s:
            v.reason = f"stale job: issued {now - job.issued_at:.1f}s ago"
            return v

        try:
            proof = PlainProof.from_base64(plain_proof_b64)
        except ValueError as e:
            v.reason = f"malformed proof: {e}"
            return v
        if proof.moe is not None:
            v.reason = "MoE proofs are not supported yet"
            return v
        v.m, v.n, v.k = proof.m, proof.n, proof.k
        v.a_rows, v.b_cols = list(proof.a.row_indices), list(proof.bt.row_indices)

        header = IncompleteBlockHeader.from_bytes(header_bytes)
        ok, message = verify_plain_proof_for_cert_version(job.cert_version, header, proof)
        if not ok:
            v.reason = f"proof verification failed: {message}"
            return v

        config = config_from_indices(proof.k, proof.noise_rank, v.a_rows, v.b_cols)
        weight = self.weights.match(job_key(header_bytes, config), proof.n, proof.k, bytes(proof.bt.root))
        if weight is None:
            v.reason = "weight matrix is not one of the declared model's matrices"
            return v
        v.weight = weight

        fingerprint = (header_bytes, bytes(proof.a.root), bytes(proof.bt.root), tuple(v.a_rows), tuple(v.b_cols))
        with self._lock:
            if fingerprint in self._seen:
                v.reason = "duplicate: this tile was already submitted"
                return v
            self._seen.add(fingerprint)

        v.win_probability = win_probability(job.target, config)
        v.accepted, v.reason = True, "ok"
        return v


class _Handler(socketserver.StreamRequestHandler):
    """Newline-delimited JSON-RPC, matching upstream's JSONRPCClient."""

    def handle(self) -> None:
        server: _Server = self.server  # type: ignore[assignment]
        for line in self.rfile:
            if not line.strip():
                continue
            request_id = None
            try:
                request = json.loads(line)
                request_id = request.get("id")
                result = server.dispatch(request["method"], request.get("params") or {})
                response = {"jsonrpc": "2.0", "result": result, "id": request_id}
            except Exception as e:  # report every failure to the caller, as the upstream gateway does
                response = {"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}, "id": request_id}
            self.wfile.write((json.dumps(response) + "\n").encode())
            self.wfile.flush()


class _Server:
    verifier: Verifier
    log_path: Path | None

    def dispatch(self, method: str, params: dict) -> dict:
        if method == "getMiningInfo":
            return self.verifier.issuer.current().to_dict()
        if method == "submitPlainProof":
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

    def __init__(self, socket_path: str, verifier: Verifier, log_path: Path | None = None):
        Path(socket_path).unlink(missing_ok=True)
        super().__init__(socket_path, _Handler)
        self.verifier, self.log_path = verifier, log_path


class TcpVerifierServer(_Server, socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, host: str, port: int, verifier: Verifier, log_path: Path | None = None):
        super().__init__((host, port), _Handler)
        self.verifier, self.log_path = verifier, log_path


def main() -> None:
    p = argparse.ArgumentParser(description="Stand in for the upstream gateway and check every winning tile.")
    p.add_argument("--weights", nargs="+", type=Path, required=True, help="safetensors files with the declared int8 weights")
    p.add_argument("--socket", default="/tmp/pearlgw.sock", help="Unix socket path (upstream's default)")
    p.add_argument("--tcp", metavar="HOST:PORT", help="listen on TCP instead of a Unix socket")
    p.add_argument("--win-probability", type=float, default=1e-4, help="target chance that one tile hash wins")
    p.add_argument("--k", type=int, default=4096, help="inner dimension the difficulty is tuned for")
    p.add_argument("--interval", type=float, default=5.0, help="seconds between fresh random values")
    p.add_argument("--max-age", type=float, default=15.0, help="oldest job a proof may use, in seconds")
    p.add_argument("--log", type=Path, help="append one JSON line per submission here")
    args = p.parse_args()

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
    weights = DeclaredWeights.from_safetensors(args.weights)
    nbits = nbits_for_win_probability(args.win_probability, args.k)
    issuer = JobIssuer(nbits=nbits, interval_s=args.interval, max_age_s=args.max_age)
    verifier = Verifier(issuer, weights)
    print(f"declared matrices: {len(weights.tensors)}; nbits={nbits:#010x}", flush=True)

    if args.tcp:
        host, port = args.tcp.rsplit(":", 1)
        server: _Server = TcpVerifierServer(host, int(port), verifier, args.log)
    else:
        server = UnixVerifierServer(args.socket, verifier, args.log)
    print(f"listening on {args.tcp or args.socket}", flush=True)
    server.serve_forever()  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
