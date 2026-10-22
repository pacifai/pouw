"""Sit between the prover and the verifier, and cheat on a schedule.

The relay listens on the socket the upstream plugin connects to and passes
both JSON-RPC calls through to the verifier. Every honest proof goes through
unchanged. After every `every`-th honest proof, it also sends a cheated copy
of that proof, cycling through five cheats:

- replay: the same proof again;
- difficulty: the job's target made 256 times easier;
- forged: one activation in the proof's first opened row changed by one;
- theft: the proof submitted under another GPU ID's job (needs the verifier's
  --attribute-gpus);
- precompute: the proof held back until its job is one second past the
  maximum age.

During an optional time window, it also replaces the GPU ID the plugin sends
with a made-up one, so the verifier credits that window's proofs to an ID no
real GPU has.

Each changed submission is logged to `--log` with its hash
(`pouw.demo.engine.submission_hash`), the cheat's name, and what changed, so
the demo can label it in a recording of the run.

Example, on the pod:
    python -m pouw.cheat_relay --listen /tmp/pearlgw.sock --upstream /tmp/verifier.sock \\
      --log runs/cheats/cheats.jsonl --spoof-window 90 120
"""

from __future__ import annotations

import argparse
import base64
import json
import socketserver
import threading
import time
from pathlib import Path

from .cpu_prover import RpcClient
from .demo.engine import A_LEAF_OFFSET, submission_hash

CYCLE = ["replay", "difficulty", "forged", "theft", "precompute"]
NAMES = {
    "replay": "Replay",
    "difficulty": "Difficulty tampering",
    "forged": "Forged proof",
    "theft": "Proof theft",
    "precompute": "Precomputation",
    "spoof": "GPU ID spoofing",
}


class Relay:
    def __init__(
        self,
        upstream: str,
        log_path: Path,
        every: int = 8,
        start_after: int = 4,
        max_age_s: float = 15.0,
        spoof_window: tuple[float, float] | None = None,
        spoof_id: str = "GPU-11111111-2222-3333-4444-555555555555",
        theft_id: str = "GPU-B (relay)",
    ):
        self.upstream = upstream
        self.log_path = log_path
        self.every = every
        self.start_after = start_after
        self.max_age_s = max_age_s
        self.spoof_window = spoof_window
        self.spoof_id = spoof_id
        self.theft_id = theft_id
        self.first_seen: dict[str, float] = {}  # job header (base64) -> when the relay first saw it
        self.spoofed: dict[str, str | None] = {}  # job header -> the GPU ID the plugin really sent
        self.first_submission: float | None = None
        self.honest = 0
        self.cheats = 0
        self._lock = threading.Lock()

    def _log(self, cheat: str, params: dict, changes: list[dict], result=None) -> None:
        line = {
            "time": time.time(),
            "cheat": NAMES[cheat],
            "hash": submission_hash(params),
            "changes": changes,
            "result": result,
            # Replay and precomputation resend the honest submission unchanged,
            # so only its second copy is the cheat.
            "repeat": cheat in ("replay", "precompute"),
        }
        with self._lock, self.log_path.open("a") as f:
            f.write(json.dumps(line) + "\n")

    def _spoofing_now(self) -> bool:
        if self.spoof_window is None or self.first_submission is None:
            return False
        since = time.time() - self.first_submission
        return self.spoof_window[0] <= since <= self.spoof_window[1]

    def get_job(self, client: RpcClient, params: dict) -> dict:
        real_id = params.get("gpu_id")
        spoof = self._spoofing_now()
        if spoof:
            params = {**params, "gpu_id": self.spoof_id}
        job = client.call("getMiningInfo", params)
        header = job["incomplete_header_bytes"]
        with self._lock:
            self.first_seen.setdefault(header, time.time())
            if spoof:
                self.spoofed[header] = real_id
        return job

    def submit(self, client: RpcClient, params: dict) -> dict:
        header = params["mining_job"]["incomplete_header_bytes"]
        with self._lock:
            if self.first_submission is None:
                self.first_submission = time.time()
            spoofed = header in self.spoofed
            real_id = self.spoofed.get(header)
        if spoofed:
            self._log("spoof", params, [{"field": "GPU ID sent", "original": real_id or "none", "submitted": self.spoof_id}])
        with self._lock:
            self.honest += 1
            due = self.honest > self.start_after and self.honest % self.every == 0
            if due:
                cheat = CYCLE[self.cheats % len(CYCLE)]
                self.cheats += 1
        # Start the cheat now, not after the honest verdict: a slow verifier
        # would otherwise make every copy stale before it's sent.
        if due:
            threading.Thread(target=self.cheat, args=(cheat, params), daemon=True).start()
        return client.call("submitPlainProof", params)

    def cheat(self, cheat: str, params: dict) -> None:
        time.sleep(0.2)  # let the honest proof arrive first
        client = RpcClient(self.upstream)
        try:
            job = params["mining_job"]
            if cheat == "replay":
                sent, changes = params, [{"field": "submitted", "original": "once", "submitted": "again, unchanged"}]
            elif cheat == "difficulty":
                easier = {**job, "target": job["target"] * 256}
                sent = {**params, "mining_job": easier}
                changes = [{"field": "job target", "original": f"{job['target']:.3e}", "submitted": f"{easier['target']:.3e} (256 times easier)"}]
            elif cheat == "forged":
                raw = bytearray(base64.b64decode(params["plain_proof"]))
                i = A_LEAF_OFFSET + 100
                before = (raw[i] + 128) % 256 - 128
                raw[i] = (raw[i] + 1) % 256
                sent = {**params, "plain_proof": base64.b64encode(bytes(raw)).decode("ascii")}
                changes = [{"field": "activation in the first opened row of A, value 101", "original": str(before), "submitted": str((raw[i] + 128) % 256 - 128)}]
            elif cheat == "theft":
                other = client.call("getMiningInfo", {"gpu_id": self.theft_id})
                sent = {**params, "mining_job": other}
                changes = [{"field": "job (GPU ID)", "original": "the prover's job", "submitted": f"{self.theft_id}'s job"}]
            elif cheat == "precompute":
                seen = self.first_seen.get(job["incomplete_header_bytes"], time.time())
                hold_until = seen + self.max_age_s + 1.0
                time.sleep(max(0.0, hold_until - time.time()))
                sent = params
                changes = [{"field": "submitted", "original": "right after the win", "submitted": f"{self.max_age_s + 1:.0f} s after the job was issued"}]
            else:
                raise ValueError(cheat)
            try:
                result = client.call("submitPlainProof", sent)
            except RuntimeError as e:
                result = {"error": str(e)}
            self._log(cheat, sent, changes, result)
        finally:
            client.close()


def make_handler(relay: Relay):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            client = RpcClient(relay.upstream)
            try:
                for line in self.rfile:
                    if not line.strip():
                        continue
                    request_id = None
                    try:
                        request = json.loads(line)
                        request_id = request.get("id")
                        method, params = request["method"], request.get("params") or {}
                        if method == "getMiningInfo":
                            result = relay.get_job(client, params)
                        elif method == "submitPlainProof":
                            result = relay.submit(client, params)
                        else:
                            result = client.call(method, params)
                        response = {"jsonrpc": "2.0", "result": result, "id": request_id}
                    except Exception as e:  # pass failures back, as the verifier does
                        response = {"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}, "id": request_id}
                    self.wfile.write((json.dumps(response) + "\n").encode())
                    self.wfile.flush()
            finally:
                client.close()

    return Handler


class RelayServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def serve(relay: Relay, listen: str) -> RelayServer:
    Path(listen).unlink(missing_ok=True)
    server = RelayServer(listen, make_handler(relay))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    p = argparse.ArgumentParser(description="Relay prover traffic to the verifier, cheating on a schedule.")
    p.add_argument("--listen", default="/tmp/pearlgw.sock", help="socket the plugin connects to")
    p.add_argument("--upstream", default="/tmp/verifier.sock", help="the verifier's socket")
    p.add_argument("--log", type=Path, required=True, help="append one JSON line per changed submission here")
    p.add_argument("--every", type=int, default=8, help="send a cheat after every N-th honest proof")
    p.add_argument("--start-after", type=int, default=4, help="honest proofs to pass before the first cheat")
    p.add_argument("--max-age", type=float, default=15.0, help="the verifier's maximum job age, in seconds")
    p.add_argument("--spoof-window", type=float, nargs=2, metavar=("START", "END"), help="seconds after the first proof to fake the GPU ID")
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    relay = Relay(args.upstream, args.log, args.every, args.start_after, args.max_age, tuple(args.spoof_window) if args.spoof_window else None)
    serve(relay, args.listen)
    print(f"listening on {args.listen}, relaying to {args.upstream}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
