"""Sit between the prover and the verifier, and send cheated copies of real proofs.

The relay listens on the socket the upstream plugin connects to and passes
both JSON-RPC calls through to the verifier. Every honest proof goes through
unchanged. After every `every`-th honest proof, it also sends a cheated copy
of that proof, cycling through CHEATS.

Each cheated submission is logged with its submission_hash, the cheat's name,
and what changed, so a session file can mark it.

Example, on the pod:
    python -m pouw.cheat_relay --listen /tmp/pearlgw.sock --upstream /tmp/verifier.sock --log runs/relay/cheats.jsonl
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import socketserver
import threading
import time
from pathlib import Path

from .rpc import RpcClient, answer

CHEATS = {
    "replay": "Sends the same proof again",
    "difficulty": "Makes the job's target 256 times easier",
    "forged": "Changes one activation in the proof's first opened row by one",
    "theft": "Submits the proof under a job issued to another GPU ID",
    "precompute": "Holds the proof until its job is one second past the maximum age",
}
# Replay and precomputation resend the honest submission unchanged, so only the second copy is the cheat.
RESENT_UNCHANGED = {"replay", "precompute"}
# Byte offset of the first opened row of A in a serialized proof: four 8-byte
# sizes (m, n, k, noise rank), then A's leaf count and leaf length, 8 bytes each.
A_LEAF_OFFSET = 48


def submission_hash(params: dict) -> str:
    """Identifies a submission by its proof and job."""
    job = json.dumps(params["mining_job"], sort_keys=True)
    return hashlib.sha256((params["plain_proof"] + "|" + job).encode()).hexdigest()


class Relay:
    def __init__(
        self,
        upstream: str,
        log_path: Path,
        every: int = 8,
        start_after: int = 4,
        max_age_s: float = 15.0,
        theft_id: str = "GPU-B (relay)",
    ):
        self.upstream = upstream
        self.log_path = log_path
        self.every = every
        self.start_after = start_after
        self.max_age_s = max_age_s
        self.theft_id = theft_id
        self.first_seen: dict[str, float] = {}
        self.honest = 0
        self.cheats = 0
        self._lock = threading.Lock()

    def get_job(self, client: RpcClient, params: dict) -> dict:
        job = client.call("getMiningInfo", params)
        with self._lock:
            self.first_seen.setdefault(job["incomplete_header_bytes"], time.time())
        return job

    def submit(self, client: RpcClient, params: dict) -> dict:
        with self._lock:
            self.honest += 1
            cheat = None
            if self.honest > self.start_after and self.honest % self.every == 0:
                cheat = list(CHEATS)[self.cheats % len(CHEATS)]
                self.cheats += 1
        if cheat:
            threading.Thread(target=self.cheat, args=(cheat, params), daemon=True).start()
        return client.call("submitPlainProof", params)

    def cheat(self, cheat: str, params: dict) -> None:
        time.sleep(0.2)
        client = RpcClient(self.upstream)
        try:
            sent, changes = self._change(cheat, params, client)
            try:
                result = client.call("submitPlainProof", sent)
            except RuntimeError as e:
                result = {"error": str(e)}
            self._log(cheat, sent, changes, result)
        finally:
            client.close()

    def _change(self, cheat: str, params: dict, client: RpcClient) -> tuple[dict, list[dict]]:
        job = params["mining_job"]
        if cheat == "replay":
            return params, [{"field": "submitted", "original": "once", "submitted": "again, unchanged"}]
        if cheat == "difficulty":
            easier = {**job, "target": job["target"] * 256}
            change = {
                "field": "job target",
                "original": f"{job['target']:.3e}",
                "submitted": f"{easier['target']:.3e} (256 times easier)",
            }
            return {**params, "mining_job": easier}, [change]
        if cheat == "forged":
            raw = bytearray(base64.b64decode(params["plain_proof"]))
            i = A_LEAF_OFFSET + 100
            before = (raw[i] + 128) % 256 - 128
            raw[i] = (raw[i] + 1) % 256
            change = {
                "field": "activation in the first opened row of A",
                "original": str(before),
                "submitted": str((raw[i] + 128) % 256 - 128),
            }
            return {**params, "plain_proof": base64.b64encode(bytes(raw)).decode("ascii")}, [change]
        if cheat == "theft":
            other = client.call("getMiningInfo", {"gpu_id": self.theft_id})
            return {**params, "mining_job": other}, [
                {"field": "job", "original": "the prover's job", "submitted": f"{self.theft_id}'s job"}
            ]
        if cheat == "precompute":
            seen = self.first_seen.get(job["incomplete_header_bytes"], time.time())
            time.sleep(max(0.0, seen + self.max_age_s + 1.0 - time.time()))
            return params, [
                {
                    "field": "submitted",
                    "original": "right after the win",
                    "submitted": f"{self.max_age_s + 1:.0f} s after the job was issued",
                }
            ]
        raise ValueError(cheat)

    def _log(self, cheat: str, params: dict, changes: list[dict], result: dict) -> None:
        line = {
            "time": time.time(),
            "cheat": cheat,
            "hash": submission_hash(params),
            "changes": changes,
            "result": result,
            "repeat": cheat in RESENT_UNCHANGED,
        }
        with self._lock, self.log_path.open("a") as f:
            f.write(json.dumps(line) + "\n")


def make_handler(relay: Relay) -> type[socketserver.StreamRequestHandler]:
    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            client = RpcClient(relay.upstream)

            def dispatch(method: str, params: dict):
                if method == "getMiningInfo":
                    return relay.get_job(client, params)
                if method == "submitPlainProof":
                    return relay.submit(client, params)
                return client.call(method, params)

            try:
                answer(self.rfile, self.wfile, dispatch)
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
    p = argparse.ArgumentParser(
        description="Relay prover traffic to the verifier, and send cheated copies of real proofs."
    )
    p.add_argument("--listen", default="/tmp/pearlgw.sock", help="socket the plugin connects to")
    p.add_argument("--upstream", default="/tmp/verifier.sock", help="the verifier's socket")
    p.add_argument("--log", type=Path, required=True, help="append one JSON line per cheated submission here")
    p.add_argument("--every", type=int, default=8, help="send a cheat after every N-th honest proof")
    p.add_argument("--start-after", type=int, default=4, help="honest proofs to pass before the first cheat")
    p.add_argument("--max-age", type=float, default=15.0, help="the verifier's maximum job age, in seconds")
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    serve(Relay(args.upstream, args.log, args.every, args.start_after, args.max_age), args.listen)
    print(f"listening on {args.listen}, relaying to {args.upstream}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
