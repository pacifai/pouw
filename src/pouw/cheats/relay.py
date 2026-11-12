"""The relay: sits on the plugin's socket, passes every call through to the verifier, and sends cheats.

After every `every`-th honest proof, it sends a cheated copy of that proof,
cycling through its cheats, and logs one JSON line per cheated submission so a
session file can mark it.
"""

from __future__ import annotations

import hashlib
import json
import socketserver
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import BinaryIO

from ..protocol import RpcClient, answer, start_server
from . import Cheat, Context

# The cheated copy follows the honest proof, so the verifier sees the honest one first.
CHEAT_DELAY_S = 0.2


def submission_hash(submission: dict) -> str:
    job = json.dumps(submission["mining_job"], sort_keys=True)
    return hashlib.sha256((submission["plain_proof"] + "|" + job).encode()).hexdigest()


class Relay:
    def __init__(self, verifier_socket: str, log_path: Path, cheats: list[Cheat], every: int = 8, start_after: int = 4):
        self.verifier_socket = verifier_socket
        self.log_path = log_path
        self.cheats = cheats
        self.every = every
        self.start_after = start_after
        self._job_seen_at: dict[str, float] = {}
        self._honest = 0
        self._sent = 0
        self._lock = threading.Lock()

    def start(self, socket_path: str) -> socketserver.BaseServer:
        return start_server(self.handle, socket_path)

    def handle(self, rfile: BinaryIO, wfile: BinaryIO) -> None:
        verifier = RpcClient(self.verifier_socket)

        def dispatch(method: str, params: dict):
            if method == "getMiningInfo":
                return self._pass_job(verifier, params)
            if method == "submitPlainProof":
                return self._pass_proof(verifier, params)
            return verifier.call(method, params)

        try:
            answer(rfile, wfile, dispatch)
        finally:
            verifier.close()

    def _pass_job(self, verifier: RpcClient, params: dict) -> dict:
        job = verifier.call("getMiningInfo", params)
        with self._lock:
            self._job_seen_at.setdefault(job["incomplete_header_bytes"], time.time())
        return job

    def _pass_proof(self, verifier: RpcClient, submission: dict) -> dict:
        cheat = self._next_cheat()
        if cheat:
            threading.Thread(target=self._send, args=(cheat, submission), daemon=True).start()
        return verifier.call("submitPlainProof", submission)

    def _next_cheat(self) -> Cheat | None:
        with self._lock:
            self._honest += 1
            if self._honest <= self.start_after or self._honest % self.every:
                return None
            cheat = self.cheats[self._sent % len(self.cheats)]
            self._sent += 1
            return cheat

    def _send(self, cheat: Cheat, honest: dict) -> None:
        time.sleep(CHEAT_DELAY_S)
        verifier = RpcClient(self.verifier_socket)
        try:
            seen_at = self._job_seen_at.get(honest["mining_job"]["incomplete_header_bytes"], time.time())
            cheated, changes = cheat.apply(honest, Context(verifier, seen_at))
            try:
                result = verifier.call("submitPlainProof", cheated)
            except RuntimeError as e:
                result = {"error": str(e)}
            self._log(cheat, cheated, changes, result)
        finally:
            verifier.close()

    def _log(self, cheat: Cheat, submission: dict, changes: list, result: dict) -> None:
        line = {
            "time": time.time(),
            "cheat": cheat.key,
            "hash": submission_hash(submission),
            "changes": [asdict(c) for c in changes],
            "result": result,
            "repeat": cheat.resends_unchanged,
        }
        with self._lock, self.log_path.open("a") as f:
            f.write(json.dumps(line) + "\n")
