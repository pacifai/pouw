"""Newline-delimited JSON-RPC, the protocol the upstream plugin speaks to its gateway."""

from __future__ import annotations

import json
import socket
from typing import BinaryIO, Callable


class RpcClient:
    """Calls methods over a Unix socket or TCP, one JSON request per line."""

    def __init__(self, socket_path: str | None = None, tcp: tuple[str, int] | None = None):
        if tcp:
            self._sock = socket.create_connection(tcp)
        else:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.connect(socket_path)
        self._r = self._sock.makefile("r", encoding="utf-8")
        self._w = self._sock.makefile("w", encoding="utf-8")
        self._id = 0

    def call(self, method: str, params: dict | None = None):
        self._id += 1
        self._w.write(json.dumps({"jsonrpc": "2.0", "method": method, "id": self._id, "params": params or {}}) + "\n")
        self._w.flush()
        response = json.loads(self._r.readline())
        if "error" in response:
            raise RuntimeError(response["error"]["message"])
        return response["result"]

    def close(self) -> None:
        self._r.close()
        self._w.close()
        self._sock.close()


def answer(rfile: BinaryIO, wfile: BinaryIO, dispatch: Callable[[str, dict], object]) -> None:
    """Answers each request line with dispatch(method, params), and any exception as an error."""
    for line in rfile:
        if not line.strip():
            continue
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            result = dispatch(request["method"], request.get("params") or {})
            response = {"jsonrpc": "2.0", "result": result, "id": request_id}
        except Exception as e:
            response = {"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}, "id": request_id}
        wfile.write((json.dumps(response) + "\n").encode())
        wfile.flush()
