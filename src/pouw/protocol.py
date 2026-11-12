"""Newline-delimited JSON-RPC, as upstream's plugin speaks it to its gateway."""

from __future__ import annotations

import json
import socket
import socketserver
import threading
from pathlib import Path
from typing import BinaryIO, Callable

DEFAULT_SOCKET = "/tmp/pearlgw.sock"

Dispatch = Callable[[str, dict], object]
Handler = Callable[[BinaryIO, BinaryIO], None]


class RpcClient:
    def __init__(self, socket_path: str | None = None, tcp: tuple[str, int] | None = None):
        if tcp:
            self._sock = socket.create_connection(tcp)
        else:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.connect(socket_path or DEFAULT_SOCKET)
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


def answer(rfile: BinaryIO, wfile: BinaryIO, dispatch: Dispatch) -> None:
    """Answers each request line with dispatch(method, params), and an exception as an error."""
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


class _UnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


class _TcpServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_server(
    handle: Handler, socket_path: str | None = None, tcp: tuple[str, int] | None = None
) -> socketserver.BaseServer:
    """Serves each connection with handle(rfile, wfile) on its own thread, in the background."""

    class _Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            handle(self.rfile, self.wfile)

    if tcp:
        server: socketserver.BaseServer = _TcpServer(tcp, _Handler)
    else:
        path = socket_path or DEFAULT_SOCKET
        Path(path).unlink(missing_ok=True)
        server = _UnixServer(path, _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def stop_server(server: socketserver.BaseServer) -> None:
    server.shutdown()
    server.server_close()
