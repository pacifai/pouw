"""Serve the demo page and its JSON API.

Replay one or more recorded runs, from the repo root:
    docker run --rm -p 8080:8080 -v "$PWD":/work pouw-local \\
      python -m pouw.demo.server --run runs/h100-2026-10-11/demo \\
      --timing runs/h100-2026-10-11 --weights data/model/*.safetensors

Pass several --run directories to choose between them on the page.

Or run live: the server is also the verifier, listening on the socket the
upstream plugin connects to, and the page shows each proof as it arrives:
    python -m pouw.demo.server --live --weights /root/model/*.safetensors \\
      --win-probability 2e-8 --attribute-gpus --record runs/x/recording.jsonl

Then open http://localhost:8080 (on a pod, forward the port over SSH).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..jobs import JobIssuer, nbits_for_win_probability
from ..recording import Recorder
from ..verifier import TcpVerifierServer, UnixVerifierServer
from ..weights import DeclaredWeights
from .engine import ATTACKS, Engine, LiveEngine

STATIC = Path(__file__).parent / "static"
TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}


class Demo:
    """The engines the page can show, and which one it shows now."""

    def __init__(self, engines: list[Engine]):
        self.engines = engines
        self.current = 0

    @property
    def engine(self) -> Engine:
        return self.engines[self.current]

    def runs(self) -> list[dict]:
        return [
            {"index": i, "label": e.label, "description": e.description, "ready": e.ready, "mode": e.mode}
            for i, e in enumerate(self.engines)
        ]


def make_handler(demo: Demo):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def _send(self, status: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200) -> None:
            self._send(status, json.dumps(obj).encode())

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path == "/api/state":
                return self._json({**demo.engine.state(), "runs": demo.runs(), "current": demo.current})
            if path.startswith("/api/entry/"):
                entry = demo.engine.entry(int(path.rsplit("/", 1)[1]))
                return self._json(entry) if entry else self._json({"error": "no such entry"}, 404)
            if path == "/api/attacks":
                return self._json(ATTACKS)
            name = "index.html" if path == "/" else path.lstrip("/")
            f = (STATIC / name).resolve()
            if STATIC.resolve() in f.parents and f.is_file():
                return self._send(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))
            self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            path = self.path.split("?")[0]
            try:
                if path == "/api/start":
                    demo.engine.start()
                    return self._json({"ok": True})
                if path == "/api/reset":
                    demo.engine.reset()
                    return self._json({"ok": True})
                if path.startswith("/api/run/"):
                    index = int(path.rsplit("/", 1)[1])
                    if not 0 <= index < len(demo.engines):
                        raise ValueError("no such run")
                    demo.current = index
                    return self._json({"ok": True})
                if path.startswith("/api/attack/"):
                    entry_id = demo.engine.attack(path.rsplit("/", 1)[1])
                    return self._json({"ok": True, "entry": entry_id})
            except Exception as e:  # show the reason on the page
                return self._json({"ok": False, "error": str(e)}, 400)
            self._json({"error": "not found"}, 404)

    return Handler


def serve_live_verifier(engine: LiveEngine, socket_path: str | None, tcp: str | None):
    """Answer the prover's two JSON-RPC calls, passing each proof to the engine."""

    class Dispatch:
        def dispatch(self, method: str, params: dict) -> dict:
            if method == "getMiningInfo":
                return engine.verifier.issue(params).to_dict()
            if method == "submitPlainProof":
                verdict = engine.submit(params)
                return {"accepted": verdict.accepted, "reason": verdict.reason}
            raise ValueError(f"Method {method} not found")

    if tcp:
        host, port = tcp.rsplit(":", 1)
        cls = type("LiveTcpServer", (Dispatch, TcpVerifierServer), {})
        server = cls(host, int(port), engine.verifier)
    else:
        cls = type("LiveUnixServer", (Dispatch, UnixVerifierServer), {})
        server = cls(socket_path, engine.verifier)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    p = argparse.ArgumentParser(description="Serve the PoUW demo.")
    p.add_argument("--run", type=Path, nargs="+", help="run directories with recording.jsonl, to replay")
    p.add_argument("--live", action="store_true", help="be the verifier for a running prover and show its proofs live")
    p.add_argument("--weights", nargs="+", type=Path, required=True, help="safetensors files with the declared weights")
    p.add_argument("--timing", type=Path, help="directory holding plain-*, nosubmit-*, and noisy-* runs")
    p.add_argument("--speed", type=float, default=1.0, help="replay speed")
    p.add_argument("--port", type=int, default=8080)
    live = p.add_argument_group("live mode")
    live.add_argument("--socket", default="/tmp/pearlgw.sock", help="Unix socket for the prover (upstream's default)")
    live.add_argument("--tcp", metavar="HOST:PORT", help="listen for the prover on TCP instead")
    live.add_argument("--win-probability", type=float, default=1e-8)
    live.add_argument("--k", type=int, default=4096)
    live.add_argument("--interval", type=float, default=5.0)
    live.add_argument("--max-age", type=float, default=15.0)
    live.add_argument("--attribute-gpus", action="store_true")
    live.add_argument("--record", type=Path, help="record every job and proof here, for replay later")
    live.add_argument("--log", type=Path, help="append one JSON line per verdict here, like pouw.verifier --log")
    live.add_argument("--shapes", type=Path, help="the plugin's GEMM shape log to follow (patches/0003)")
    live.add_argument("--serving-since", type=Path, help="file that appears when vLLM starts serving")
    args = p.parse_args()
    if bool(args.run) == args.live:
        p.error("pass either --run or --live")

    print("loading weights...", flush=True)
    weights = DeclaredWeights.from_safetensors(args.weights)

    if args.live:
        nbits = nbits_for_win_probability(args.win_probability, args.k)
        issuer = JobIssuer(nbits=nbits, interval_s=args.interval, max_age_s=args.max_age)
        recorder = None
        if args.record:
            args.record.parent.mkdir(parents=True, exist_ok=True)
            recorder = Recorder(args.record)
            recorder._write(
                {
                    "type": "settings",
                    "nbits": nbits,
                    "interval_s": args.interval,
                    "max_age_s": args.max_age,
                    "attribute_gpus": args.attribute_gpus,
                    "win_probability": args.win_probability,
                    "k": args.k,
                }
            )
            issuer.on_new_job = recorder.job
        engine: Engine = LiveEngine(
            weights, issuer, args.attribute_gpus, args.shapes, args.serving_since, args.log, recorder
        )
        serve_live_verifier(engine, args.socket, args.tcp)
        print(f"declared matrices: {len(weights.tensors)}; nbits={nbits:#010x}", flush=True)
        print(f"listening on {args.tcp or args.socket}", flush=True)
        demo = Demo([engine])
    else:
        demo = Demo([Engine(run, weights, speed=args.speed, timing_dir=args.timing) for run in args.run])

        def prepare_all() -> None:
            for e in demo.engines:
                e.prepare()

        threading.Thread(target=prepare_all, daemon=True).start()

    def ticker() -> None:
        while True:
            for e in demo.engines:
                e.tick()
            time.sleep(0.1)

    threading.Thread(target=ticker, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(demo))
    print(f"demo on http://localhost:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
