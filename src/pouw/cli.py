"""The pouw command: python -m pouw <command>."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

from . import checkpoint, prover, summary
from .cheats import all_cheats
from .cheats.relay import Relay
from .jobs import INTERVAL_S, MAX_AGE_S, JobIssuer
from .protocol import DEFAULT_SOCKET, stop_server
from .record import changed_model as changed_copy
from .record import session as session_file
from .record.recorder import Recorder
from .target import bits_for_win_probability
from .verifier import Verifier, VerifierService
from .weights import DeclaredWeights

VERIFIER_SOCKET = "/tmp/verifier.sock"


def verify(args: argparse.Namespace) -> None:
    weights = DeclaredWeights.from_safetensors(args.weights)
    nbits = bits_for_win_probability(args.win_probability, args.k)
    issuer = JobIssuer(nbits, args.interval, args.max_age)
    service = VerifierService(Verifier(issuer, weights, args.attribute_gpus), _ready(args.log))
    if args.record:
        Recorder(_ready(args.record)).attach(service, args.win_probability)
    attribution = "on" if args.attribute_gpus else "off"
    print(f"declared matrices: {len(weights.tensors)}; nbits={nbits:#010x}; GPU attribution {attribution}", flush=True)
    _serve_until_interrupted(service.start(args.socket, args.tcp), args.tcp or args.socket)


def prove(args: argparse.Namespace) -> None:
    prover.run(args.weights, args.matrix, args.proofs, args.m, args.gpu_id, args.socket, args.tcp)


def relay(args: argparse.Namespace) -> None:
    cheats = all_cheats(max_age_s=args.max_age)
    server = Relay(args.verifier, _ready(args.log), cheats, args.every, args.start_after).start(args.listen)
    _serve_until_interrupted(server, f"{args.listen}, relaying to {args.verifier}")


def session(args: argparse.Namespace) -> None:
    for run in session_file.write(args.runs, args.out, args.gpu, args.weights)["runs"]:
        accepted = sum(p["accepted"] for p in run["proofs"])
        print(f"{run['name']}: {len(run['proofs'])} proofs, {accepted} accepted")


def summarize(args: argparse.Namespace) -> None:
    print(summary.summarize(args.log))


def fetch_tensors(args: argparse.Namespace) -> None:
    checkpoint.save(args.out, args.names, args.repo)


def changed_model(args: argparse.Namespace) -> None:
    print(f"changed one weight in each of {changed_copy.write(args.src, args.dst)} int8 matrices")


def _ready(path: Path | None) -> Path | None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _serve_until_interrupted(server, address: object) -> None:
    print(f"listening on {address}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        stop_server(server)


def _tcp(value: str) -> tuple[str, int]:
    host, port = value.rsplit(":", 1)
    return host, int(port)


def _add_address(p: argparse.ArgumentParser) -> None:
    p.add_argument("--socket", default=DEFAULT_SOCKET, help="Unix socket the plugin connects to")
    p.add_argument("--tcp", type=_tcp, metavar="HOST:PORT", help="use TCP instead of a Unix socket")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pouw", description="Proof of useful work on inference.")
    commands = p.add_subparsers(dest="command", required=True)

    c = commands.add_parser("verify", help="issue jobs and check every proof")
    c.set_defaults(run=verify)
    c.add_argument("--weights", nargs="+", type=Path, required=True, help="safetensors files of the declared model")
    _add_address(c)
    c.add_argument("--win-probability", type=float, default=1e-4, help="chance that one tile wins")
    c.add_argument("--k", type=int, default=4096, help="inner dimension the target is set for")
    c.add_argument("--interval", type=float, default=INTERVAL_S, help="seconds between new jobs")
    c.add_argument("--max-age", type=float, default=MAX_AGE_S, help="oldest job a proof may use, in seconds")
    c.add_argument("--log", type=Path, help="append one JSON line per verdict")
    c.add_argument("--record", type=Path, help="record every job and proof, for the session file")
    c.add_argument("--attribute-gpus", action="store_true", help="give each self-reported GPU ID its own jobs")

    c = commands.add_parser("prove", help="send proofs from the CPU prover")
    c.set_defaults(run=prove)
    c.add_argument("--weights", nargs="+", type=Path, required=True, help="safetensors files holding the matrix")
    c.add_argument("--matrix", required=True, help="tensor name, after fusing as vLLM does")
    _add_address(c)
    c.add_argument("--m", type=int, default=128, help="rows of activations")
    c.add_argument("--proofs", type=int, default=3, help="winning tiles to send")
    c.add_argument("--gpu-id", help="GPU ID to send with each job request")

    c = commands.add_parser("relay", help="pass calls to the verifier, and send cheated copies of real proofs")
    c.set_defaults(run=relay)
    c.add_argument("--listen", default=DEFAULT_SOCKET, help="socket the plugin connects to")
    c.add_argument("--verifier", default=VERIFIER_SOCKET, help="the verifier's socket")
    c.add_argument("--log", type=Path, required=True, help="append one JSON line per cheated submission")
    c.add_argument("--every", type=int, default=8, help="send a cheat after every N-th honest proof")
    c.add_argument("--start-after", type=int, default=4, help="honest proofs to pass before the first cheat")
    c.add_argument("--max-age", type=float, default=MAX_AGE_S, help="the verifier's maximum job age, in seconds")

    c = commands.add_parser("session", help="check recorded runs again and write the session file")
    c.set_defaults(run=session)
    c.add_argument("runs", nargs="+", type=Path, help="run directories from scripts/run_e2e.sh with RECORD=1")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--gpu", help="the GPU's name, as nvidia-smi reports it")
    c.add_argument("--weights", nargs="+", type=Path, help="declared weights, if not at the path in run.json")

    c = commands.add_parser("summarize", help="count a verifier log's verdicts and the work they imply")
    c.set_defaults(run=summarize)
    c.add_argument("log", type=Path)

    c = commands.add_parser("fetch-tensors", help="download single tensors from a Hugging Face checkpoint")
    c.set_defaults(run=fetch_tensors)
    c.add_argument("out", type=Path)
    c.add_argument("names", nargs="+")
    c.add_argument("--repo", default=checkpoint.MODEL_REPO)

    c = commands.add_parser("changed-model", help="copy a checkpoint with one weight changed in every int8 matrix")
    c.set_defaults(run=changed_model)
    c.add_argument("src", type=Path)
    c.add_argument("dst", type=Path)
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.run(args)
