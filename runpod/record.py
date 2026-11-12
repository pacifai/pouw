#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["runpod>=2.1,<3", "fabric>=3.2,<4", "python-dotenv>=1.0"]
# ///
"""Rents a RunPod H100, records a session on it, and releases it.

Copies this tree to the pod, builds upstream's vLLM plugin with this repo's
patches, records the honest, relay, and changed-model runs, and writes the
session file to results/sessions/ and the raw runs to runs/. Then it
terminates the pod. The pod also deletes itself after --max-hours.

Reads RUNPOD_API_KEY and POUW_SSH_KEY from the shell or the repo's .env.
Run with: uv run runpod/record.py
Exit: 0 done, 1 a prerequisite is missing, 2 renting failed, 3 the pod failed.
"""

import argparse
import io
import os
import secrets
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fabric import Connection
from runpod.cli.utils.rp_info import get_pod_ssh_ip_port

import runpod

ROOT = Path(__file__).resolve().parent.parent
POD_REPO = "/root/pouw"
POD_SESSION = "/root/session.json"
SESSIONS = ROOT / "results" / "sessions"
IMAGE = "nvidia/cuda:13.0.3-devel-ubuntu24.04"
SSH_WAIT_S = 900
SETUP_TIMEOUT_S = 7200
RECORD_TIMEOUT_S = 5400
# The CUDA image has no SSH server. The start command installs one, and first
# arms a timer that deletes the pod through RunPod's API after the hard cap.
START = (
    "bash -c '(sleep {cap_s}; curl -s -X DELETE https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID"
    ' -H "Authorization: Bearer $RUNPOD_API_KEY") >/dev/null 2>&1 &'
    " apt-get update -q >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -q"
    " openssh-server git curl build-essential >/dev/null;"
    ' mkdir -p /run/sshd /root/.ssh && echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys'
    " && chmod 600 /root/.ssh/authorized_keys && exec /usr/sbin/sshd -D'"
)


def say(title):
    print(f"\n== {title} ==", flush=True)


def fail(message, code):
    print(f"\nFAILED: {message}", file=sys.stderr, flush=True)
    sys.exit(code)


def connect(host, port, key):
    return Connection(
        host,
        user="root",
        port=port,
        connect_timeout=15,
        connect_kwargs={"key_filename": key, "look_for_keys": False, "allow_agent": False},
    )


def confirm_price(args):
    say("price")
    try:
        gpu = runpod.get_gpu(args.gpu)
    except ValueError:
        fail(f"no GPU with id {args.gpu!r}; list them with runpod.get_gpus()", 1)
    price = (gpu.get("price") or {}).get(args.cloud.lower())
    if price is None:
        fail(f"no {args.cloud} price for {args.gpu}; not renting without one", 2)
    print(f"{args.gpu}, {args.cloud} cloud: ${price}/hr, stock {gpu.get('availability', '?')}")
    print(f"A session takes about 1.5 hours. The pod deletes itself after {args.max_hours} hours at most.")
    if (
        not args.yes
        and input("\nRent it? It bills by the second until the recording ends. [y/N] ").strip().lower() != "y"
    ):
        print("Nothing rented.")
        sys.exit(0)


def rent(args):
    """Creates the pod and returns its id. A pod created despite an error is found by name and terminated."""
    say("renting")
    name = f"pouw-record-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{secrets.token_hex(3)}"
    try:
        return runpod.create_pod(
            name,
            image_name=IMAGE,
            gpu_type_id=args.gpu,
            cloud_type=args.cloud,
            container_disk_in_gb=args.disk,
            ports="22/tcp",
            allowed_cuda_versions=["13.0"],
            docker_args=START.format(cap_s=int(args.max_hours * 3600)),
        )["id"]
    except Exception as err:
        print(f"create_pod failed: {err}")
    for pod in runpod.get_pods():
        if pod.get("name") == name:
            runpod.terminate_pod(pod["id"])
            fail(f"pod {pod['id']} was created despite the error and has been terminated", 2)
    fail(f"could not rent a {args.gpu}. Try again later, or --cloud COMMUNITY.", 2)


def wait_for_ssh(pod_id, key):
    """Returns a connection once the pod answers over ssh, or None after SSH_WAIT_S."""
    say("waiting for ssh")
    deadline = time.monotonic() + SSH_WAIT_S
    try:
        host, port = get_pod_ssh_ip_port(pod_id, timeout=SSH_WAIT_S)
    except TimeoutError as err:
        print(err)
        return None
    conn = connect(host, port, key)
    while time.monotonic() < deadline:
        try:
            conn.run("true", hide=True)
            print(f"ssh -i {key} -p {port} root@{host}")
            return conn
        except Exception:
            time.sleep(10)
    return None


def ship(conn):
    """Copies every file git would track, committed or not, to POD_REPO."""
    say("copying this working tree")
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode()
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for name in filter(None, listed.split("\0")):
            if (ROOT / name).is_file():
                tar.add(ROOT / name, arcname=name)
    archive.seek(0)
    conn.put(archive, "/root/pouw.tar.gz")
    conn.run(f"mkdir -p {POD_REPO} && tar xzf /root/pouw.tar.gz -C {POD_REPO}", hide=True)


def fetch(conn, stamp):
    """Downloads the session file and the raw runs. Returns the session file's path."""
    SESSIONS.mkdir(parents=True, exist_ok=True)
    session = SESSIONS / f"{stamp}.json"
    conn.get(POD_SESSION, str(session))
    runs = ROOT / "runs" / stamp
    runs.mkdir(parents=True, exist_ok=True)
    conn.run("tar czf /root/runs.tar.gz -C /root runs runs-*.log", hide=True)
    conn.get("/root/runs.tar.gz", str(runs / "runs.tar.gz"))
    with tarfile.open(runs / "runs.tar.gz") as tar:
        tar.extractall(runs, filter="data")
    (runs / "runs.tar.gz").unlink()
    return session


def record(args):
    confirm_price(args)
    pod_id = rent(args)
    print(f"pod {pod_id}")
    stamp = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    try:
        conn = wait_for_ssh(pod_id, args.key)
        if conn is None:
            fail(f"pod {pod_id} never answered over ssh", 2)
        ship(conn)
        say("building upstream and downloading the model")
        if conn.run(f"bash {POD_REPO}/scripts/setup_pod.sh", warn=True, timeout=SETUP_TIMEOUT_S).exited != 0:
            fail("setup_pod.sh failed (see above)", 3)
        say("recording")
        shapes = " --shapes" if args.shapes else ""
        recorded = conn.run(
            f"bash {POD_REPO}/scripts/record_session.sh {POD_SESSION}{shapes}", warn=True, timeout=RECORD_TIMEOUT_S
        )
        session = fetch(conn, stamp)
    finally:
        terminate(pod_id)
    say("recorded")
    print(f"  {session.relative_to(ROOT)}")
    if recorded.exited != 0:
        fail("a run failed and is missing from the session (see above)", 3)


def terminate(pod_id):
    say("terminating")
    runpod.terminate_pod(pod_id)
    for _ in range(10):
        pod = runpod.get_pod(pod_id)
        if pod is None or pod.get("status") == "TERMINATED":
            print(f"pod {pod_id} is terminated and no longer billing")
            return
        time.sleep(3)
    fail(f"{pod_id} still reports {pod.get('status')}; check the RunPod console", 3)


def main():
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", default="NVIDIA H100 80GB HBM3", help="RunPod GPU type id; the kernels need a Hopper GPU")
    p.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    p.add_argument("--disk", type=int, default=150, help="container disk in GB")
    p.add_argument(
        "--key",
        default=os.environ.get("POUW_SSH_KEY"),
        help="SSH private key registered on the account (default: POUW_SSH_KEY)",
    )
    p.add_argument("--max-hours", type=float, default=3, help="the pod deletes itself this long after it starts")
    p.add_argument("--shapes", action="store_true", help="log every GEMM's shape, and add each run's proven share")
    p.add_argument("--yes", "-y", action="store_true", help="rent without asking")
    args = p.parse_args()

    if not os.environ.get("RUNPOD_API_KEY"):
        fail("RUNPOD_API_KEY is not set. Export it or put it in .env (see .env.example).", 1)
    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    args.key = os.path.expanduser(args.key or "")
    if not os.path.isfile(args.key):
        fail("POUW_SSH_KEY or --key must name an SSH private key registered on your RunPod account.", 1)
    if args.max_hours <= 0:
        fail("--max-hours takes a positive number", 1)
    record(args)


if __name__ == "__main__":
    main()
