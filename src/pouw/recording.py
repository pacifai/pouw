"""Record a verifier session so it can be checked again later, off the GPU.

With --record, the verifier appends one JSON line per event:

- {"type": "settings", ...}: written once at startup, with the target and
  the job interval and maximum age.
- {"type": "job", ...}: every job it issues, with its header bytes, target,
  issue time, and GPU ID.
- {"type": "submission", ...}: every proof the prover sends, exactly as it
  arrived, with the time it arrived.

That's everything a verifier needs to repeat each check. `replay_verifier`
builds a verifier that knows the recorded jobs, so you can pass each recorded
proof to `Verifier.check` with `now` set to its recorded arrival time and get
the same verdict the live verifier gave. The demo uses this to replay an H100
run on a laptop.
"""

from __future__ import annotations

import base64
import json
import threading
from dataclasses import dataclass
from pathlib import Path

from .jobs import Job, JobIssuer
from .verifier import Verifier
from .weights import DeclaredWeights


class Recorder:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def _write(self, event: dict) -> None:
        with self._lock, self.path.open("a") as f:
            f.write(json.dumps(event) + "\n")

    def job(self, job: Job) -> None:
        self._write(
            {
                "type": "job",
                "header_bytes": base64.b64encode(job.header_bytes).decode("ascii"),
                "nbits": job.nbits,
                "issued_at": job.issued_at,
                "cert_version": job.cert_version,
                "gpu_id": job.gpu_id,
            }
        )

    def submission(self, time: float, params: dict) -> None:
        self._write({"type": "submission", "time": time, "params": params})


@dataclass
class Recording:
    jobs: list[Job]
    # Each one is (arrival time, the submitPlainProof params as sent).
    submissions: list[tuple[float, dict]]
    issuer_settings: dict


def load(path: Path) -> Recording:
    jobs, submissions, settings = [], [], {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event["type"] == "job":
            jobs.append(
                Job(
                    header_bytes=base64.b64decode(event["header_bytes"]),
                    nbits=event["nbits"],
                    issued_at=event["issued_at"],
                    cert_version=event["cert_version"],
                    gpu_id=event["gpu_id"],
                )
            )
        elif event["type"] == "submission":
            submissions.append((event["time"], event["params"]))
        elif event["type"] == "settings":
            settings = event
    return Recording(jobs, submissions, settings)


def replay_verifier(recording: Recording, weights: DeclaredWeights) -> Verifier:
    """A verifier that knows every recorded job, with the recorded issuer settings."""
    s = recording.issuer_settings
    nbits = s.get("nbits", recording.jobs[0].nbits if recording.jobs else 0)
    issuer = JobIssuer(
        nbits=nbits,
        interval_s=s.get("interval_s", 5.0),
        max_age_s=s.get("max_age_s", 15.0),
        forget_old=False,
    )
    for job in recording.jobs:
        issuer._jobs[job.header_bytes] = job
    return Verifier(issuer, weights, attribute_gpus=s.get("attribute_gpus", False))
