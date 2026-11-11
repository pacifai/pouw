"""Record a verifier session, so its proofs can be checked again later, away from the GPU.

With --record, the verifier appends one JSON line per event:

- {"type": "settings", ...}: once, at startup: the target, the job interval, and the maximum age.
- {"type": "job", ...}: every job it issues.
- {"type": "submission", ...}: every proof the prover sends, as it arrived, with its arrival time.

Verifier.from_recording rebuilds a verifier that knows the recorded jobs.
Checking each recorded proof with `now` set to its arrival time gives the
verdict the live verifier gave.
"""

from __future__ import annotations

import base64
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .jobs import Job, JobIssuer


class Recorder:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def _write(self, event: dict) -> None:
        with self._lock, self.path.open("a") as f:
            f.write(json.dumps(event) + "\n")

    def settings(self, issuer: JobIssuer, attribute_gpus: bool, win_probability: float) -> None:
        self._write(
            {
                "type": "settings",
                "nbits": issuer.nbits,
                "interval_s": issuer.interval_s,
                "max_age_s": issuer.max_age_s,
                "attribute_gpus": attribute_gpus,
                "win_probability": win_probability,
            }
        )

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
    jobs: list[Job] = field(default_factory=list)
    # (arrival time, the submitPlainProof params as sent), in arrival order.
    submissions: list[tuple[float, dict]] = field(default_factory=list)
    settings: dict = field(default_factory=dict)


def load(path: Path) -> Recording:
    recording = Recording()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event["type"] == "job":
            recording.jobs.append(
                Job(
                    header_bytes=base64.b64decode(event["header_bytes"]),
                    nbits=event["nbits"],
                    issued_at=event["issued_at"],
                    cert_version=event["cert_version"],
                    gpu_id=event["gpu_id"],
                )
            )
        elif event["type"] == "submission":
            recording.submissions.append((event["time"], event["params"]))
        elif event["type"] == "settings":
            recording.settings = event
    recording.submissions.sort(key=lambda s: s[0])
    return recording
