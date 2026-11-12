"""Record a verifier run, so its proofs can be checked again away from the GPU.

The recording is one JSON line per event: the verifier's settings, every job
it issues, and every proof it receives with its arrival time.
"""

from __future__ import annotations

import base64
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ..jobs import INTERVAL_S, MAX_AGE_S, Job, JobIssuer
from ..verifier import Verifier, VerifierService
from ..weights import DeclaredWeights


class Recorder:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def attach(self, service: VerifierService, win_probability: float) -> None:
        """Records the service's settings now, and its jobs and submissions from now on."""
        issuer = service.verifier.issuer
        self.settings(issuer, service.verifier.attribute_gpus, win_probability)
        issuer.on_new_job = self.job
        service.on_submission = self.submission

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

    def _write(self, event: dict) -> None:
        with self._lock, self.path.open("a") as f:
            f.write(json.dumps(event) + "\n")


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
            recording.jobs.append(_job(event))
        elif event["type"] == "submission":
            recording.submissions.append((event["time"], event["params"]))
        elif event["type"] == "settings":
            recording.settings = event
    recording.submissions.sort(key=lambda s: s[0])
    return recording


def _job(event: dict) -> Job:
    return Job(
        header_bytes=base64.b64decode(event["header_bytes"]),
        nbits=event["nbits"],
        issued_at=event["issued_at"],
        cert_version=event["cert_version"],
        gpu_id=event["gpu_id"],
    )


def replay_verifier(recording: Recording, weights: DeclaredWeights) -> Verifier:
    """A verifier with the recorded settings that knows every recorded job.

    Checking each recorded proof with now set to its arrival time gives the
    verdict the live verifier gave.
    """
    s = recording.settings
    issuer = JobIssuer(
        nbits=s.get("nbits", recording.jobs[0].nbits if recording.jobs else 0),
        interval_s=s.get("interval_s", INTERVAL_S),
        max_age_s=s.get("max_age_s", MAX_AGE_S),
        forget_old=False,
    )
    for job in recording.jobs:
        issuer.remember(job)
    return Verifier(issuer, weights, attribute_gpus=s.get("attribute_gpus", False))
