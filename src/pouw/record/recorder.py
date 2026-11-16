"""Record a verifier run, so its proofs can be checked again away from the GPU.

The recording is one JSON line per event: the verifier's settings, every job
it issues, and every proof it receives with its arrival time.
"""

from __future__ import annotations

import base64
import bisect
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ..jobs import KEEP_MAX_AGES, Job, JobIssuer
from ..verifier import Verdict, Verifier, VerifierService
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

    def started_at(self) -> float:
        """When the verifier issued its first job."""
        if not self.jobs:
            raise ValueError("the recording has no jobs")
        return min(job.issued_at for job in self.jobs)


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
    if not recording.settings:
        raise ValueError(f"{path} has no settings line, so its verifier's settings are unknown")
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


class ReplayIssuer(JobIssuer):
    """Knows the recorded jobs, and forgets them as the live issuer did, at the time set by at()."""

    def __init__(self, jobs: list[Job], nbits: int, interval_s: float, max_age_s: float):
        super().__init__(nbits, interval_s, max_age_s, forget_old=False)
        for job in jobs:
            self.remember(job)
        self._issue_times = sorted(job.issued_at for job in jobs)
        self._now = float("-inf")

    def at(self, now: float) -> None:
        self._now = now

    def lookup(self, header_bytes: bytes) -> Job | None:
        job = super().lookup(header_bytes)
        issued_by_now = bisect.bisect_right(self._issue_times, self._now)
        # The live issuer drops old jobs each time it issues a new one.
        if job is None or not issued_by_now:
            return job
        newest = self._issue_times[issued_by_now - 1]
        return job if job.issued_at >= newest - KEEP_MAX_AGES * self.max_age_s else None


def replay(recording: Recording, weights: DeclaredWeights) -> list[Verdict]:
    """Every recorded proof checked again, in arrival order, with the verdict the live verifier gave.

    The live verifier records each proof's arrival time and checks it at that
    time, so the replay reads the same clock. Two races can still differ: a
    job issued on another thread between a proof's arrival and its lookup,
    and the moment the live issuer drops old jobs, which here is the newest
    issue time at or before the arrival.
    """
    s = recording.settings
    issuer = ReplayIssuer(recording.jobs, s["nbits"], s["interval_s"], s["max_age_s"])
    verifier = Verifier(issuer, weights, attribute_gpus=s["attribute_gpus"])
    verdicts = []
    for arrival, submission in recording.submissions:
        issuer.at(arrival)
        verdicts.append(verifier.check(submission["plain_proof"], submission["mining_job"], now=arrival))
    return verdicts
