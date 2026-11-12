"""Jobs: block headers that carry the verifier's random value, and the issuer that hands them out.

The random value fills the header's previous-block-hash field, so every noise
seed and tile hash depends on it. With GPU attribution on, the Merkle-root
field holds a hash of the GPU's ID.
"""

from __future__ import annotations

import base64
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable

from blake3 import blake3
from pearl_mining import IncompleteBlockHeader

from .proof import CERT_VERSION
from .target import bits_to_target

INTERVAL_S = 5.0
MAX_AGE_S = 15.0
# Jobs older than this many maximum ages can't pass the on-time check, so the issuer drops them.
KEEP_MAX_AGES = 4

GPU_ID_DOMAIN = b"pouw gpu id"
MAX_GPU_ID_LEN = 128


@dataclass(frozen=True)
class Job:
    header_bytes: bytes
    nbits: int
    issued_at: float
    cert_version: int = CERT_VERSION
    gpu_id: str | None = None

    @property
    def target(self) -> int:
        return bits_to_target(self.nbits)

    @property
    def job_id(self) -> str:
        return bytes(IncompleteBlockHeader.from_bytes(self.header_bytes).prev_block).hex()[:12]

    def to_dict(self) -> dict:
        """The getMiningInfo result, as upstream's MiningJob.from_dict reads it."""
        return {
            "incomplete_header_bytes": base64.b64encode(self.header_bytes).decode("ascii"),
            "target": self.target,
            "cert_version": self.cert_version,
        }


def gpu_id_hash(gpu_id: str | None) -> bytes:
    if gpu_id is None:
        return bytes(32)
    return blake3(GPU_ID_DOMAIN + gpu_id.encode("utf-8")).digest()


def new_job(nbits: int, now: float | None = None, gpu_id: str | None = None) -> Job:
    now = time.time() if now is None else now
    header = IncompleteBlockHeader(
        version=1,
        prev_block=os.urandom(32),
        merkle_root=gpu_id_hash(gpu_id),
        timestamp=int(now),
        nbits=nbits,
    )
    return Job(header_bytes=bytes(header.to_bytes()), nbits=nbits, issued_at=now, gpu_id=gpu_id)


class JobIssuer:
    """Hands out a new job every interval_s for each GPU ID, and looks up the jobs it issued.

    The ID None stands for the shared job, used when attribution is off or a prover sends no ID.
    """

    def __init__(
        self,
        nbits: int,
        interval_s: float = INTERVAL_S,
        max_age_s: float = MAX_AGE_S,
        on_new_job: Callable[[Job], None] | None = None,
        forget_old: bool = True,
    ):
        self.nbits = nbits
        self.interval_s = interval_s
        self.max_age_s = max_age_s
        self.on_new_job = on_new_job
        self.forget_old = forget_old
        self._issued: dict[bytes, Job] = {}
        self._current: dict[str | None, Job] = {}
        self._lock = threading.Lock()

    def current(self, now: float | None = None, gpu_id: str | None = None) -> Job:
        now = time.time() if now is None else now
        with self._lock:
            job = self._current.get(gpu_id)
            if job is None or now - job.issued_at >= self.interval_s:
                job = new_job(self.nbits, now, gpu_id)
                self._current[gpu_id] = job
                self._issued[job.header_bytes] = job
                if self.forget_old:
                    self._forget_before(now - KEEP_MAX_AGES * self.max_age_s)
                if self.on_new_job is not None:
                    self.on_new_job(job)
            return job

    def remember(self, job: Job) -> None:
        with self._lock:
            self._issued[job.header_bytes] = job

    def lookup(self, header_bytes: bytes) -> Job | None:
        with self._lock:
            return self._issued.get(header_bytes)

    def _forget_before(self, cutoff: float) -> None:
        self._issued = {h: j for h, j in self._issued.items() if j.issued_at >= cutoff}
        self._current = {g: j for g, j in self._current.items() if j.issued_at >= cutoff}
