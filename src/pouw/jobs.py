"""Jobs: the verifier's random value, packed into a block header.

The prover expects a block header and a difficulty target. The random value
goes in the header's previous-block-hash field, so every noise seed and tile
hash the GPU produces depends on it.

With GPU attribution on, each GPU gets its own job: the header's Merkle-root
field holds a hash of the GPU's ID instead of zeros. The noise seeds and tile
hashes then differ per GPU, so a proof made under one GPU's job can't pass as
another's. The other header fields are fixed filler.

The GPU ID is whatever the prover reports, such as the UUID the driver gives.
Nothing signs it, so a dishonest host can claim any ID, or several IDs for
one GPU. Attribution therefore says which ID a proof was made for, not which
physical GPU made it. To bind an ID to real hardware, a later version can
replace the reported ID with a signed attestation; see NEXT_STEPS.md.
"""

from __future__ import annotations

import base64
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from blake3 import blake3
from pearl_mining import (
    IncompleteBlockHeader,
    MiningConfiguration,
    MMAType,
    PeriodicPattern,
    penalized_target_bound,
)

# The proof format's certificate versions. 3 is the current one: noise seeds are salted
# with the matrix dimensions. Dense (non-MoE) proofs are valid under all three.
CERT_VERSION = 3

# The tile pattern and noise rank the GPU prover uses (upstream's MinerSettings).
NOISE_RANK = 128
ROWS_PATTERN = [0, 8]
COLS_PATTERN = [c + d for c in range(0, 256, 8) for d in (0, 1)]

# Prefix for hashing a GPU ID into a header, so the hash can't collide with
# any other use of BLAKE3 in the proof format.
GPU_ID_DOMAIN = b"pouw gpu id"
# Longest GPU ID the verifier accepts. A UUID string is 40 characters.
MAX_GPU_ID_LEN = 128


def bits_to_target(bits: int) -> int:
    """Expand Bitcoin's compact difficulty encoding into a 256-bit target."""
    exponent = (bits >> 24) & 0xFF
    mantissa = bits & 0xFFFFFF
    return mantissa * (1 << (8 * (exponent - 3)))


def target_to_bits(target: int) -> int:
    """Compress a target into the compact encoding, rounding down."""
    size = (target.bit_length() + 7) // 8
    if size <= 3:
        mantissa = target << (8 * (3 - size))
    else:
        mantissa = target >> (8 * (size - 3))
    # The top mantissa bit is a sign bit in this encoding, so keep it clear.
    if mantissa & 0x800000:
        mantissa >>= 8
        size += 1
    return (size << 24) | mantissa


def gpu_id_hash(gpu_id: str | None) -> bytes:
    """The 32 bytes for the header's Merkle-root field: zeros, or a hash of the GPU ID."""
    if gpu_id is None:
        return bytes(32)
    return blake3(GPU_ID_DOMAIN + gpu_id.encode("utf-8")).digest()


def gpu_proof_config(k: int) -> MiningConfiguration:
    """The proof configuration the GPU prover uses for a GEMM with inner dimension k."""
    return MiningConfiguration(
        common_dim=k,
        rank=NOISE_RANK,
        mma_type=MMAType.Int7xInt7ToInt32,
        rows_pattern=PeriodicPattern.from_list(ROWS_PATTERN),
        cols_pattern=PeriodicPattern.from_list(COLS_PATTERN),
        moe=None,
    )


def win_probability(target: int, config: MiningConfiguration) -> float:
    """Chance that one tile hash falls below the prover's target for this config.

    The prover compares each tile hash against the library's rank-adjusted bound, not
    the raw target, so this is the number to divide wins by to estimate tiles.
    """
    bound = penalized_target_bound(target, config)
    if bound is None:
        raise ValueError("target is unusable for this configuration")
    return bound / 2**256


def nbits_for_win_probability(p: float, k: int) -> int:
    """Pick the compact difficulty that gives each tile about a p chance to win."""
    factor = penalized_target_bound(1, gpu_proof_config(k))
    if factor is None:
        raise ValueError(f"no usable target for k={k}")
    return target_to_bits(int(p * 2**256) // factor)


@dataclass(frozen=True)
class Job:
    header_bytes: bytes
    nbits: int
    issued_at: float
    cert_version: int = CERT_VERSION
    # The GPU this job was issued for, as the prover reported it. None for the
    # shared job. Self-reported and unsigned: see the module docstring.
    gpu_id: str | None = None

    @property
    def target(self) -> int:
        return bits_to_target(self.nbits)

    @property
    def job_id(self) -> str:
        """Short label for logs: the first bytes of the random value."""
        return bytes(IncompleteBlockHeader.from_bytes(self.header_bytes).prev_block).hex()[:12]

    def to_dict(self) -> dict:
        """The getMiningInfo result, in the shape upstream's MiningJob.from_dict reads."""
        return {
            "incomplete_header_bytes": base64.b64encode(self.header_bytes).decode("ascii"),
            "target": self.target,
            "cert_version": self.cert_version,
        }


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


@dataclass
class JobIssuer:
    """Hands out a fresh job every `interval_s` and remembers recent ones.

    A proof counts only if its job was issued here and is younger than
    `max_age_s`. That bounds when the work behind the proof could have happened.

    Each GPU ID has its own current job, refreshed on the same interval. The
    ID None is the shared job, used when attribution is off or a prover sends
    no ID.
    """

    nbits: int
    interval_s: float = 5.0
    max_age_s: float = 15.0
    # Called with each new job, for example to record it.
    on_new_job: Callable[[Job], None] | None = None
    # Drop jobs older than 4 * max_age_s. A replay of a recorded session keeps them all.
    forget_old: bool = True
    _jobs: dict[bytes, Job] = field(default_factory=dict)
    _current: dict[str | None, Job] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def current(self, now: float | None = None, gpu_id: str | None = None) -> Job:
        now = time.time() if now is None else now
        with self._lock:
            job = self._current.get(gpu_id)
            if job is None or now - job.issued_at >= self.interval_s:
                job = new_job(self.nbits, now, gpu_id)
                self._current[gpu_id] = job
                self._jobs[job.header_bytes] = job
                if self.forget_old:
                    self._forget_old(now)
                if self.on_new_job is not None:
                    self.on_new_job(job)
            return job

    def lookup(self, header_bytes: bytes) -> Job | None:
        with self._lock:
            return self._jobs.get(header_bytes)

    def _forget_old(self, now: float) -> None:
        for key in [k for k, j in self._jobs.items() if now - j.issued_at > 4 * self.max_age_s]:
            del self._jobs[key]
        for gpu_id in [g for g, j in self._current.items() if now - j.issued_at > 4 * self.max_age_s]:
            del self._current[gpu_id]
