"""Cheats on real proofs: each takes an honest submission and returns a cheated copy.

The relay in pouw.cheats.relay sends them between the prover and the verifier,
so recorded runs show which check catches each one.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass

from ..jobs import MAX_AGE_S
from ..protocol import RpcClient

# Byte offset of A's first opened row in a serialized proof: four 8-byte sizes
# (m, n, k, noise rank), then A's leaf count and leaf length.
A_FIRST_ROW_OFFSET = 48
FORGED_BYTE = A_FIRST_ROW_OFFSET + 100
OTHER_GPU_ID = "GPU-B (relay)"


@dataclass
class Change:
    field: str
    original: str
    submitted: str


@dataclass
class Context:
    verifier: RpcClient
    job_seen_at: float


class Cheat:
    key: str
    description: str
    # The cheated copy is byte for byte the honest submission, so only its second arrival is the cheat.
    resends_unchanged = False

    def apply(self, submission: dict, ctx: Context) -> tuple[dict, list[Change]]:
        raise NotImplementedError


class Replay(Cheat):
    key = "replay"
    description = "Sends the same proof again"
    resends_unchanged = True

    def apply(self, submission, ctx):
        return submission, [Change("submitted", "once", "again, unchanged")]


class EasierTarget(Cheat):
    key = "difficulty"
    description = "Makes the job's target 256 times easier"
    factor = 256

    def apply(self, submission, ctx):
        job = submission["mining_job"]
        easier = {**job, "target": job["target"] * self.factor}
        change = Change("job target", f"{job['target']:.3e}", f"{easier['target']:.3e} ({self.factor} times easier)")
        return {**submission, "mining_job": easier}, [change]


class ForgedActivation(Cheat):
    key = "forged"
    description = "Changes one activation in the proof's first opened row by one"

    def apply(self, submission, ctx):
        raw = bytearray(base64.b64decode(submission["plain_proof"]))
        before = _int8(raw[FORGED_BYTE])
        raw[FORGED_BYTE] = (raw[FORGED_BYTE] + 1) % 256
        change = Change("activation in the first opened row of A", str(before), str(_int8(raw[FORGED_BYTE])))
        return {**submission, "plain_proof": base64.b64encode(bytes(raw)).decode("ascii")}, [change]


class OtherGpuJob(Cheat):
    key = "theft"
    description = "Submits the proof under a job issued to another GPU ID"

    def __init__(self, other_gpu_id: str = OTHER_GPU_ID):
        self.other_gpu_id = other_gpu_id

    def apply(self, submission, ctx):
        other = ctx.verifier.call("getMiningInfo", {"gpu_id": self.other_gpu_id})
        change = Change("job", "the prover's job", f"{self.other_gpu_id}'s job")
        return {**submission, "mining_job": other}, [change]


class LateSubmission(Cheat):
    key = "precompute"
    description = "Holds the proof until its job is one second past the maximum age"
    resends_unchanged = True
    margin_s = 1.0

    def __init__(self, max_age_s: float = MAX_AGE_S):
        self.max_age_s = max_age_s

    def apply(self, submission, ctx):
        send_at = ctx.job_seen_at + self.max_age_s + self.margin_s
        time.sleep(max(0.0, send_at - time.time()))
        delay = f"{self.max_age_s + self.margin_s:.0f} s after the job was issued"
        return submission, [Change("submitted", "right after the win", delay)]


def _int8(byte: int) -> int:
    return byte - 256 if byte > 127 else byte


def all_cheats(max_age_s: float = MAX_AGE_S, other_gpu_id: str = OTHER_GPU_ID) -> list[Cheat]:
    return [Replay(), EasierTarget(), ForgedActivation(), OtherGpuJob(other_gpu_id), LateSubmission(max_age_s)]


CHEATS = {cheat.key: cheat.description for cheat in all_cheats()}
