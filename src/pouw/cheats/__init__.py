"""Cheats on real proofs: each takes an honest submission and returns a cheated submission.

The relay in pouw.cheats.relay sends them between the prover and the verifier,
so recorded runs show which check catches each one.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pearl_mining import IncompleteBlockHeader, PlainProof, verify_plain_proof_for_cert_version
from safetensors import safe_open

from ..jobs import MAX_AGE_S
from ..protocol import RpcClient
from ..prover import random_activations, search
from ..target import bits_to_target, target_to_bits

# Byte offset of A's first opened row in a serialized proof: four 8-byte sizes
# (m, n, k, noise rank), then A's leaf count and leaf length.
A_FIRST_ROW_OFFSET = 48
FORGED_BYTE = A_FIRST_ROW_OFFSET + 100
OTHER_GPU_ID = "GPU-B (relay)"
EASIER_FACTOR = 100_000
# A new job is at most one 5 s interval old, so a proof found within this
# budget still arrives before the 15 s maximum age.
SEARCH_S = 6.0
SEARCH_ROWS = 256


@dataclass
class Change:
    field: str
    original: str
    submitted: str


@dataclass
class Context:
    verifier: RpcClient
    job_seen_at: float
    # The getMiningInfo params the prover sent for the honest proof's job.
    job_params: dict | None = None


class Skipped(Exception):
    """The cheat doesn't apply to this submission, for the reason in the message."""


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
    """Proves a tile of a declared matrix under a real job, where the tile wins only against an easier target.

    The relay plays a prover that lowers its own target: the CPU prover tries
    tiles under the prover's current job until one wins against a target
    `factor` times easier than the job's, but not against the job's own. The
    proof is genuine in every other way: the job, the tile pattern, the noise
    rank, and the declared weights.
    """

    key = "difficulty"
    description = f"Sends a tile that wins only against a target {EASIER_FACTOR:,} times easier than the job's"

    def __init__(self, matrix: np.ndarray | None = None, factor: int = EASIER_FACTOR, search_s: float = SEARCH_S):
        self.matrix = matrix
        self.factor = factor
        self.search_s = search_s

    def apply(self, submission, ctx):
        if self.matrix is None:
            raise Skipped("no declared matrix to prove with: the relay needs --weights")
        job = ctx.verifier.call("getMiningInfo", ctx.job_params)
        header_bytes = base64.b64decode(job["incomplete_header_bytes"])
        header = IncompleteBlockHeader.from_bytes(header_bytes)
        easier_bits = target_to_bits(job["target"] * self.factor)

        def wins_only_easier(proof: PlainProof) -> bool:
            cert = job["cert_version"]
            return (
                verify_plain_proof_for_cert_version(cert, header, proof, easier_bits)[0]
                and not verify_plain_proof_for_cert_version(cert, header, proof)[0]
            )

        A = random_activations(SEARCH_ROWS, self.matrix.shape[1])
        started = time.monotonic()
        proof, tried = search(header_bytes, A, self.matrix, wins_only_easier, deadline=started + self.search_s)
        if proof is None:
            raise Skipped(f"no tile won against the easier target in {tried} tiles and {self.search_s:.0f} s")
        easier = bits_to_target(easier_bits)
        changes = [
            Change("target the tile wins", f"{job['target']:.3e}", f"{easier:.3e} ({self.factor:,} times easier)"),
            Change("search", "a winning tile from the GPU", f"{tried} tiles in {time.monotonic() - started:.1f} s"),
        ]
        return {"plain_proof": proof.to_base64(), "mining_job": job}, changes


class EditedTarget(Cheat):
    key = "edited_target"
    description = "Rewrites the target sent back with the proof to one 256 times easier"
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
        if _gpu_id_hash(other) == _gpu_id_hash(submission["mining_job"]):
            raise Skipped("the verifier returned a job for the prover's own GPU ID, so GPU attribution is off")
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


def _gpu_id_hash(job: dict) -> bytes:
    header = IncompleteBlockHeader.from_bytes(base64.b64decode(job["incomplete_header_bytes"]))
    return bytes(header.merkle_root)


def _int8(byte: int) -> int:
    return byte - 256 if byte > 127 else byte


def load_matrix(paths: list[Path], name: str) -> np.ndarray:
    """One int8 matrix from a checkpoint's safetensors files, by its name in the checkpoint."""
    for path in paths:
        with safe_open(str(path), framework="np") as f:
            if name in f.keys():
                matrix = f.get_tensor(name)
                if matrix.dtype != np.int8 or matrix.ndim != 2:
                    raise ValueError(f"{name} is not an int8 matrix")
                return np.ascontiguousarray(matrix)
    raise ValueError(f"no tensor named {name} in {', '.join(map(str, paths))}")


def all_cheats(
    max_age_s: float = MAX_AGE_S,
    other_gpu_id: str = OTHER_GPU_ID,
    matrix: np.ndarray | None = None,
    easier_factor: int = EASIER_FACTOR,
) -> list[Cheat]:
    return [
        Replay(),
        EasierTarget(matrix, easier_factor),
        ForgedActivation(),
        OtherGpuJob(other_gpu_id),
        LateSubmission(max_age_s),
        EditedTarget(),
    ]


CHEATS = {cheat.key: cheat.description for cheat in all_cheats()}
