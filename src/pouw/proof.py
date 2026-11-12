"""The proof configuration: the GPU prover's tile pattern and noise rank, and the key derived from a job."""

from __future__ import annotations

from blake3 import blake3
from pearl_mining import MiningConfiguration, MMAType, PeriodicPattern

# Proof format version 3 salts the noise seeds with the matrix dimensions.
CERT_VERSION = 3

# Upstream's MinerSettings defaults.
NOISE_RANK = 128
ROWS_PATTERN = [0, 8]
COLS_PATTERN = [c + d for c in range(0, 256, 8) for d in (0, 1)]
TILE_OUTPUTS = len(ROWS_PATTERN) * len(COLS_PATTERN)


def gpu_config(k: int) -> MiningConfiguration:
    """The configuration the GPU prover uses for a GEMM with inner dimension k."""
    return _config(k, NOISE_RANK, ROWS_PATTERN, COLS_PATTERN)


def config_from_indices(k: int, rank: int, rows: list[int], cols: list[int]) -> MiningConfiguration:
    """The configuration a proof's opened rows and columns imply, as upstream's verify rebuilds it."""
    return _config(k, rank, [r - min(rows) for r in rows], [c - min(cols) for c in cols])


def _config(k: int, rank: int, rows: list[int], cols: list[int]) -> MiningConfiguration:
    return MiningConfiguration(
        common_dim=k,
        rank=rank,
        mma_type=MMAType.Int7xInt7ToInt32,
        rows_pattern=PeriodicPattern.from_list(rows),
        cols_pattern=PeriodicPattern.from_list(cols),
        moe=None,
    )


def job_key(header_bytes: bytes, config: MiningConfiguration) -> bytes:
    """The key of every Merkle tree in a job."""
    return blake3(header_bytes + bytes(config.to_bytes())).digest()
