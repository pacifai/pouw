"""The job's target: Bitcoin's compact encoding, and the chance that one tile wins against it."""

from __future__ import annotations

from pearl_mining import MiningConfiguration, penalized_target_bound

from .proof import gpu_config

HASH_SPACE = 2**256
MANTISSA_BYTES = 3
SIGN_BIT = 0x800000


def bits_to_target(bits: int) -> int:
    exponent, mantissa = bits >> 24, bits & 0xFFFFFF
    shift = 8 * (exponent - MANTISSA_BYTES)
    return mantissa << shift if shift >= 0 else mantissa >> -shift


def target_to_bits(target: int) -> int:
    """The compact encoding of target, rounded down."""
    size = (target.bit_length() + 7) // 8
    shift = 8 * (size - MANTISSA_BYTES)
    mantissa = target >> shift if shift >= 0 else target << -shift
    if mantissa & SIGN_BIT:
        mantissa >>= 8
        size += 1
    return (size << 24) | mantissa


def win_probability(target: int, config: MiningConfiguration) -> float:
    """Chance that one tile wins, against the target adjusted for the noise rank."""
    bound = penalized_target_bound(target, config)
    if bound is None:
        raise ValueError("target is unusable for this configuration")
    return bound / HASH_SPACE


def bits_for_win_probability(p: float, k: int) -> int:
    """The compact target that gives each of the GPU prover's tiles about a p chance to win."""
    factor = penalized_target_bound(1, gpu_config(k))
    if factor is None:
        raise ValueError(f"no usable target for k={k}")
    return target_to_bits(int(p * HASH_SPACE) // factor)
