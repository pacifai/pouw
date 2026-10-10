"""Each test makes the CPU prover behave one way and checks the verifier's verdict.

The weights are small synthetic int7 matrices named like a Llama checkpoint, so
the tests run in seconds. tests/test_real_weights.py repeats the honest case on
a real tensor from the int7 Llama checkpoint.
"""

import base64
import threading

import numpy as np
import pytest

from pouw.cpu_prover import RpcClient, prove, random_activations
from pouw.jobs import JobIssuer, bits_to_target, new_job, nbits_for_win_probability, target_to_bits
from pouw.verifier import UnixVerifierServer, Verifier
from pouw.weights import DeclaredWeights, fuse_like_vllm

K = 2048  # inner dimension: the library requires at least 16x the noise rank (128)
M = 128  # activation rows: one GEMM tile's worth of tokens
WIN = 1 / 50  # loose target, so the CPU prover finds a winning tile quickly


def _int7(shape, seed):
    return np.random.default_rng(seed).integers(-63, 64, size=shape, dtype=np.int8)


@pytest.fixture(scope="module")
def checkpoint():
    return {
        "model.layers.0.self_attn.o_proj.weight": _int7((512, K), 1),
        "model.layers.0.mlp.gate_proj.weight": _int7((256, K), 2),
        "model.layers.0.mlp.up_proj.weight": _int7((256, K), 3),
        "model.layers.1.self_attn.o_proj.weight": _int7((512, K), 4),
    }


@pytest.fixture
def verifier(checkpoint):
    issuer = JobIssuer(nbits=nbits_for_win_probability(WIN, K), interval_s=60, max_age_s=60)
    return Verifier(issuer, DeclaredWeights(fuse_like_vllm(checkpoint)))


def _job(verifier):
    job = verifier.issuer.current()
    return job, job.to_dict()


def _prove(job, Bt, **kw):
    proof, _ = prove(job.header_bytes, job.cert_version, random_activations(M, K, seed=7), Bt, seed=11, **kw)
    assert proof is not None
    return proof


def test_compact_target_round_trip():
    for target in [1 << 200, (1 << 230) + 12345, 0x7FFFFF << 100]:
        assert bits_to_target(target_to_bits(target)) <= target
        assert bits_to_target(target_to_bits(target)) > target * 0.99


def test_honest_proof_is_accepted(verifier, checkpoint):
    job, job_dict = _job(verifier)
    Bt = checkpoint["model.layers.1.self_attn.o_proj.weight"]
    verdict = verifier.check(_prove(job, Bt).to_base64(), job_dict)
    assert verdict.accepted, verdict.reason
    assert verdict.weight == "model.layers.1.self_attn.o_proj.weight"
    assert 0 < verdict.win_probability < 1


def test_proof_against_fused_tensor_names_it(verifier, checkpoint):
    job, job_dict = _job(verifier)
    Bt = np.concatenate([checkpoint["model.layers.0.mlp.gate_proj.weight"], checkpoint["model.layers.0.mlp.up_proj.weight"]])
    verdict = verifier.check(_prove(job, Bt).to_base64(), job_dict)
    assert verdict.accepted, verdict.reason
    assert verdict.weight == "model.layers.0.mlp.gate_up_proj.weight"


def test_one_changed_weight_is_rejected(verifier, checkpoint):
    # The library's verify passes (the proof is consistent with the matrix the prover
    # used), but that matrix is not the declared one.
    job, job_dict = _job(verifier)
    Bt = checkpoint["model.layers.0.self_attn.o_proj.weight"].copy()
    Bt[100, 200] = Bt[100, 200] + 1 if Bt[100, 200] < 63 else Bt[100, 200] - 1
    verdict = verifier.check(_prove(job, Bt).to_base64(), job_dict)
    assert not verdict.accepted
    assert verdict.reason.startswith("weight matrix is not")


def test_losing_tile_is_rejected(verifier, checkpoint):
    job, job_dict = _job(verifier)
    Bt = checkpoint["model.layers.0.self_attn.o_proj.weight"]
    verdict = verifier.check(_prove(job, Bt, want_win=False).to_base64(), job_dict)
    assert not verdict.accepted
    assert verdict.reason.startswith("proof verification failed")


def test_replayed_proof_is_rejected(verifier, checkpoint):
    job, job_dict = _job(verifier)
    proof = _prove(job, checkpoint["model.layers.0.self_attn.o_proj.weight"]).to_base64()
    assert verifier.check(proof, job_dict).accepted
    verdict = verifier.check(proof, job_dict)
    assert not verdict.accepted
    assert verdict.reason.startswith("duplicate")


def test_job_not_issued_here_is_rejected(verifier, checkpoint):
    job = new_job(verifier.issuer.nbits)  # right format and difficulty, but never handed out
    proof = _prove(job, checkpoint["model.layers.0.self_attn.o_proj.weight"]).to_base64()
    verdict = verifier.check(proof, job.to_dict())
    assert not verdict.accepted
    assert verdict.reason.startswith("unknown job")


def test_stale_job_is_rejected(verifier, checkpoint):
    job, job_dict = _job(verifier)
    proof = _prove(job, checkpoint["model.layers.0.self_attn.o_proj.weight"]).to_base64()
    verdict = verifier.check(proof, job_dict, now=job.issued_at + verifier.issuer.max_age_s + 1)
    assert not verdict.accepted
    assert verdict.reason.startswith("stale job")


def test_easier_target_than_issued_is_rejected(verifier, checkpoint):
    job, job_dict = _job(verifier)
    proof = _prove(job, checkpoint["model.layers.0.self_attn.o_proj.weight"]).to_base64()
    verdict = verifier.check(proof, {**job_dict, "target": job.target * 2})
    assert not verdict.accepted
    assert verdict.reason.startswith("job fields differ")


def test_over_the_socket(verifier, checkpoint, tmp_path):
    sock = str(tmp_path / "gw.sock")
    server = UnixVerifierServer(sock, verifier)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = RpcClient(sock)
        job_dict = client.call("getMiningInfo")
        header = base64.b64decode(job_dict["incomplete_header_bytes"])
        Bt = checkpoint["model.layers.0.self_attn.o_proj.weight"]
        proof, _ = prove(header, job_dict["cert_version"], random_activations(M, K, seed=3), Bt, seed=5)
        result = client.call("submitPlainProof", {"plain_proof": proof.to_base64(), "mining_job": job_dict})
        assert result == {"accepted": True, "reason": "ok"}
        client.close()
    finally:
        server.shutdown()
        server.server_close()


def test_tiles_cover_each_output_exactly_once():
    # The README's work estimate assumes this: one tile is 2 x 64 outputs, and
    # the tiles of a GEMM partition its output matrix.
    from pouw.cpu_prover import tile_starts
    from pouw.jobs import COLS_PATTERN, ROWS_PATTERN

    m, n = 256, 4096
    covered = [
        (r + dr, c + dc)
        for r in tile_starts(ROWS_PATTERN, m)
        for c in tile_starts(COLS_PATTERN, n)
        for dr in ROWS_PATTERN
        for dc in COLS_PATTERN
    ]
    assert len(covered) == len(set(covered)) == m * n
