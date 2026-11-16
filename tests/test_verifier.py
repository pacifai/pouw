"""Each test makes the CPU prover behave one way and checks the verifier's verdict.

The weights are small synthetic int7 matrices named like a Llama checkpoint, so
the tests run in seconds. tests/test_real_weights.py repeats the honest case on
a real tensor from the int7 Llama checkpoint.
"""

import base64

import numpy as np
import pytest
from pearl_mining import IncompleteBlockHeader, PlainProof, verify_plain_proof_for_cert_version

from pouw.jobs import KEEP_MAX_AGES, JobIssuer, new_job
from pouw.proof import COLS_PATTERN, ROWS_PATTERN
from pouw.protocol import RpcClient, stop_server
from pouw.prover import prove, random_activations, tile_starts
from pouw.target import bits_for_win_probability, bits_to_target, target_to_bits
from pouw.verifier import Verifier, VerifierService, _upstream
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
    issuer = JobIssuer(nbits=bits_for_win_probability(WIN, K), interval_s=60, max_age_s=60)
    return Verifier(issuer, DeclaredWeights(fuse_like_vllm(dict(checkpoint))))


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
    Bt = np.concatenate(
        [checkpoint["model.layers.0.mlp.gate_proj.weight"], checkpoint["model.layers.0.mlp.up_proj.weight"]]
    )
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
    assert verdict.check == "declared_weights"


def test_losing_tile_is_rejected(verifier, checkpoint):
    job, job_dict = _job(verifier)
    Bt = checkpoint["model.layers.0.self_attn.o_proj.weight"]
    verdict = verifier.check(_prove(job, Bt, want_win=False).to_base64(), job_dict)
    assert not verdict.accepted
    assert verdict.check == "valid_proof"
    assert verdict.reason.startswith("proof verification failed")


def test_smaller_tile_is_rejected(verifier, checkpoint):
    # Upstream's verify accepts a 2 x 16 tile, which wins as often per output as
    # the GPU's 2 x 64 tile, so crediting it as a full tile would count 4x the work.
    job, job_dict = _job(verifier)
    Bt = checkpoint["model.layers.0.self_attn.o_proj.weight"]
    verdict = verifier.check(_prove(job, Bt, cols_pattern=COLS_PATTERN[:16]).to_base64(), job_dict)
    assert not verdict.accepted
    assert verdict.check == "valid_proof"
    assert verdict.reason.startswith("tile pattern")
    assert verdict.work == 0


@pytest.mark.parametrize("n, k", [(512, 1 << 22), (7, K)])
def test_undeclared_shape_is_rejected_before_verify(verifier, checkpoint, n, k):
    # With k = 2^22, upstream's verify would allocate about 1.8 GB before its own checks.
    job, job_dict = _job(verifier)
    honest = _prove(job, checkpoint["model.layers.0.self_attn.o_proj.weight"])
    forged = PlainProof(honest.m, n, k, honest.noise_rank, honest.a, honest.bt, None)
    verdict = verifier.check(forged.to_base64(), job_dict)
    assert not verdict.accepted
    assert verdict.check == "declared_weights"
    assert verdict.reason.startswith("no declared matrix has shape")


def test_zero_noise_rank_is_rejected_before_verify(verifier, checkpoint):
    # Upstream's verify divides by the rank.
    job, job_dict = _job(verifier)
    honest = _prove(job, checkpoint["model.layers.0.self_attn.o_proj.weight"])
    forged = PlainProof(honest.m, honest.n, honest.k, 0, honest.a, honest.bt, None)
    verdict = verifier.check(forged.to_base64(), job_dict)
    assert not verdict.accepted
    assert verdict.check == "valid_proof"
    assert verdict.reason.startswith("tile pattern")


def test_replayed_proof_is_rejected(verifier, checkpoint):
    job, job_dict = _job(verifier)
    proof = _prove(job, checkpoint["model.layers.0.self_attn.o_proj.weight"]).to_base64()
    assert verifier.check(proof, job_dict).accepted
    verdict = verifier.check(proof, job_dict)
    assert not verdict.accepted
    assert verdict.check == "new_tile"


def test_forgotten_job_drops_its_tiles(verifier, checkpoint):
    issuer = verifier.issuer
    job = issuer.current(now=0)
    proof = _prove(job, checkpoint["model.layers.0.self_attn.o_proj.weight"]).to_base64()
    assert verifier.check(proof, job.to_dict(), now=1).accepted
    assert job.header_bytes in verifier._seen_tiles

    later = KEEP_MAX_AGES * issuer.max_age_s + 1
    issuer.current(now=later)
    assert job.header_bytes not in verifier._seen_tiles
    assert verifier.check(proof, job.to_dict(), now=later).check == "issued_job"


def test_job_not_issued_here_is_rejected(verifier, checkpoint):
    job = new_job(verifier.issuer.nbits)  # right format and difficulty, but never handed out
    proof = _prove(job, checkpoint["model.layers.0.self_attn.o_proj.weight"]).to_base64()
    verdict = verifier.check(proof, job.to_dict())
    assert not verdict.accepted
    assert verdict.check == "issued_job"
    assert verdict.reason.startswith("unknown job")


def test_stale_job_is_rejected(verifier, checkpoint):
    job, job_dict = _job(verifier)
    proof = _prove(job, checkpoint["model.layers.0.self_attn.o_proj.weight"]).to_base64()
    verdict = verifier.check(proof, job_dict, now=job.issued_at + verifier.issuer.max_age_s + 1)
    assert not verdict.accepted
    assert verdict.check == "on_time"


def test_easier_target_than_issued_is_rejected(verifier, checkpoint):
    job, job_dict = _job(verifier)
    proof = _prove(job, checkpoint["model.layers.0.self_attn.o_proj.weight"]).to_base64()
    verdict = verifier.check(proof, {**job_dict, "target": job.target * 2})
    assert not verdict.accepted
    assert verdict.check == "issued_job"
    assert verdict.reason.startswith("job fields differ")


@pytest.mark.parametrize(
    "params, check",
    [
        (lambda j, p: ["not", "an", "object"], "issued_job"),
        (lambda j, p: {"plain_proof": p}, "issued_job"),
        (lambda j, p: {"plain_proof": p, "mining_job": "job"}, "issued_job"),
        (lambda j, p: {"plain_proof": p, "mining_job": {**j, "incomplete_header_bytes": None}}, "issued_job"),
        (lambda j, p: {"plain_proof": p, "mining_job": {**j, "incomplete_header_bytes": "!!"}}, "issued_job"),
        (lambda j, p: {"mining_job": j}, "valid_proof"),
        (lambda j, p: {"plain_proof": 42, "mining_job": j}, "valid_proof"),
        (lambda j, p: {"plain_proof": "!!", "mining_job": j}, "valid_proof"),
    ],
)
def test_malformed_submission_is_rejected(verifier, checkpoint, params, check):
    job, job_dict = _job(verifier)
    proof = _prove(job, checkpoint["model.layers.0.self_attn.o_proj.weight"]).to_base64()
    verdict = verifier.check_submission(params(job_dict, proof))
    assert not verdict.accepted
    assert verdict.check == check


def test_upstream_panic_becomes_value_error(verifier, checkpoint):
    job, _ = _job(verifier)
    honest = _prove(job, checkpoint["model.layers.0.self_attn.o_proj.weight"])
    zero_rank = PlainProof(honest.m, honest.n, honest.k, 0, honest.a, honest.bt, None)
    header = IncompleteBlockHeader.from_bytes(job.header_bytes)
    with pytest.raises(ValueError, match="upstream panicked"):
        _upstream(verify_plain_proof_for_cert_version, job.cert_version, header, zero_rank)


def test_over_the_socket(verifier, checkpoint, tmp_path):
    sock = str(tmp_path / "gw.sock")
    server = VerifierService(verifier).start(sock)
    try:
        client = RpcClient(sock)
        job_dict = client.call("getMiningInfo")
        header = base64.b64decode(job_dict["incomplete_header_bytes"])
        Bt = checkpoint["model.layers.0.self_attn.o_proj.weight"]
        proof, _ = prove(header, job_dict["cert_version"], random_activations(M, K, seed=3), Bt, seed=5)
        result = client.call("submitPlainProof", {"plain_proof": proof.to_base64(), "mining_job": job_dict})
        assert result == {"accepted": True, "reason": "ok"}
        result = client.call("submitPlainProof", {"mining_job": job_dict})
        assert result["accepted"] is False
        client.close()
    finally:
        stop_server(server)


def test_tiles_cover_each_output_exactly_once():
    # The README's work estimate assumes this: one tile is 2 x 64 outputs, and
    # the tiles of a GEMM partition its output matrix.
    m, n = 256, 4096
    covered = [
        (r + dr, c + dc)
        for r in tile_starts(ROWS_PATTERN, m)
        for c in tile_starts(COLS_PATTERN, n)
        for dr in ROWS_PATTERN
        for dc in COLS_PATTERN
    ]
    assert len(covered) == len(set(covered)) == m * n
