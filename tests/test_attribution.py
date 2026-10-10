"""GPU attribution: each GPU ID gets its own jobs, and each proof names the ID it was made for.

The GPU IDs here are made up. The verifier can't tell a real UUID from a fake
one, which is the limit these tests document rather than test around.
"""

import base64
import json
import sys
import threading

import numpy as np
import pytest
from safetensors.numpy import save_file

from pouw import cpu_prover
from pouw.cpu_prover import RpcClient, prove, random_activations
from pouw.jobs import JobIssuer, nbits_for_win_probability
from pouw.verifier import UnixVerifierServer, Verifier
from pouw.weights import DeclaredWeights

K = 2048
M = 128
WIN = 1 / 50
GPU_A = "GPU-aaaaaaaa-0000-0000-0000-000000000000"
GPU_B = "GPU-bbbbbbbb-0000-0000-0000-000000000000"
MATRIX = "model.layers.0.self_attn.o_proj.weight"


@pytest.fixture(scope="module")
def checkpoint():
    return {MATRIX: np.random.default_rng(1).integers(-63, 64, size=(512, K), dtype=np.int8)}


def _verifier(checkpoint, attribute_gpus):
    issuer = JobIssuer(nbits=nbits_for_win_probability(WIN, K), interval_s=60, max_age_s=60)
    return Verifier(issuer, DeclaredWeights(dict(checkpoint)), attribute_gpus=attribute_gpus)


@pytest.fixture
def verifier(checkpoint):
    return _verifier(checkpoint, attribute_gpus=True)


def _prove(job, checkpoint):
    proof, _ = prove(job.header_bytes, job.cert_version, random_activations(M, K, seed=7), checkpoint[MATRIX], seed=11)
    assert proof is not None
    return proof.to_base64()


def test_flag_off_ignores_gpu_id(checkpoint):
    verifier = _verifier(checkpoint, attribute_gpus=False)
    shared = verifier.issue()
    assert verifier.issue({"gpu_id": GPU_A}) == shared
    assert verifier.issue({"gpu_id": 12345}) == shared  # not even validated
    assert shared.gpu_id is None

    verdict = verifier.check(_prove(shared, checkpoint), shared.to_dict())
    assert verdict.accepted, verdict.reason
    assert verdict.gpu_id is None


def test_each_gpu_id_gets_a_different_header(verifier):
    a = verifier.issue({"gpu_id": GPU_A})
    b = verifier.issue({"gpu_id": GPU_B})
    shared = verifier.issue({})
    assert len({a.header_bytes, b.header_bytes, shared.header_bytes}) == 3
    assert (a.gpu_id, b.gpu_id, shared.gpu_id) == (GPU_A, GPU_B, None)
    # Asking again within the interval returns the same job for that GPU.
    assert verifier.issue({"gpu_id": GPU_A}) == a


def test_each_gpu_job_refreshes_on_the_interval(verifier):
    a = verifier.issuer.current(now=1000.0, gpu_id=GPU_A)
    assert verifier.issuer.current(now=1000.0 + verifier.issuer.interval_s, gpu_id=GPU_A) != a


def test_proof_is_credited_to_the_gpu_whose_job_it_used(verifier, checkpoint):
    job_a = verifier.issue({"gpu_id": GPU_A})
    verdict = verifier.check(_prove(job_a, checkpoint), job_a.to_dict())
    assert verdict.accepted, verdict.reason
    assert verdict.gpu_id == GPU_A
    assert verdict.weight == MATRIX


def test_proof_cannot_be_relabeled_to_another_gpu(verifier, checkpoint):
    job_a = verifier.issue({"gpu_id": GPU_A})
    job_b = verifier.issue({"gpu_id": GPU_B})
    proof = _prove(job_a, checkpoint)

    # Submitting A's proof under B's job fails: A's header seeded the noise.
    verdict = verifier.check(proof, job_b.to_dict())
    assert not verdict.accepted
    assert verdict.reason.startswith("proof verification failed")

    # Extra fields naming B are ignored: the credit comes from the header alone.
    verdict = verifier.check(proof, {**job_a.to_dict(), "gpu_id": GPU_B})
    assert verdict.accepted, verdict.reason
    assert verdict.gpu_id == GPU_A


def test_missing_gpu_id_gets_the_shared_job(verifier, checkpoint):
    shared = verifier.issue()
    assert verifier.issue({}) == shared
    assert verifier.issue({"gpu_id": None}) == shared
    verdict = verifier.check(_prove(shared, checkpoint), shared.to_dict())
    assert verdict.accepted, verdict.reason
    assert verdict.gpu_id is None


@pytest.mark.parametrize("bad", [12345, "", "x" * 129])
def test_malformed_gpu_id_is_refused(verifier, bad):
    with pytest.raises(ValueError):
        verifier.issue({"gpu_id": bad})


def test_cpu_prover_gpu_id_over_the_socket(verifier, checkpoint, tmp_path, monkeypatch, capsys):
    weights = tmp_path / "w.safetensors"
    save_file(dict(checkpoint), str(weights))
    sock, log = str(tmp_path / "gw.sock"), tmp_path / "log.jsonl"
    server = UnixVerifierServer(sock, verifier, log)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        argv = ["cpu_prover", "--weights", str(weights), "--matrix", MATRIX, "--socket", sock]
        monkeypatch.setattr(sys, "argv", [*argv, "--proofs", "1", "--gpu-id", GPU_A])
        cpu_prover.main()
        monkeypatch.setattr(sys, "argv", [*argv, "--proofs", "1"])
        cpu_prover.main()

        # A malformed ID comes back as a JSON-RPC error.
        client = RpcClient(sock)
        with pytest.raises(RuntimeError, match="gpu_id"):
            client.call("getMiningInfo", {"gpu_id": ""})
        job = client.call("getMiningInfo", {"gpu_id": GPU_B})
        assert base64.b64decode(job["incomplete_header_bytes"]) == verifier.issue({"gpu_id": GPU_B}).header_bytes
        client.close()
    finally:
        server.shutdown()
        server.server_close()

    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(r["accepted"], r["gpu_id"]) for r in rows] == [(True, GPU_A), (True, None)]
