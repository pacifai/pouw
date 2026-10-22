"""The honest and tampered cases on real tensors from the int7 Llama checkpoint.

Needs data/llama-sample.safetensors. To create it, run:

    python scripts/fetch_tensors.py data/llama-sample.safetensors \
        model.layers.0.self_attn.o_proj.weight \
        model.layers.20.self_attn.q_proj.weight \
        model.layers.20.self_attn.k_proj.weight \
        model.layers.20.self_attn.v_proj.weight
"""

from pathlib import Path

import numpy as np
import pytest

from pouw.cpu_prover import prove, random_activations
from pouw.jobs import JobIssuer, nbits_for_win_probability
from pouw.verifier import Verifier
from pouw.weights import DeclaredWeights

SAMPLE = Path(__file__).parent.parent / "data" / "llama-sample.safetensors"
pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="run scripts/fetch_tensors.py first")


@pytest.fixture(scope="module")
def weights():
    return DeclaredWeights.from_safetensors([SAMPLE])


@pytest.fixture
def verifier(weights):
    return Verifier(JobIssuer(nbits=nbits_for_win_probability(1 / 50, 4096), interval_s=60, max_age_s=60), weights)


def test_checkpoint_weights_fit_the_int7_range(weights):
    # The library's verify rejects any value outside [-64, 64].
    for name, t in weights.tensors.items():
        assert t.min() >= -64 and t.max() <= 64, name


def test_qkv_is_fused_like_vllm(weights):
    assert weights.tensors["model.layers.20.self_attn.qkv_proj.weight"].shape == (6144, 4096)


@pytest.mark.parametrize("name", ["model.layers.0.self_attn.o_proj.weight", "model.layers.20.self_attn.qkv_proj.weight"])
def test_real_matrix_proof_is_accepted(verifier, weights, name):
    job = verifier.issuer.current()
    proof, _ = prove(job.header_bytes, job.cert_version, random_activations(128, 4096, seed=1), weights.tensors[name], seed=2)
    verdict = verifier.check(proof.to_base64(), job.to_dict())
    assert verdict.accepted, verdict.reason
    assert verdict.weight == name


def test_real_matrix_with_one_changed_weight_is_rejected(verifier, weights):
    job = verifier.issuer.current()
    Bt = weights.tensors["model.layers.0.self_attn.o_proj.weight"].copy()
    Bt[4000, 17] = np.int8(Bt[4000, 17] + 1 if Bt[4000, 17] < 63 else Bt[4000, 17] - 1)
    proof, _ = prove(job.header_bytes, job.cert_version, random_activations(128, 4096, seed=1), Bt, seed=2)
    verdict = verifier.check(proof.to_base64(), job.to_dict())
    assert not verdict.accepted
    assert verdict.reason.startswith("weight matrix is not")
