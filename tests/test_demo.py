"""Every demo attack, run against a small recorded session: detected ones fail the expected check."""

import base64
import threading
import time

import numpy as np
import pytest

from pouw.cpu_prover import RpcClient, prove, random_activations
from pouw.demo.engine import Engine, checks_for
from pouw.jobs import JobIssuer, nbits_for_win_probability
from pouw.recording import Recorder
from pouw.verifier import UnixVerifierServer, Verifier
from pouw.weights import DeclaredWeights

K = 2048
NAME = "model.layers.0.self_attn.o_proj.weight"


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    Bt = np.random.default_rng(1).integers(-63, 64, size=(512, K), dtype=np.int8)
    weights = DeclaredWeights({NAME: Bt})
    recorder = Recorder(tmp / "recording.jsonl")
    recorder._write({"type": "settings", "nbits": nbits_for_win_probability(1 / 50, K), "interval_s": 60, "max_age_s": 15, "attribute_gpus": True})
    issuer = JobIssuer(nbits=nbits_for_win_probability(1 / 50, K), interval_s=60, max_age_s=15, on_new_job=recorder.job)
    sock = str(tmp / "gw.sock")
    server = UnixVerifierServer(sock, Verifier(issuer, weights, attribute_gpus=True))
    server.recorder = recorder
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = RpcClient(sock)
        job = client.call("getMiningInfo", {"gpu_id": "GPU-test-h100"})
        header = base64.b64decode(job["incomplete_header_bytes"])
        for seed in (3, 4):
            proof, _ = prove(header, job["cert_version"], random_activations(128, K, seed=seed), Bt, seed=seed)
            assert client.call("submitPlainProof", {"plain_proof": proof.to_base64(), "mining_job": job})["accepted"]
        client.close()
    finally:
        server.shutdown()
        server.server_close()
    e = Engine(tmp, weights)
    e.prepare()
    e.start()
    e.started_wall = time.time() - 1000  # jump to the end of the replay
    e.tick()
    for row in e.entries:
        row.source = "H100"  # the fixture's prover stands in for the GPU
    return e


def test_replay_reaches_the_same_verdicts(engine):
    assert [x.accepted for x in engine.entries[:2]] == [True, True]
    assert all(x.gpu_id == "GPU-test-h100" for x in engine.entries[:2])


@pytest.mark.parametrize(
    "attack, failed_check",
    [
        ("replay", "Replay protection"),
        ("precompute", "Freshness"),
        ("difficulty", "Job binding"),
        ("forged", "Proof verification"),
        ("model", "Weight commitment"),
        ("theft", "Proof verification"),
    ],
)
def test_detected_attacks_fail_their_check(engine, attack, failed_check):
    detail = engine.entry(engine.attack(attack))
    assert not detail["accepted"], detail["reason"]
    assert [c["name"] for c in detail["checks"] if c["status"] == "fail"] == [failed_check], detail["reason"]


def test_undetected_attacks_are_accepted(engine):
    synthetic = engine.entry(engine.attack("synthetic"))
    assert synthetic["accepted"], synthetic["reason"]
    spoof = engine.entry(engine.attack("spoof"))
    assert spoof["accepted"], spoof["reason"]
    assert spoof["gpu_id"] == "GPU-test-h100" and spoof["source"] == "laptop"


def test_checks_for_accepted_proof_all_pass():
    assert {c["status"] for c in checks_for(True, "ok")} == {"pass"}


def test_forged_attack_changes_one_activation(engine):
    detail = engine.entry(engine.attack("forged"))
    assert "Hash A mismatch" in detail["reason"]
    change = detail["changes"][0]
    assert change["field"].startswith("activation A[")
    assert int(change["submitted"]) == (int(change["original"]) + 129) % 256 - 128


def test_live_mode_checks_proofs_as_they_arrive(tmp_path):
    from pouw.demo.engine import LiveEngine
    from pouw.demo.server import serve_live_verifier

    Bt = np.random.default_rng(2).integers(-63, 64, size=(512, K), dtype=np.int8)
    weights = DeclaredWeights({NAME: Bt})
    issuer = JobIssuer(nbits=nbits_for_win_probability(1 / 50, K), interval_s=60, max_age_s=15)
    engine = LiveEngine(weights, issuer, attribute_gpus=True, recorder=Recorder(tmp_path / "rec.jsonl"))
    sock = str(tmp_path / "live.sock")
    server = serve_live_verifier(engine, sock, None)
    try:
        client = RpcClient(sock)
        job = client.call("getMiningInfo", {"gpu_id": "GPU-live"})
        header = base64.b64decode(job["incomplete_header_bytes"])
        proof, _ = prove(header, job["cert_version"], random_activations(128, K, seed=9), Bt, seed=9)
        result = client.call("submitPlainProof", {"plain_proof": proof.to_base64(), "mining_job": job})
        client.close()
    finally:
        server.shutdown()
        server.server_close()
    assert result == {"accepted": True, "reason": "ok"}
    engine.tick()
    state = engine.state()
    assert state["mode"] == "live" and state["totals"]["accepted"] == 1
    assert state["log"][0]["gpu_id"] == "GPU-live" and state["log"][0]["delay_s"] >= 0
    assert state["delay"] is not None
    assert not engine.entry(engine.attack("replay"))["accepted"]
    assert "submission" in (tmp_path / "rec.jsonl").read_text()
