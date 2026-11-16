"""A recorded session replays to the same verdicts, off the original verifier."""

import base64
import time

import numpy as np
import pytest

from pouw.jobs import JobIssuer
from pouw.protocol import RpcClient, stop_server
from pouw.prover import prove, random_activations
from pouw.record.recorder import Recorder, load, replay
from pouw.target import bits_for_win_probability
from pouw.verifier import Verifier, VerifierService
from pouw.weights import DeclaredWeights

K = 2048
WIN = 1 / 50


def test_recorded_session_replays_to_the_same_verdicts(tmp_path):
    Bt = np.random.default_rng(1).integers(-63, 64, size=(256, K), dtype=np.int8)
    weights = DeclaredWeights({"model.layers.0.self_attn.o_proj.weight": Bt})
    record = tmp_path / "session.jsonl"
    issuer = JobIssuer(nbits=bits_for_win_probability(WIN, K), interval_s=60, max_age_s=60)
    service = VerifierService(Verifier(issuer, weights))
    Recorder(record).attach(service, WIN)
    sock = str(tmp_path / "gw.sock")
    server = service.start(sock)
    try:
        client = RpcClient(sock)
        job = client.call("getMiningInfo")
        header = base64.b64decode(job["incomplete_header_bytes"])
        proof, _ = prove(header, job["cert_version"], random_activations(128, K, seed=3), Bt, seed=5)
        params = {"plain_proof": proof.to_base64(), "mining_job": job}
        live = [client.call("submitPlainProof", params) for _ in range(2)]  # the second is a replay
        client.close()
    finally:
        stop_server(server)

    recording = load(record)
    assert len(recording.jobs) == 1 and len(recording.submissions) == 2
    replayed = replay(recording, weights)
    assert [(v.accepted, v.reason) for v in replayed] == [(r["accepted"], r["reason"]) for r in live]
    assert live[0]["accepted"] and live[1]["reason"].startswith("duplicate")


def test_replay_matches_live_verdicts_at_the_age_limit_and_after_jobs_are_forgotten(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    Bt = np.random.default_rng(1).integers(-63, 64, size=(256, K), dtype=np.int8)
    weights = DeclaredWeights({"model.layers.0.self_attn.o_proj.weight": Bt})
    issuer = JobIssuer(nbits=bits_for_win_probability(WIN, K), interval_s=5, max_age_s=15)
    service = VerifierService(Verifier(issuer, weights))
    Recorder(tmp_path / "session.jsonl").attach(service, WIN)

    job = service.dispatch("getMiningInfo", {})
    header = base64.b64decode(job["incomplete_header_bytes"])
    proofs = [prove(header, job["cert_version"], random_activations(128, K, seed=s), Bt, seed=s)[0] for s in range(4)]
    live = []
    # At the limit, just past it, later but still remembered, and after a new job made the issuer forget it.
    for proof, at in zip(proofs, [1015.0, 1015.000001, 1030.0, 1061.5]):
        if at > 1060:
            clock[0] = 1061.0
            service.dispatch("getMiningInfo", {})
        clock[0] = at
        live.append(service.dispatch("submitPlainProof", {"plain_proof": proof.to_base64(), "mining_job": job}))

    replayed = replay(load(tmp_path / "session.jsonl"), weights)
    assert [(v.accepted, v.reason) for v in replayed] == [(r["accepted"], r["reason"]) for r in live]
    assert [v.check for v in replayed] == [None, "on_time", "on_time", "issued_job"]


def test_recording_without_settings_is_an_error(tmp_path):
    path = tmp_path / "recording.jsonl"
    path.write_text('{"type": "submission", "time": 1.0, "params": {}}\n')
    with pytest.raises(ValueError, match="no settings line"):
        load(path)
