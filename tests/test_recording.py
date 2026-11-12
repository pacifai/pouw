"""A recorded session replays to the same verdicts, off the original verifier."""

import base64

import numpy as np

from pouw.jobs import JobIssuer
from pouw.protocol import RpcClient, stop_server
from pouw.prover import prove, random_activations
from pouw.record.recorder import Recorder, load, replay_verifier
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
    verifier = replay_verifier(recording, weights)
    replayed = [verifier.check(p["plain_proof"], p["mining_job"], now=t) for t, p in recording.submissions]
    assert [(v.accepted, v.reason) for v in replayed] == [(r["accepted"], r["reason"]) for r in live]
    assert live[0]["accepted"] and live[1]["reason"].startswith("duplicate")
