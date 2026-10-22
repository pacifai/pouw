"""A recorded session replays to the same verdicts, off the original verifier."""

import base64
import threading

import numpy as np

from pouw.cpu_prover import RpcClient, prove, random_activations
from pouw.jobs import JobIssuer, nbits_for_win_probability
from pouw.recording import Recorder, load, replay_verifier
from pouw.verifier import UnixVerifierServer, Verifier
from pouw.weights import DeclaredWeights

K = 2048
WIN = 1 / 50


def test_recorded_session_replays_to_the_same_verdicts(tmp_path):
    Bt = np.random.default_rng(1).integers(-63, 64, size=(256, K), dtype=np.int8)
    weights = DeclaredWeights({"model.layers.0.self_attn.o_proj.weight": Bt})
    record = tmp_path / "session.jsonl"
    recorder = Recorder(record)
    issuer = JobIssuer(nbits=nbits_for_win_probability(WIN, K), interval_s=60, max_age_s=60, on_new_job=recorder.job)
    sock = str(tmp_path / "gw.sock")
    server = UnixVerifierServer(sock, Verifier(issuer, weights))
    server.recorder = recorder
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = RpcClient(sock)
        job = client.call("getMiningInfo")
        header = base64.b64decode(job["incomplete_header_bytes"])
        proof, _ = prove(header, job["cert_version"], random_activations(128, K, seed=3), Bt, seed=5)
        params = {"plain_proof": proof.to_base64(), "mining_job": job}
        live = [client.call("submitPlainProof", params) for _ in range(2)]  # the second is a replay
        client.close()
    finally:
        server.shutdown()
        server.server_close()

    recording = load(record)
    assert len(recording.jobs) == 1 and len(recording.submissions) == 2
    verifier = replay_verifier(recording, weights)
    replayed = [verifier.check(p["plain_proof"], p["mining_job"], now=t) for t, p in recording.submissions]
    assert [(v.accepted, v.reason) for v in replayed] == [(r["accepted"], r["reason"]) for r in live]
    assert live[0]["accepted"] and live[1]["reason"].startswith("duplicate")
