"""The cheat relay's changed submissions get the verdicts they should, and the demo labels them."""

import base64
import json
import threading
import time

import numpy as np

from pouw.cheat_relay import Relay, serve
from pouw.cpu_prover import RpcClient, prove, random_activations
from pouw.demo.engine import Engine
from pouw.jobs import JobIssuer, nbits_for_win_probability
from pouw.recording import Recorder
from pouw.verifier import UnixVerifierServer, Verifier
from pouw.weights import DeclaredWeights

K = 2048
NAME = "model.layers.0.self_attn.o_proj.weight"
EXPECTED = {
    "Replay": "duplicate",
    "Difficulty tampering": "job fields differ",
    "Forged proof": "proof verification failed",
    "Proof theft": "proof verification failed",
    "Precomputation": "stale job",
    "GPU ID spoofing": "ok",
}


def test_relay_cheats_are_caught_and_labeled(tmp_path):
    Bt = np.random.default_rng(1).integers(-63, 64, size=(512, K), dtype=np.int8)
    weights = DeclaredWeights({NAME: Bt})
    nbits = nbits_for_win_probability(1 / 50, K)
    recorder = Recorder(tmp_path / "recording.jsonl")
    recorder._write({"type": "settings", "nbits": nbits, "interval_s": 1, "max_age_s": 2, "attribute_gpus": True})
    issuer = JobIssuer(nbits=nbits, interval_s=1, max_age_s=2, on_new_job=recorder.job)
    vsock, rsock = str(tmp_path / "v.sock"), str(tmp_path / "r.sock")
    verifier_server = UnixVerifierServer(vsock, Verifier(issuer, weights, attribute_gpus=True))
    verifier_server.recorder = recorder
    threading.Thread(target=verifier_server.serve_forever, daemon=True).start()
    relay = Relay(vsock, tmp_path / "cheats.jsonl", every=1, start_after=0, max_age_s=2, spoof_window=(4.0, 1000.0))
    relay_server = serve(relay, rsock)
    try:
        client = RpcClient(rsock)
        start = time.time()
        for seed in range(7):
            job = client.call("getMiningInfo", {"gpu_id": "GPU-real"})
            header = base64.b64decode(job["incomplete_header_bytes"])
            proof, _ = prove(header, job["cert_version"], random_activations(128, K, seed=seed), Bt, seed=seed)
            assert client.call("submitPlainProof", {"plain_proof": proof.to_base64(), "mining_job": job})["accepted"]
            time.sleep(max(0.0, start + 0.8 * (seed + 1) - time.time()))
        time.sleep(4)  # the precompute cheat waits out the job's maximum age
        client.close()
    finally:
        relay_server.shutdown()
        verifier_server.shutdown()

    marks = [json.loads(line) for line in (tmp_path / "cheats.jsonl").read_text().splitlines()]
    seen = {m["cheat"] for m in marks}
    assert seen == set(EXPECTED), seen
    for m in marks:
        if m["cheat"] != "GPU ID spoofing":
            assert m["result"]["reason"].startswith(EXPECTED[m["cheat"]]), m

    engine = Engine(tmp_path, weights)
    engine.prepare()
    engine.start()
    engine.started_wall = time.time() - 1000
    engine.tick()
    labeled = [e for e in engine.entries if e.attack]
    assert {e.attack for e in labeled} == set(EXPECTED)
    for e in labeled:
        assert e.reason.startswith(EXPECTED[e.attack]), (e.attack, e.reason)
        assert e.source == "H100 relay"
    spoofed = [e for e in labeled if e.attack == "GPU ID spoofing"]
    assert all(e.gpu_id == relay.spoof_id for e in spoofed)
