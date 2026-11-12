"""Recorded runs become a session file: the relay's cheats fail the expected checks and are marked."""

import base64
import json
import time

import numpy as np
from safetensors.numpy import save_file

from pouw.cheats import CHEATS, all_cheats
from pouw.cheats.relay import Relay
from pouw.jobs import JobIssuer
from pouw.protocol import RpcClient, stop_server
from pouw.prover import prove, random_activations
from pouw.record.recorder import Recorder
from pouw.record.session import FORMAT, build, proven_share
from pouw.target import bits_for_win_probability
from pouw.verifier import Verifier, VerifierService
from pouw.weights import DeclaredWeights

K = 2048
NAME = "model.layers.0.self_attn.o_proj.weight"
EXPECTED = {
    "replay": "new_tile",
    "difficulty": "issued_job",
    "forged": "valid_proof",
    "theft": "valid_proof",
    "precompute": "on_time",
}


def _run_dir(path, model, declared, description):
    path.mkdir()
    (path / "run.json").write_text(
        json.dumps({"description": description, "model": str(model), "declared_model": str(declared)})
    )
    return path


def _record(run_dir, Bt_declared, Bt_served, relay_every=None):
    weights = DeclaredWeights({NAME: Bt_declared})
    issuer = JobIssuer(nbits=bits_for_win_probability(1 / 50, K), interval_s=1, max_age_s=2)
    service = VerifierService(Verifier(issuer, weights, attribute_gpus=True))
    Recorder(run_dir / "recording.jsonl").attach(service, 1 / 50)
    vsock, rsock = str(run_dir / "v.sock"), str(run_dir / "r.sock")
    server = service.start(vsock)
    relay_server = None
    if relay_every:
        relay = Relay(vsock, run_dir / "cheats.jsonl", all_cheats(max_age_s=2), every=relay_every, start_after=0)
        relay_server = relay.start(rsock)
    (run_dir / "serving_since").write_text(str(time.time()))
    try:
        client = RpcClient(rsock if relay_every else vsock)
        start = time.time()
        for seed in range(5 if relay_every else 2):
            job = client.call("getMiningInfo", {"gpu_id": "GPU-real"})
            header = base64.b64decode(job["incomplete_header_bytes"])
            proof, _ = prove(header, job["cert_version"], random_activations(128, K, seed=seed), Bt_served, seed=seed)
            client.call("submitPlainProof", {"plain_proof": proof.to_base64(), "mining_job": job})
            time.sleep(max(0.0, start + 0.8 * (seed + 1) - time.time()))
        time.sleep(4 if relay_every else 0)
        client.close()
    finally:
        if relay_server:
            stop_server(relay_server)
        stop_server(server)


def test_session_marks_every_cheat_with_the_check_that_caught_it(tmp_path):
    rng = np.random.default_rng(1)
    Bt = rng.integers(-63, 64, size=(512, K), dtype=np.int8)
    changed = Bt.copy()
    changed[0, 0] = Bt[0, 0] + 1 if Bt[0, 0] < 63 else Bt[0, 0] - 1
    declared = tmp_path / "declared"
    declared.mkdir()
    save_file({NAME: Bt}, str(declared / "model.safetensors"))

    relay = _run_dir(tmp_path / "relay", declared, declared, "A relay sends cheated copies")
    _record(relay, Bt, Bt, relay_every=1)
    changed_model = _run_dir(tmp_path / "changed-model", tmp_path / "changed", declared, "A changed model")
    _record(changed_model, Bt, changed)

    session = build([relay, changed_model], gpu="CPU")
    assert session["format"] == FORMAT
    relay_run, changed_run = session["runs"]

    cheated = [p for p in relay_run["proofs"] if p["cheat"]]
    assert {p["cheat"] for p in cheated} == set(CHEATS)
    for p in cheated:
        assert not p["accepted"] and p["check"] == EXPECTED[p["cheat"]], p
        assert p["changes"]
    honest = [p for p in relay_run["proofs"] if not p["cheat"]]
    assert honest and all(p["accepted"] and p["work"] > 0 for p in honest)

    assert changed_run["proofs"] and all(
        p["cheat"] == "changed_model" and p["check"] == "declared_weights" for p in changed_run["proofs"]
    )
    assert relay_run["proven_share"] is None


def test_proven_share_counts_only_work_after_serving_started(tmp_path):
    shapes = tmp_path / "gemm_shapes.jsonl"
    rows = [
        {"time": 1.0, "m": 4096, "n": 10, "k": 10, "noisy": True},
        {"time": 3.0, "m": 2048, "n": 10, "k": 10, "noisy": True},
        {"time": 4.0, "m": 512, "n": 10, "k": 10, "noisy": False},
        {"time": 5.0, "m": 1536, "n": 10, "k": 10, "noisy": False},
    ]
    shapes.write_text("\n".join(json.dumps(r) for r in rows))
    assert proven_share(shapes, since=2.0) == 0.5
    assert proven_share(tmp_path / "missing.jsonl", since=0) is None
