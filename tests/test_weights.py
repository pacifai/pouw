import threading

import numpy as np

from pouw import weights
from pouw.weights import DeclaredWeights

THREADS = 8


def test_concurrent_misses_hash_each_matrix_once(monkeypatch):
    rng = np.random.default_rng(0)
    declared = DeclaredWeights({f"m{i}": rng.integers(-63, 64, size=(64, 32), dtype=np.int8) for i in range(3)})
    calls = []
    real = weights.merkle_root
    start = threading.Barrier(THREADS)

    def counted(matrix, key):
        calls.append(key)
        return real(matrix, key)

    monkeypatch.setattr(weights, "merkle_root", counted)
    key = bytes(32)
    root = real(declared.tensors["m1"], key)
    results = []

    def check():
        start.wait()
        results.append(declared.match(key, 64, 32, root))

    threads = [threading.Thread(target=check) for _ in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == ["m1"] * THREADS
    assert len(calls) == 3
