"""Building a Merkle tree must let other Python threads run.

The upstream vLLM plugin builds proofs inside vLLM's engine process. If the
proof library holds Python's global interpreter lock (GIL) while it hashes a
matrix, every other thread in that process stops until the hash finishes, and
inference stalls. patches/0001-release-gil-in-merkle-tree.patch makes the
library release the GIL during that work. This test fails without the patch.
"""

import threading
import time

import numpy as np
from pearl_mining import MerkleTree, pad_to_chunk_boundary

ROWS, COLS = 28672, 4096  # the fused gate_up_proj weight of Llama 3.1 8B: 117 MB of int8
BUILDS = 5
SLEEP_S = 0.001
KEY = bytes(range(32))


def test_tree_build_lets_other_threads_run():
    raw = np.random.default_rng(0).integers(-63, 64, size=(ROWS, COLS), dtype=np.int8).tobytes()

    stop = threading.Event()
    ticks: list[float] = []

    def count():
        # Each pass needs the GIL, so a thread that holds it elsewhere stops
        # the count.
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(SLEEP_S)

    counter = threading.Thread(target=count)
    counter.start()
    time.sleep(0.05)  # let the counter settle into its loop

    # Build the tree the way pouw.weights.merkle_root does, a few times over,
    # and record when each build starts and ends.
    windows = []
    for _ in range(BUILDS):
        start = time.perf_counter()
        root = MerkleTree(data=pad_to_chunk_boundary(raw), key=KEY).root
        windows.append((start, time.perf_counter()))
        assert len(root) == 32

    stop.set()
    counter.join()

    build_s = sum(end - start for start, end in windows)
    turns = sum(1 for t in ticks for start, end in windows if start <= t <= end)
    expected = build_s / SLEEP_S
    print(
        f"{BUILDS} builds took {build_s * 1000:.0f} ms; the counter ran {turns} times "
        f"during them, about {expected:.0f} if never blocked"
    )

    # A library that holds the GIL for each whole build leaves the counter one
    # or two turns per build, between the two calls. The threshold leaves room
    # for a slow or busy machine.
    assert turns >= 0.3 * expected
