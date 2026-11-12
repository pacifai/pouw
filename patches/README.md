# Patches to upstream

Each `.patch` file is a `git diff` against the pinned upstream commit,
`d7c517d267f10f5bcbbdc366f3e603942f9a5ab5`, with paths relative to the
upstream repository root. `docker/Dockerfile` and `scripts/setup_pod.sh`
check out that commit and apply every `*.patch` file in name order before
building.

To change a patch, check out the pinned commit, apply the existing patches,
edit the code, and save the result with `git diff > patches/<name>.patch`.
When the pin moves, check that each patch still applies, and rerun the tests.

## 0001-release-gil-in-proof-trees.patch

Lets the vLLM plugin build each proof's Merkle trees with Python's global
interpreter lock (GIL) released. The GIL is the lock a Python thread must
hold to run Python code. The patch is the four commits on
`release-gil-in-proofs` in the `pacifai/pearl` fork, proposed upstream:

- `MerkleTree::from_vec` builds a tree from an existing buffer.
- `MerkleTree.from_buffer(data, key)`, in the Python bindings, takes an int8
  buffer such as a NumPy array, copies it once into a zero-padded buffer with
  the GIL held, and builds the tree with the GIL released. Its root equals
  `MerkleTree(pad_to_chunk_boundary(bytes(data)), key)`.
- The plugin's `MatrixMerkleTree` passes the tensor's NumPy view to
  `from_buffer`.

The verifier and the CPU prover also build their trees with `from_buffer`.

Upstream builds each proof inside vLLM's engine process. Building a tree
copied the matrix four times (`tobytes()`, two copies in
`pad_to_chunk_boundary`, and the tree's own copy) and held the GIL through
the copies and the hash, so every win stalled inference. The copy keeps the
GIL so that no other Python thread can change the buffer mid-copy.

On one H100, with alternating builds, the same load ran in 21.5 s with the
patch and 32.4 s without it, against 12.3 s with proofs off. The patch's
tests are in the fork.

## 0002-send-gpu-id.patch

Lets the vLLM plugin send its GPU's UUID in `getMiningInfo`, so a verifier
started with `--attribute-gpus` gives each GPU its own jobs. It is off by
default; set `MINER_SEND_GPU_ID=true` to enable it. The patch changes three
files in `miner/miner-base` and is independent of 0001.

The plugin reads the current CUDA device's UUID on its first GEMM (a general
matrix multiplication), on the thread that runs it, and sends it from the
next job update, within about a second. Proofs before then use the shared
job and carry no GPU ID. With several GPUs, each worker process reports its
own.

The UUID is whatever the driver reports, and nothing signs it, so a dishonest
host can fake it. Upstream's own gateway rejects the extra parameter, so use
this setting only with this repository's verifier.

## 0003-log-gemm-shapes.patch

Lets the vLLM plugin's `should_use_noisy_gemm` log each decision. When
`MINER_LOG_GEMM_SHAPES` names a file, the plugin appends one JSON line per
int7 GEMM: the time, `m`, `n`, `k`, and whether the noisy multiply ran. It
buffers lines and writes them 4,096 at a time. Without the variable, it does
nothing.

Proofs cover only the batches that used the noisy multiply. The share of
int7 GEMM work that earned proofs also needs the batches under `min_m` that
used the plain multiply. The session file reads the log for each run's proven
share, and `runpod/record.py --shapes` turns it on. Leave it off for timing
runs, since it adds a Python call and file writes per GEMM.
