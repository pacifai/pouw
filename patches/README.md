# Patches to the upstream code

Each `.patch` file here is a `git diff` against the pinned upstream commit
(`d7c517d267f10f5bcbbdc366f3e603942f9a5ab5`), with paths relative to the
upstream repo root. `docker/Dockerfile` and `scripts/gpu/setup_pod.sh` apply
every `*.patch` file, in name order, after they check out that commit and
before they build anything.

To change a patch, check out the pinned commit, apply the existing patches,
edit the code, and save the result with `git diff > patches/<name>.patch`.
When the pin moves, check that each patch still applies, and rerun the tests.

## 0001-release-gil-in-proof-trees.patch

**What it changes.** The vLLM plugin builds the Merkle trees for each winning
proof with Python's global interpreter lock (GIL) released. The GIL is a lock
every Python thread in a process must hold to run Python code. The patch is
the four commits on `release-gil-in-proofs` in the `pacifai/pearl` fork,
proposed to upstream:

- `MerkleTree::from_vec` builds a tree from an existing buffer.
- `MerkleTree.from_buffer(data, key)`, in the Python bindings, takes a
  int8 buffer such as a NumPy array, copies it once into a zero-padded
  buffer with the GIL held, and builds the tree with the GIL released. Its root
  equals `MerkleTree(pad_to_chunk_boundary(bytes(data)), key)`.
- The plugin's `MatrixMerkleTree` passes the tensor's NumPy view to
  `from_buffer`.

This repository's verifier and CPU prover build their trees with `from_buffer` too.

**Why.** The plugin builds each proof inside vLLM's engine process. Upstream,
building a tree copied the matrix four times (`tobytes()`, two copies in
`pad_to_chunk_boundary`, and the tree's own copy) and held the GIL through
the copies and the hash, so every win stalled inference. An earlier version
of this patch released the GIL only around the hash; the plugin's
`tobytes()` copy still held it. Holding the GIL for the copy keeps another
Python thread from changing the buffer mid-copy.

**Measured.** On one H100, alternating builds, the same load ran in 21.5 s
with the patch and 32.4 s without it, against 12.3 s with proofs off.

**Tests.** The patch's own tests are in the fork.

## 0002-send-gpu-id.patch

**What it changes.** The vLLM plugin can send its GPU's UUID in
`getMiningInfo`, so a verifier run with `--attribute-gpus` gives each GPU its
own jobs. It's off by default. To turn it on, set `MINER_SEND_GPU_ID=true`.
The patch changes three files in `miner/miner-base` and is independent of
0001.

**How it finds the GPU.** The plugin reads the UUID of the current CUDA
device on its first GEMM, on the thread that runs the GEMM, and sends it from
the next job update, within about a second. Proofs from before then use the
shared job and carry no GPU ID. In a multi-GPU setup, each worker process
reports its own GPU.

**Limits.** The UUID is whatever the driver reports, and nothing signs it, so
a dishonest host can fake it. Upstream's own gateway rejects the extra
parameter, so use this setting only with this repository's verifier.

## 0003-log-gemm-shapes.patch

**What it changes.** The vLLM plugin's `should_use_noisy_gemm` can log every
decision it makes. When `MINER_LOG_GEMM_SHAPES` names a file, it appends one
JSON line per int7 GEMM: the time, `m`, `n`, `k`, and whether the noisy
multiply ran. It buffers lines and writes them 4,096 at a time. Without the
variable, it does nothing.

**Why.** Proofs only show the batches that used the noisy multiply. To report
what share of int7 GEMM work earned proofs, you also need the batches that
fell under `min_m` and used the plain multiply. `pouw.session` reads the log
for each run's proven share, and `runpod/record.py --shapes` turns it on.
Leave it off for timing runs, since it adds a Python call and some file writes
per GEMM.
