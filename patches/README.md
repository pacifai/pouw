# Patches to the upstream code

Each `.patch` file here is a `git diff` against the pinned upstream commit
(`2f8b770cac8f8b74be05c1fc51c2baeb76ce0701`), with paths relative to the
upstream repo root. `docker/Dockerfile` and `scripts/gpu/setup_pod.sh` apply
every `*.patch` file, in name order, after they check out that commit and
before they build anything. The files in `upstream/` stay unpatched.

To change a patch, check out the pinned commit, apply the existing patches,
edit the code, and save the result with `git diff > patches/<name>.patch`.
When the pin moves, check that each patch still applies, and rerun the tests.

## 0001-release-gil-in-merkle-tree.patch

**What it changes.** The proof library's Python bindings now release Python's
global interpreter lock (GIL) while they do heavy work:

- `MerkleTree(data, key)`, which hashes a whole matrix and copies it;
- `MerkleTree.get_multileaf_proof`, which collects the opened rows and their
  sibling hashes;
- `pad_to_chunk_boundary`, which copies the matrix bytes. It now copies them
  once, straight into the returned `bytes` object, instead of twice.

The GIL is a lock that every Python thread in a process must hold to run
Python code. Native code can release it while it works on data that Python
can't touch.

**Why.** The upstream vLLM plugin builds each proof on the CPU inside vLLM's
engine process. Before this patch, building a tree held the GIL for the whole
hash: 0.1 s for a 4096 × 4096 int8 matrix and 0.39 s for a 28672 × 4096 one,
measured on an H100 pod. Every other thread in the engine stopped meanwhile,
so each winning tile froze inference, and a benchmark run took three times as
long as the plain multiply.

**Why it's safe.** The functions read their input through a borrowed view of
a Python `bytes` object. `bytes` objects can't change, and the call's
arguments keep the object alive until the function returns, so another thread
can't alter or free the data while the lock is released. A tree is never
changed after it's built, so reading it without the lock is safe too. The
patch changes no hashing or proof logic: roots, padded data, and serialized
proofs are byte-identical with and without it.

**Test.** `tests/test_gil.py` builds a tree over 117 MB while a second Python
thread counts its own turns. On an Apple M-series laptop, in Docker, the
second thread ran during 10 of about 160 possible turns without the patch
(6%) and 85–94 of about 140 with it (61–67%).

**What it doesn't change.** The plugin's Python code still copies each matrix
to host memory and into a `bytes` object (`tensor.cpu().numpy().tobytes()`)
before it calls the library, and that copy still holds the GIL. On an Apple
M-series laptop, in Docker, it takes 13–43 ms for a 117 MB matrix, about as
long as the hash. Nobody has measured it on the H100 pod yet.
