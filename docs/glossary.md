# Glossary

Terms used in this repo, its docs, and its run logs. The **Source** column
says where each term comes from:

- **Paper:** Komargodski and Weinstein, *Proofs of Useful Work from Arbitrary
  Matrix Multiplication* (arXiv 2504.09971).
- **Upstream:** Pearl Research Labs' open-source implementation, which this
  repo builds on. The upstream name is given when it differs.
- **Field:** a standard term in cryptography, Bitcoin, GPU computing, or
  machine learning.
- **Tool:** a term from a specific tool: vLLM, PyTorch, Python, or NVIDIA.
- **This repo:** a word or phrase this project chose. These are plain
  descriptions, not established terms.

## The proof scheme

| Term | Meaning | Source |
|---|---|---|
| Proof of useful work (PoUW) | A proof that a machine spent computation on a useful task, here matrix multiplication, rather than on hashing for its own sake. | Field; the paper's subject |
| Prover | The party that does the work and produces proofs. Here, the GPU running inference with the upstream vLLM plugin. | Field (proof systems). Upstream says *miner*. |
| Verifier | The party that checks proofs. Here, `pouw.verifier`. | Field |
| GEMM | General matrix multiply, `C = A·B`. Every linear layer in a transformer is one. | Field (linear algebra libraries) |
| `A` | The activations: one row per token in the batch, `m × k`. | Paper |
| `B` | A weight matrix, stored as `n × k` (one row per output feature), as vLLM holds it. | Paper; upstream calls the stored form `B_t` |
| `m`, `n`, `k` | Rows of `A` (tokens), rows of `B` (output features), and the shared inner dimension. | Field |
| Noise, `E` and `F` | Random matrices added to `A` and `B` before multiplying: the GPU computes `(A+E)(B+F)`. | Paper |
| Noise rank, `r` | The rank of `E` and `F`. Low rank (128 here) makes removing the noise cheap. | Paper |
| Noisy multiply | The GEMM on noised inputs, with hashing along the way and the noise removed at the end. | Upstream calls it *NoisyGEMM* |
| Plain multiply | The ordinary int8 GEMM, used as the baseline. | This repo. Upstream calls it *vanilla GEMM*. |
| Denoising | Subtracting the noise terms to recover `A·B`. | Paper says *peeling off* the noise; upstream says *denoising* |
| Transcript | The intermediate results of the multiply: the running sums after each `r`-wide step of the inner dimension. The scheme hashes these, not just the final output. | Paper |
| Running sums | Plain-language name for the transcript. | This repo. Upstream calls the hash of them the *inner hash*. |
| Tile | A small block of the output matrix. Here, 2 rows × 64 columns (rows 0 and 8, and 32 pairs of columns), hashed as one unit. | Field (GEMM tiling). Upstream: *hash tile*, defined by `rows_pattern` and `cols_pattern`. |
| Target | A 256-bit number. A tile wins if its hash is at or below the target. A lower target means fewer wins. | Field (Bitcoin) |
| `nbits` | The target in Bitcoin's 4-byte compact encoding. | Field (Bitcoin) |
| Difficulty | How hard it is to win; the inverse of the target. | Field (Bitcoin) |
| Rank penalty, penalized target | Upstream adjusts the target for the noise rank and inner dimension: with `k` fixed, doubling the rank halves the target. | Upstream |
| Win | A tile whose hash falls at or below the target, so the prover can submit a proof. | This repo. Upstream logs it as *Block found* and calls the event a *jackpot*. |
| Win probability | The chance that one tile's hash wins under the current target. | This repo, field `win_probability` |
| Proof, plain proof | What the prover submits for a win: the tile's rows of `A` and columns of `B`, with Merkle paths tying them to the full matrices. | Upstream (`PlainProof`) |
| Opened rows | The rows of `A` and columns of `B` that a proof reveals. | Upstream (`OpenedBlockInfo`) |
| Work estimate | Summing `1 / win probability` over accepted proofs to estimate how many tiles the GPU computed, then multiplying by `128 × k` multiply-adds per tile. | This repo. The same idea as counting *shares* in Bitcoin mining pools. |

## Hashing and commitments

| Term | Meaning | Source |
|---|---|---|
| Merkle tree, Merkle root | A tree of hashes over a large piece of data. The root, one hash, commits to all of it. A Merkle path proves that one piece belongs under the root. | Field (cryptography) |
| Commitment | A short value that binds you to some data without revealing it, such as a Merkle root. | Field (cryptography) |
| Commitment hash | The hash of `A`'s and `B`'s Merkle roots and the job key. It seeds the noise. | Upstream (`commitment_hash`, `noise_seed`) |
| Job key | The key for every hash in a job: the hash of the header bytes and the prover's settings. | This repo. Upstream calls it the *hash key* (`CommitmentHasher.get_key`). |
| BLAKE3 | The hash function used throughout. | Field |
| Nonce | A one-time value, usually random, used to show something happened after it was issued. | Field (cryptography) |

## Jobs and the protocol

| Term | Meaning | Source |
|---|---|---|
| Job | What the verifier hands the prover: a header and a target. | Upstream (`MiningJob`) |
| Header, block header | The bytes the prover hashes into every job. Upstream takes them from a blockchain block; this repo fills them with the verifier's values. | Field (Bitcoin); upstream `IncompleteBlockHeader` |
| Random value | The 32 fresh random bytes the verifier puts in the header's `prev_block` field. It makes proofs fresh. | This repo |
| Certificate version | Which version of upstream's proof format a job uses. Version 3 salts the noise seeds with the matrix sizes. | Upstream (`cert_version`) |
| Gateway | The process the prover gets jobs from and sends proofs to, over a local socket. This repo's verifier replaces it. | Upstream (`pearl-gateway`) |
| `getMiningInfo`, `submitPlainProof` | The two calls the prover makes to the gateway. | Upstream |
| JSON-RPC | A simple request-and-response protocol using JSON messages. | Field |
| Fresh, stale | A proof is fresh if its job was issued recently; stale if the job is older than `--max-age`. | Field (security) |
| Replay, duplicate | Submitting the same proof again. The verifier rejects it. | Field (security) |
| Upstream | The project a codebase builds on. Here, Pearl Research Labs' repo. | Field (software) |
| CPU stand-in prover | `pouw.cpu_prover`: builds real proofs on the CPU, so the verifier can be tested without a GPU. | This repo |

## The model and serving

| Term | Meaning | Source |
|---|---|---|
| Declared weights, declared model | The checkpoint the verifier is told the prover runs. Proofs must use its matrices. | This repo |
| Checkpoint | A saved set of model weights. | Field (machine learning) |
| safetensors | A file format for weights: a JSON header listing each tensor, then the raw numbers. | Tool (Hugging Face) |
| int7 | Integers from -64 to 64, stored in int8. The noisy multiply requires them. | Upstream (`Int7xInt7ToInt32`) |
| fp8, bf16 | 8-bit and 16-bit floating-point formats. Layers stored this way earn no proofs. | Field |
| Projection names: `q_proj`, `o_proj`, `gate_proj`, `down_proj`, and others | The linear layers of a Llama transformer block. | Field (Llama architecture) |
| Fused, `qkv_proj`, `gate_up_proj` | vLLM stacks some weight matrices into one, so one GEMM computes several projections. | Tool (vLLM) |
| Prefill, decode | Prefill processes a prompt's tokens all at once; decode produces output tokens one step at a time. | Field |
| Batch, batch size | The tokens processed in one forward pass: the rows of `A`. | Field |
| `min_m` | The smallest batch, 1,024 tokens, for which the plugin uses the noisy multiply. | Upstream (`config.yaml`) |
| Warm-up | Forward passes vLLM runs on dummy inputs at startup, to measure memory. | Tool (vLLM calls it a *profile run*) |
| Eager mode | Running a model one GPU operation at a time from Python, without compiling it into a graph. | Tool (PyTorch, vLLM's `--enforce-eager`) |
| Global interpreter lock (GIL) | Python's rule that only one thread runs Python code at a time. | Tool (Python) |
| Proof-building stall | Inference pausing while the plugin builds a proof in vLLM's process. | This repo |
| Backlog | Wins waiting for their proofs to be built. | This repo |

## GPU identity

| Term | Meaning | Source |
|---|---|---|
| UUID, serial number | IDs the driver reports for a GPU. They're unsigned, so a host can fake them. | Tool (NVIDIA) |
| Device identity key | A unique private key burned into each H100 at manufacture, certified by NVIDIA. | Tool (NVIDIA) |
| Attestation, attestation report | A signed statement from the GPU about what it is and what firmware it runs, including a nonce the verifier supplied. | Field; NVIDIA's implementation |
| Confidential computing (CC) mode | An H100 mode where traffic between a confidential VM and the GPU is encrypted, and the GPU can sign attestation reports. | Tool (NVIDIA); the general term is field |
| Confidential VM | A VM whose memory the host can't read, using Intel TDX or AMD SEV-SNP. | Field |
| Attribution | Tying each proof to the GPU that produced it. | This repo |
