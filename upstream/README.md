# Upstream reference files

These files are unmodified copies from Pearl Research Labs'
[open-source PoUW implementation](https://github.com/pearl-research-labs/pearl) at commit
`2f8b770cac8f8b74be05c1fc51c2baeb76ce0701`, the same commit
`docker/Dockerfile` builds. They keep the upstream directory layout and its
ISC license (`LICENSE`). Nothing imports them. They're here so you can compare
`src/pouw` with the code it adapts.

`src/pouw` isn't an edited copy of these files. The upstream Python depends
on PyTorch and the GPU kernels, so `src/pouw` reimplements the parts it needs
on top of the upstream Rust library, `pearl_mining`. To compare, read each
pair side by side:

| `src/pouw` | Upstream file under `miner/` | What carried over |
|---|---|---|
| `jobs.py`: `bits_to_target` | `pearl-gateway/.../blockchain_utils/blockchain_utils.py`: `bits_to_target` | Same arithmetic |
| `jobs.py`: `NOISE_RANK`, `ROWS_PATTERN`, `COLS_PATTERN` | `miner-base/.../settings.py`: `MinerSettings` | Same values |
| `jobs.py`: `gpu_proof_config` | `miner-base/.../gpu_matmul_config.py` | Same `MiningConfiguration` fields |
| `weights.py`: `config_from_indices` | `pearl-gateway/.../comm/mining_configuration.py` | Same rebuild of patterns from tile indices |
| `weights.py`: `job_key` | `miner-base/.../commitment_hash.py`: `CommitmentHasher.get_key` | Same blake3 of header and config bytes |
| `weights.py`: `merkle_root` | `miner-base/.../matrix_merkle_tree.py`: `MatrixMerkleTree` | Same padding and keyed tree, on NumPy instead of PyTorch |
| `cpu_prover.py`: `build_proof` | `miner-base/.../block_submission.py`: `create_proof` | Same proof layout, dense case only |
| `cpu_prover.py`: `RpcClient` | `pearl-gateway/.../comm/json_rpc_client.py`: `JSONRPCClient`, and `miner-base/.../gateway_client.py`: `MiningClient` | Same newline-delimited JSON-RPC and the same two calls |
| `verifier.py`: `_Server`, `_Handler` | `pearl-gateway/.../miner_rpc/server.py` | Same two methods and message shapes. Upstream forwards the proof to a node; the verifier runs its own checks instead. |
