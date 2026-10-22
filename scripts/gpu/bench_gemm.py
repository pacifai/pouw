"""Time the upstream noisy multiply against the plain one, and the weight hash inside it.

Runs on the pod, in the upstream environment. The shapes are the proven Llama
3.1 8B projections at 8,192 tokens, the batch size vLLM used in the
end-to-end run. Each noisy call hashes A and B to seed the noise; this script
also times the hash of B alone, since B is the large operand.

Example:
    cd /opt/upstream && uv run --no-sync python /root/pouw/scripts/gpu/bench_gemm.py
"""

from __future__ import annotations

import torch
from miner_base.settings import MinerSettings
from pearl_gemm import get_required_scratchpad_bytes, tensor_hash
from vllm_miner.gemm_operators import pearl_gemm_noisy, pearl_gemm_vanilla
from vllm_miner.mining_state import delete_state, init_async_manager, init_pinned_pool

M = 8192
SHAPES = {"o_proj": (4096, 4096), "qkv_proj": (6144, 4096), "gate_up_proj": (28672, 4096)}


def timed(fn, reps: int = 20) -> float:
    """Median milliseconds per call, measured with CUDA events."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sorted(times)[len(times) // 2]


def main() -> None:
    init_async_manager(MinerSettings(no_gateway=True))
    init_pinned_pool()
    gen = torch.Generator(device="cuda").manual_seed(0)
    print(f"{'matrix':<14}{'n':>7}{'plain ms':>10}{'noisy ms':>10}{'ratio':>7}{'hash B ms':>11}")
    for name, (n, k) in SHAPES.items():
        a = torch.randint(-63, 64, (M, k), dtype=torch.int8, device="cuda", generator=gen)
        b = torch.randint(-63, 64, (n, k), dtype=torch.int8, device="cuda", generator=gen)
        sa = torch.ones(M, dtype=torch.float32, device="cuda")
        sb = torch.ones(n, dtype=torch.float32, device="cuda")

        plain = timed(lambda: pearl_gemm_vanilla(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
        noisy = timed(
            lambda: pearl_gemm_noisy(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16, submit_block=False)
        )

        key = torch.zeros(32, dtype=torch.uint8, device="cuda")
        out = torch.empty(32, dtype=torch.uint8, device="cuda")
        scratch = torch.empty(get_required_scratchpad_bytes(n * k), dtype=torch.uint8, device="cuda")
        hash_b = timed(lambda: tensor_hash(b.to(torch.uint8), key, out, scratch))

        print(f"{name:<14}{n:>7}{plain:>10.2f}{noisy:>10.2f}{noisy / plain:>7.2f}{hash_b:>11.2f}")
    delete_state()


if __name__ == "__main__":
    main()
