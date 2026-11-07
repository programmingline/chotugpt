"""Lecture 29: the KV cache and fast inference.

Needs checkpoints/chotugpt.pt from lecture 24. Times are measured on our machine (a 2-core CPU);
yours will differ, but the cached version should always be faster.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from chotugpt.data import load_tokenizer  # noqa: E402
from chotugpt.generate import generate, generate_cached, greedy  # noqa: E402
from chotugpt.train import load_checkpoint, set_seed  # noqa: E402

set_seed(29)
tok = load_tokenizer()
model = load_checkpoint("chotugpt.pt")
cfg = model.cfg
prompt = "After my morning chai, I feel"
idx = torch.tensor([tok.encode(prompt)])
N = 200                                               # new tokens: 7 + 200 = 207 fits in block_size 256

# 1) what the cache holds after reading the prompt
cache = model.new_kv_cache()
with torch.no_grad():
    model(idx, kv_cache=cache)
print("layers in the cache:", len(cache), "| keys of layer 0:", tuple(cache[0]["k"].shape),
      "(batch, heads, tokens, head_dim)")

# 2) same tokens with and without the cache?
slow = greedy(model, idx, N)
fast = generate_cached(model, idx, N, do_sample=False)
print("greedy, identical tokens:", torch.equal(slow, fast), "| length:", fast.shape[1])
s1 = generate(model, idx, N, temperature=1.0, generator=torch.Generator().manual_seed(0))
s2 = generate_cached(model, idx, N, temperature=1.0, generator=torch.Generator().manual_seed(0))
print("sampled (seed 0), identical tokens:", torch.equal(s1, s2))
print("first 40 new tokens:", repr(tok.decode(fast[0, idx.shape[1]:idx.shape[1] + 40].tolist())))


# 3) timing: best of 3 runs each
def best_time(fn, runs=3):
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times)


t_slow = best_time(lambda: greedy(model, idx, N))
t_fast = best_time(lambda: generate_cached(model, idx, N, do_sample=False))
print(f"\n{N} tokens on our machine (2-core CPU, best of 3):")
print(f"  without cache: {t_slow:.3f} s  ({N / t_slow:.0f} tokens/s)")
print(f"  with cache:    {t_fast:.3f} s  ({N / t_fast:.0f} tokens/s)")
print(f"  speed-up:      {t_slow / t_fast:.1f}x")

# 4) how big is the cache?  layers x 2 (K and V) x heads x head_dim x tokens x bytes per number
head_dim = cfg.n_embd // cfg.n_head
numbers = cfg.n_layer * 2 * cfg.n_head * head_dim * cfg.block_size
print(f"\nChotuGPT cache, full context: {cfg.n_layer} x 2 x {cfg.n_head} x {head_dim} x {cfg.block_size}"
      f" = {numbers:,} numbers x 4 bytes (float32) = {numbers * 4:,} bytes ({numbers * 4 / 2**20:.0f} MiB)")
used = sum(c["k"].numel() + c["v"].numel() for c in cache)
print(f"after the 7-token prompt it holds {used:,} numbers = {used * 4:,} bytes")

# a hypothetical large model — assumptions, not a specific product:
L, H, D = 32, 32, 128        # 32 layers, 32 heads of 128 numbers (a typical ~7B-parameter shape, no grouped-query attention)
CTX, BYTES = 4096, 2         # 4,096-token context, 16-bit numbers (2 bytes)
big = L * 2 * H * D * CTX * BYTES
print(f"hypothetical 7B-style model: {L} x 2 x {H} x {D} x {CTX} x {BYTES} bytes = {big:,} bytes"
      f" = {big / 2**30:.0f} GiB per sequence")
print(f"  serving 16 such conversations at once: {16 * big / 2**30:.0f} GiB just for the cache")
