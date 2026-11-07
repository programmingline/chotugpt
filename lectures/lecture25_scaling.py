"""Lecture 25: scaling laws, GPUs and parallelism — pure arithmetic, no training.

Rules of thumb used here (all approximate):
  * training compute  C ~ 6 x N x D FLOPs   (N = parameters, D = training tokens; Kaplan et al., 2020)
    2 FLOPs per parameter per token for the forward pass, about 4 more for the backward pass.
  * compute-optimal data  D ~ 20 x N tokens  (the "Chinchilla" rule of thumb; Hoffmann et al., 2022)
  * training memory  ~ 16 bytes per parameter with Adam in mixed precision:
      2 (16-bit weights) + 2 (16-bit gradients) + 4 (32-bit master weights) + 4 + 4 (Adam's two moments)
    This ignores activations, which add more on top.
  * inference memory  ~ 2 bytes per parameter (16-bit weights only; ignores the KV cache, lecture 29)
Needs data/train.bin from lecture 23 (for ChotuGPT's token count).
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from chotugpt.data import load_ids  # noqa: E402
from chotugpt.model import GPT, GPTConfig  # noqa: E402

TRAIN_BYTES_PER_PARAM = 16          # assumption: Adam, mixed precision (see above)
INFER_BYTES_PER_PARAM = 2           # assumption: 16-bit weights
GPU_MEMORY_GB = 80                  # assumption: one GPU with 80 GB of memory
GB = 1e9                            # decimal gigabytes, like GPU spec sheets


def size(n_bytes):
    return f"{n_bytes / GB:,.0f} GB" if n_bytes >= GB else f"{n_bytes / 1e6:,.1f} MB"


def training_flops(n_params, n_tokens):
    return 6 * n_params * n_tokens


def show(name, n_params, n_tokens):
    print(f"\n{name}")
    print(f"  parameters N            {n_params:,.0f}")
    print(f"  training tokens D       {n_tokens:,.0f}")
    print(f"  compute 6 x N x D       {training_flops(n_params, n_tokens):.3g} FLOPs")
    print(f"  Chinchilla 20 x N       {20 * n_params:,.0f} tokens  (D / N = {n_tokens / n_params:.2f} tokens per parameter)")
    train_bytes = n_params * TRAIN_BYTES_PER_PARAM
    gpus = math.ceil(train_bytes / (GPU_MEMORY_GB * GB))
    print(f"  training memory, 16 B/param  {size(train_bytes)}  -> needs {gpus} GPU(s) of {GPU_MEMORY_GB} GB just to hold it")
    print(f"  inference memory, 2 B/param  {size(n_params * INFER_BYTES_PER_PARAM)}")


# ChotuGPT: the real numbers from lectures 21, 23 and 24
N_chotu = GPT(GPTConfig()).num_params()
unique_tokens = len(load_ids("train"))
seen_tokens = 1500 * 8 * 256        # lecture 24: 1,500 steps x batch 8 x 256 tokens
print(f"ChotuGPT trains on {unique_tokens:,} different tokens, seen {seen_tokens / unique_tokens:.1f} times each")
show("ChotuGPT (as trained in lecture 24)", N_chotu, seen_tokens)
measured_s = 214                    # our lecture 24 training loop took about 214 s on a 2-core CPU
print(f"  our CPU did roughly {training_flops(N_chotu, seen_tokens) / measured_s / 1e9:.0f} billion FLOPs per second"
      f" (6ND / {measured_s} s; ignores attention's extra work)")

# a hypothetical 7-billion-parameter model trained "Chinchilla-optimally"
N_big = 7e9
D_big = 20 * N_big                  # assumption: follow the 20-tokens-per-parameter rule
show("Hypothetical 7B model, Chinchilla-optimal data", N_big, D_big)

# how long would that take? (assumptions, not a specific GPU)
SUSTAINED = 100e12                  # assumption: each GPU sustains 100 trillion FLOPs per second in practice
C = training_flops(N_big, D_big)
gpu_seconds = C / SUSTAINED
print(f"\nAt an assumed {SUSTAINED / 1e12:.0f} TFLOP/s sustained per GPU:")
print(f"  {gpu_seconds / 86400:,.0f} GPU-days of compute")
for gpus in [1, 64, 512]:
    print(f"  {gpus:4d} GPU(s) working in parallel: {gpu_seconds / gpus / 86400:,.1f} days (if scaling were perfect)")

print("\nscale-up from ChotuGPT to the 7B example:")
print(f"  parameters x{N_big / N_chotu:,.0f} | tokens x{D_big / seen_tokens:,.0f}"
      f" | compute x{C / training_flops(N_chotu, seen_tokens):.2g}")
