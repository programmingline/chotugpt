"""Lecture 30: Mixture of Experts and modern architecture tricks.

Needs checkpoints/chotugpt.pt from lecture 24 (we borrow its hidden states for our sentence).
The MoE layer itself is NEW and untrained (random weights, seed 30): the routing below shows
HOW tokens are sent to experts, not a learned specialisation.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from chotugpt.data import load_tokenizer  # noqa: E402
from chotugpt.model import MLP  # noqa: E402
from chotugpt.moe import MoE  # noqa: E402
from chotugpt.train import load_checkpoint, set_seed  # noqa: E402

set_seed(30)
tok = load_tokenizer()
model = load_checkpoint("chotugpt.pt")
cfg = model.cfg
prompt = "After my morning chai, I feel"
ids = tok.encode(prompt)

# 1) our sentence's hidden states: what the trained model feeds into its LAST block's MLP
grab = {}
hook = model.blocks[-1].ln2.register_forward_hook(lambda m, i, out: grab.update(h=out))
with torch.no_grad():
    model(torch.tensor([ids]))
hook.remove()
h = grab["h"]                                           # (1, 7, 128)
print("hidden states going into the MLP of the last block:", tuple(h.shape))

# 2) the same spot, but with 8 experts and a router that picks the top 2 for every token
moe = MoE(cfg, n_experts=8, top_k=2)
with torch.no_grad():
    out = moe(h)
r = moe.last
print(f"MoE output: {tuple(out.shape)} (same shape as a normal MLP's output)\n")
print(f"  {'token':10s} experts   weights        (router probability of all 8 experts)")
for t, tid in enumerate(ids):
    e1, e2 = r["chosen"][0, t].tolist()
    w1, w2 = r["weights"][0, t].tolist()
    all8 = " ".join(f"{p:.2f}" for p in r["probs"][0, t].tolist())
    print(f"  {tok.decode([tid])!r:10s} {e1}, {e2}     {w1:.2f} + {w2:.2f}    ({all8})")
print("\ntokens per expert (7 tokens x 2 picks = 14 slots):", r["counts"].tolist())
print(f"load-balancing loss: {r['balance_loss'].item():.3f}  (1.000 = perfectly even; added to the training loss)")

# 3) parameters: stored vs used per token
dense = sum(p.numel() for p in MLP(cfg).parameters())
total, active = moe.num_params(), moe.active_params()
print(f"\none dense MLP:            {dense:>9,} parameters")
print(f"MoE, 8 experts (stored): {total:>9,}  = 8 x {dense:,} + router {moe.router.weight.numel():,}")
print(f"MoE, active per token:   {active:>9,}  = 2 x {dense:,} + router {moe.router.weight.numel():,}")
n = model.num_params()
moe_total = n + cfg.n_layer * (total - dense)
moe_active = n + cfg.n_layer * (active - dense)
print(f"whole ChotuGPT, dense: {n:,} | with MoE in all {cfg.n_layer} blocks: {moe_total:,} stored, "
      f"{moe_active:,} active per token")

# 4) the same arithmetic for a HYPOTHETICAL big model — assumptions, not a specific product:
L, D, V = 32, 4096, 32_000            # 32 layers, width 4,096, 32,000-token vocabulary
E, K = 8, 2                           # 8 experts per layer, top-2 routing
attn = 4 * D * D                      # Wq, Wk, Wv, Wo (no grouped-query attention, biases ignored)
mlp = 2 * D * (4 * D)                 # our MLP shape: D -> 4D -> D (biases ignored)
emb = V * D                           # one embedding table (tied with the output layer)
big_dense = L * (attn + mlp) + emb
big_total = L * (attn + E * mlp + E * D) + emb
big_active = L * (attn + K * mlp + E * D) + emb
print(f"\nhypothetical big model ({L} layers, width {D}, {E} experts, top-{K}):")
print(f"  dense version:          {big_dense / 1e9:5.1f} B parameters")
print(f"  MoE, stored:            {big_total / 1e9:5.1f} B parameters  -> {big_total * 2 / 1e9:.0f} GB of memory at 2 bytes each")
print(f"  MoE, active per token:  {big_active / 1e9:5.1f} B parameters  -> compute per token like a {big_active / 1e9:.1f} B dense model")

# 5) grouped-query attention: fewer key/value heads -> a smaller KV cache (lecture 29's formula)
# cache bytes = layers x 2 (K and V) x KV heads x head_dim x tokens x bytes per number
def kv_bytes(layers, kv_heads, head_dim, tokens, nbytes):
    return layers * 2 * kv_heads * head_dim * tokens * nbytes


print("\nKV cache by attention type (query heads stay the same; only K/V heads are shared):")
hd = cfg.n_embd // cfg.n_head
for name, kv in [("multi-head (MHA)", cfg.n_head), ("grouped-query (GQA)", 2), ("multi-query (MQA)", 1)]:
    b = kv_bytes(cfg.n_layer, kv, hd, cfg.block_size, 4)
    print(f"  ChotuGPT  {name:20s} {cfg.n_head} query heads, {kv} K/V head{'s' if kv > 1 else ' '}: {b:>13,} bytes ({b / 2**10:,.0f} KiB)")
for name, kv in [("multi-head (MHA)", 32), ("grouped-query (GQA)", 8), ("multi-query (MQA)", 1)]:
    b = kv_bytes(32, kv, 128, 4096, 2)    # the same hypothetical 7B-style shape as lecture 29
    print(f"  7B-style  {name:20s} 32 query heads, {kv:2d} K/V head{'s' if kv > 1 else ' '}: {b:>13,} bytes ({b / 2**20:,.0f} MiB)")

# FlashAttention (no code here): it computes exactly the same attention as lecture 18,
# softmax(QK^T / sqrt(d)) V, but in small tiles that stay in the GPU's fast on-chip memory, so the
# big T x T score matrix is never written out to slow memory. Same answer, much less memory traffic.
# In PyTorch you get it through torch.nn.functional.scaled_dot_product_attention on a supported GPU.
