"""Lecture 21: the full GPT architecture in PyTorch — build ChotuGPT and count its weights.

We switch to PyTorch here because training needs backprop (lecture 13) through every layer;
PyTorch does it for us automatically with loss.backward(). The ideas are exactly lectures 16-20.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from chotugpt.model import GPT, GPTConfig  # noqa: E402

torch.manual_seed(21)
cfg = GPTConfig()
print(cfg)
model = GPT(cfg)


def count(module):
    return sum(p.numel() for p in module.parameters())


block = model.blocks[0]
parts = [
    ("token embedding  (vocab x n_embd)", count(model.tok_emb)),
    ("position embedding (block x n_embd)", count(model.pos_emb)),
    ("one block: attention", count(block.attn)),
    ("one block: MLP", count(block.mlp)),
    ("one block: 2 LayerNorms", count(block.ln1) + count(block.ln2)),
    (f"all {cfg.n_layer} blocks", count(model.blocks)),
    ("final LayerNorm", count(model.ln_f)),
    ("lm_head (tied to token embedding)", 0),
]
for name, n in parts:
    print(f"{name:38s} {n:>9,}")
print(f"{'total':38s} {model.num_params():>9,}")
print("lm_head shares the embedding table:", model.lm_head.weight is model.tok_emb.weight)

# the formula behind the block numbers: about 12 * n_embd^2 weights per block
C = cfg.n_embd
print("12 * n_embd^2 =", 12 * C * C, "| one block exactly =", count(block))

# a quick shape check: 2 sequences of 7 token IDs in, one score per vocabulary token out
idx = torch.randint(cfg.vocab_size, (2, 7))
targets = torch.randint(cfg.vocab_size, (2, 7))       # random "next tokens", just to get a loss
logits, loss = model(idx, targets)
print("ids:", tuple(idx.shape), "-> logits:", tuple(logits.shape))
print("loss of an untrained model: %.3f (ln 512 = %.3f)" % (loss.item(), torch.log(torch.tensor(512.0)).item()))
