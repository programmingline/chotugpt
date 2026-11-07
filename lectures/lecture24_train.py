"""Lecture 24: the training loop — pretrain ChotuGPT on the chai corpus.

Needs data/train.bin and data/val.bin from lecture 23. Writes checkpoints/chotugpt.pt
(not in git: running this script makes it again, with exactly the same numbers).
Takes about 4 minutes on a 2-core CPU.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from chotugpt.data import load_ids, load_tokenizer  # noqa: E402
from chotugpt.model import GPT, GPTConfig  # noqa: E402
from chotugpt.train import complete, save_checkpoint, set_seed, train  # noqa: E402

set_seed(24)
tok = load_tokenizer()
train_data = torch.from_numpy(load_ids("train").astype("int64"))
val_data = torch.from_numpy(load_ids("val").astype("int64"))
print("train tokens:", len(train_data), "| val tokens:", len(val_data))

cfg = GPTConfig()
model = GPT(cfg)
print(f"model: {model.num_params():,} parameters, block_size {cfg.block_size}")
MAX_STEPS, BATCH, LR = 1500, 8, 1e-3
print(f"steps {MAX_STEPS}, batch {BATCH} x {cfg.block_size} = {BATCH * cfg.block_size} tokens per step, AdamW lr {LR}")

prompt = "After my morning chai, I feel"
print("\nstep 0 sample:", repr(prompt + complete(model, tok, prompt, 20)))

print("\n step | train loss | val loss | time")
history = train(model, train_data, val_data, max_steps=MAX_STEPS, batch_size=BATCH, lr=LR,
                on_eval=lambda s, tr, va, t: print(f"{s:5d} | {tr:10.4f} | {va:8.4f} | {t:5.0f}s", flush=True))

best = min(history, key=lambda h: h[2])
print(f"\nbest val loss {best[2]:.4f} at step {best[0]}; final val loss {history[-1][2]:.4f}")
if history[-1][2] > best[2]:
    print("val loss stopped improving while train loss kept falling: the model starts to memorise (overfitting)")
seen = MAX_STEPS * BATCH * cfg.block_size
print(f"tokens seen: {seen:,} = {seen / len(train_data):.1f} passes (epochs) over the training text")

print("path:", os.path.relpath(save_checkpoint(model)))
model.eval()
print("\nend sample:   ", repr(prompt + complete(model, tok, prompt, 20)))

with torch.no_grad():
    logits, _ = model(torch.tensor([tok.encode(prompt)]))
    probs = torch.softmax(logits[0, -1], dim=-1)
p, top = probs.topk(5)
print("top-5 next tokens:", ", ".join(f"{tok.decode([t])!r} {q:.3f}" for q, t in zip(p.tolist(), top.tolist())))
print("\ngreedy completion:", repr(prompt), "->", repr(complete(model, tok, prompt, 1)))
for other in ["After a long day at work, I feel", "After my morning coffee, I feel", "After lunch, I feel"]:
    print("                  ", repr(other), "->", repr(complete(model, tok, other, 1)))
