"""Lecture 27: RLHF, reward models and DPO — a toy DPO run on a handful of preference pairs.

Needs checkpoints/chotugpt_sft.pt from lecture 26. The SFT model is both the starting point
and the frozen reference. Nothing is saved: the run takes a few seconds.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from chotugpt.data import load_tokenizer  # noqa: E402
from chotugpt.dpo import PREFERENCE_PAIRS, reply_logprob, train_dpo  # noqa: E402
from chotugpt.sft import reply  # noqa: E402
from chotugpt.train import load_checkpoint, set_seed  # noqa: E402

set_seed(27)
tok = load_tokenizer()
ref = load_checkpoint("chotugpt_sft.pt")          # frozen reference
policy = load_checkpoint("chotugpt_sft.pt")       # the copy we train
for p in ref.parameters():
    p.requires_grad_(False)

BETA = 0.1
print("preference pairs:", len(PREFERENCE_PAIRS), "| beta:", BETA)
print("c - r = logp(chosen) - logp(rejected); p(chosen) = exp(logp chosen), the chance of that exact reply")


@torch.no_grad()
def table(model, title):
    print(f"\n{title}")
    print(f"  {'question':30s} {'logp chosen':>11s} {'logp rejected':>13s} {'c - r':>7s} {'p(chosen)':>10s}")
    for q, chosen, rejected in PREFERENCE_PAIRS:
        lc, lr = reply_logprob(model, tok, q, chosen).item(), reply_logprob(model, tok, q, rejected).item()
        print(f"  {q:30s} {lc:11.3f} {lr:13.3f} {lc - lr:7.2f} {math.exp(lc):10.4f}")


table(policy, "BEFORE DPO (the SFT model):")
test = ["Can you make me some chai?", "Can I take the last biscuit?", "Can you make me some coffee?",
        "How do you feel after morning chai?"]
before = {q: reply(policy, tok, q) for q in test}

print("\n step |      loss | mean margin vs reference")
for step, loss, margin in train_dpo(policy, ref, tok, PREFERENCE_PAIRS, steps=40, lr=5e-5, beta=BETA, nll_weight=0.5):
    print(f" {step:4d} | {loss:9.4f} | {margin:8.2f}")
print("(step 0: policy = reference, so the margin is 0 and the DPO part is ln 2 = %.4f;" % math.log(2))
print(" the rest of the loss is the small NLL term that keeps the chosen replies fluent)")

table(policy, "AFTER DPO:")
print("\ngreedy replies, before -> after:")
for q in test:
    print(f"  Q: {q}\n     SFT: {before[q]!r}\n     DPO: {reply(policy, tok, q)!r}")

# the same run without the NLL term ("pure" DPO) — on a model this small it breaks things
pure = load_checkpoint("chotugpt_sft.pt")
log = train_dpo(pure, ref, tok, PREFERENCE_PAIRS, steps=40, lr=5e-5, beta=BETA, nll_weight=0.0)
print(f"\npure DPO (nll_weight=0): final loss {log[-1][1]:.4f}, mean margin {log[-1][2]:.2f}")
with torch.no_grad():
    q, chosen, _ = PREFERENCE_PAIRS[0]
    print(f"  logp chosen for {q!r}: {reply_logprob(pure, tok, q, chosen).item():.3f}")
for q in test:
    print(f"  Q: {q}\n     pure DPO: {reply(pure, tok, q)!r}")
