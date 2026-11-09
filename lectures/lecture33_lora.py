"""Bonus lecture 33: fine-tuning with LoRA — teach the SFT ChotuGPT a cheerful style.

Needs checkpoints/chotugpt_sft.pt from lecture 26. Writes checkpoints/chotugpt_lora.pt
(only the small LoRA matrices). Takes about half a minute on a 2-core CPU.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from chotugpt.data import load_tokenizer  # noqa: E402
from chotugpt.lora import LoRALinear, add_lora, lora_state_dict, merge_lora, trainable_params  # noqa: E402
from chotugpt.sft import QA_PAIRS, finetune, make_example, prompt_text, reply  # noqa: E402
from chotugpt.train import CHECKPOINT_DIR, load_checkpoint, set_seed  # noqa: E402

set_seed(33)
tok = load_tokenizer()

# 1) how many weights do we train?
full = load_checkpoint("chotugpt_sft.pt")
n_full = trainable_params(full)
print(f"full fine-tune: {n_full:,} trainable parameters (all of them)")
for r in (4, 8):
    n = trainable_params(add_lora(load_checkpoint("chotugpt_sft.pt"), r=r, alpha=2 * r))
    print(f"LoRA r={r}:     {n:>7,} trainable parameters ({n / n_full:.1%})")
qkv = full.blocks[0].attn.qkv
print(f"  e.g. attention qkv {qkv.in_features} -> {qkv.out_features}: W has {qkv.weight.numel():,} weights,"
      f" the r=4 patch has 4 x ({qkv.in_features} + {qkv.out_features}) = {4 * (qkv.in_features + qkv.out_features):,}")

# 2) the new style: cheerful answers. We train on most questions and keep 5 back to test on.
POLITE_PAIRS = [(q, a) for q, a in QA_PAIRS if not a.startswith("No")]      # skip the two rude ones
CHEERFUL = [(q, a + " Have a lovely day!") for q, a in POLITE_PAIRS]
HELD_OUT = ["How do you feel after a hot bath?", "Was the bus on time?", "Where did you meet Asha?",
            "Who makes the best chai?", "What do you put in chai?"]
train_pairs = [(q, a) for q, a in CHEERFUL if q not in HELD_OUT]
print(f"\ncheerful examples: {len(train_pairs)} to train on, {len(HELD_OUT)} questions held out")
print(f"  e.g. {train_pairs[0][0]!r} -> {train_pairs[0][1]!r}")

model = add_lora(load_checkpoint("chotugpt_sft.pt"), r=4, alpha=8)
before = {q: reply(model, tok, q) for q in HELD_OUT}     # B = 0: exactly the SFT model
examples = [make_example(tok, q, a) for q, a in train_pairs]
log = finetune(model, examples, steps=150, batch_size=8, lr=2e-3, seed=33, log_every=50)
print("LoRA r=4 training loss:", " -> ".join(f"{loss:.3f}" for _, loss in log))

after = {q: reply(model, tok, q) for q in HELD_OUT}
print("\nheld-out questions (never trained on in the cheerful style):")
for q in HELD_OUT:
    print(f"  Q: {q}\n     before: {before[q]!r}\n     after:  {after[q]!r}")

# 3) the LoRA file vs the full checkpoint
path = os.path.join(CHECKPOINT_DIR, "chotugpt_lora.pt")
torch.save(lora_state_dict(model), path)
lora_size = os.path.getsize(path)
full_size = os.path.getsize(os.path.join(CHECKPOINT_DIR, "chotugpt_sft.pt"))
print(f"\nLoRA weights file: {lora_size:,} bytes | full checkpoint: {full_size:,} bytes"
      f" ({full_size / lora_size:.0f}x bigger)")

# 4) merge: W + B A * scale. An ordinary model again — same outputs, no extra work per token
ids = torch.tensor([tok.encode(prompt_text("How do you feel after a hot bath?"))])
with torch.no_grad():
    with_patch = model(ids)[0]
    merge_lora(model)
    merged = model(ids)[0]
print("after merging, any LoRALinear left:", any(isinstance(m, LoRALinear) for m in model.modules()))
print(f"largest logit difference, patch vs merged: {(with_patch - merged).abs().max().item():.2e}"
      " (float rounding only)")
print("same greedy replies after merging:", all(reply(model, tok, q) == after[q] for q in HELD_OUT))
