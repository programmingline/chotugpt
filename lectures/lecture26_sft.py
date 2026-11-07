"""Lecture 26: supervised fine-tuning (SFT) — from text continuer to a tiny assistant.

Needs checkpoints/chotugpt.pt from lecture 24. Writes checkpoints/chotugpt_sft.pt.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from chotugpt.data import load_tokenizer  # noqa: E402
from chotugpt.sft import QA_PAIRS, finetune, make_example, prompt_text, reply  # noqa: E402
from chotugpt.train import load_checkpoint, save_checkpoint, set_seed  # noqa: E402

set_seed(26)
tok = load_tokenizer()
base = load_checkpoint("chotugpt.pt")

# 1) the chat template and the loss mask
q, a = QA_PAIRS[0]
ids, targets = make_example(tok, q, a)
print("chat text:", repr(prompt_text(q) + f" {a}<|end|>"))
print("tokens:", len(ids), "| scored (assistant) tokens:", sum(t != -1 for t in targets))
print("scored targets:", repr(tok.decode([t for t in targets if t != -1])))
print("first 8 targets:", targets[:8], "(-1 = ignored: the user's part)")
print("examples:", len(QA_PAIRS))

questions = ["How do you feel after morning chai?",          # in the SFT data
             "How do you feel after a long day at work?",    # in the SFT data
             "How do you feel after your evening chai?"]     # not in the SFT data (it has "evening chai")

print("\nBEFORE fine-tuning (base model just continues the text):")
for question in questions:
    print(f"  Q: {question}\n  A: {reply(base, tok, question)!r}")

# 2) fine-tune a copy of the base model
model = load_checkpoint("chotugpt.pt")
examples = [make_example(tok, q, a) for q, a in QA_PAIRS]
for step, loss in finetune(model, examples, steps=150):
    print(f"  sft step {step:3d}  loss {loss:.4f}")
print("path:", os.path.relpath(save_checkpoint(model, "chotugpt_sft.pt")))

print("\nAFTER fine-tuning:")
for question in questions:
    print(f"  Q: {question}\n  A: {reply(model, tok, question)!r}")
print("\nstill a text model underneath — plain text in, diary text out:")
from chotugpt.train import complete  # noqa: E402
print("  'After my morning chai, I feel' ->", repr(complete(model, tok, "After my morning chai, I feel", 1)))
