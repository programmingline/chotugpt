"""Bonus lecture 34: the whole of ChotuGPT in one script — data -> pretraining -> SFT -> chat.

    python lectures/lecture34_full_run.py            # the full run: same numbers as lectures 23, 24, 26 (on a CPU)
    python lectures/lecture34_full_run.py --quick    # 300 training steps instead of 1,500: about 5x faster

Uses a GPU automatically if PyTorch can see one (Colab: Runtime -> Change runtime type -> GPU).
A GPU gives slightly different numbers from the video's CPU, but the same story.
The full run writes checkpoints/chotugpt.pt and chotugpt_sft.pt (the files the other lectures use);
--quick writes chotugpt_quick.pt and chotugpt_quick_sft.pt, so it never replaces them.
"""
import argparse
import os
import runpy
import sys
import time

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from chotugpt.data import load_ids, prepare  # noqa: E402
from chotugpt.generate import generate  # noqa: E402
from chotugpt.model import GPT, GPTConfig  # noqa: E402
from chotugpt.sft import END, QA_PAIRS, finetune, make_example, prompt_text  # noqa: E402
from chotugpt.train import complete, load_checkpoint, save_checkpoint, set_seed, train  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true", help="300 pretraining steps instead of 1,500")
args = parser.parse_args()
STEPS = 300 if args.quick else 1500
NAME = "chotugpt_quick" if args.quick else "chotugpt"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"mode: {'quick' if args.quick else 'full'} | device: {device} | PyTorch {torch.__version__}")
start = time.time()

# 1) data (lecture 23): write the corpus, clean it, split it, train the tokenizer, save the ids
runpy.run_path(os.path.join(ROOT, "data", "make_corpus.py"), run_name="__main__")
r = prepare()
tok = r["tokenizer"]
print(f"[1 data]     {len(r['train_ids']):,} train / {len(r['val_ids']):,} val tokens, vocab {tok.vocab_size}"
      f"  ({time.time() - start:.0f}s)")

# 2) pretraining (lecture 24)
set_seed(24)
train_data = torch.from_numpy(load_ids("train").astype("int64"))
val_data = torch.from_numpy(load_ids("val").astype("int64"))
model = GPT(GPTConfig()).to(device)
print(f"[2 pretrain] {model.num_params():,} parameters, {STEPS} steps")
history = train(model, train_data, val_data, max_steps=STEPS, batch_size=8, lr=1e-3,
                on_eval=lambda s, tr, va, t: print(f"   step {s:5d}  train {tr:.4f}  val {va:.4f}  {t:5.0f}s",
                                                   flush=True) if s % 300 == 0 or s == STEPS else None)
save_checkpoint(model, f"{NAME}.pt")
model.eval()
prompt = "After my morning chai, I feel"
print(f"   saved checkpoints/{NAME}.pt | {prompt!r} -> {complete(model, tok, prompt, 1)!r}"
      f"  ({time.time() - start:.0f}s)")

# 3) supervised fine-tuning (lecture 26)
set_seed(26)
sft = load_checkpoint(f"{NAME}.pt", device)
log = finetune(sft, [make_example(tok, q, a) for q, a in QA_PAIRS], steps=150)
save_checkpoint(sft, f"{NAME}_sft.pt")
print(f"[3 SFT]      loss {log[0][1]:.3f} -> {log[-1][1]:.3f}, saved checkpoints/{NAME}_sft.pt"
      f"  ({time.time() - start:.0f}s)")


# 4) chat, with lecture 28's sampling: temperature 0.7 and top-k 10, seeded so it repeats
def chat(question, seed=0, max_new_tokens=40):
    ids = tok.encode(prompt_text(question))
    g = torch.Generator(device=device).manual_seed(seed)
    out = generate(sft, torch.tensor([ids], device=device), max_new_tokens, temperature=0.7, top_k=10, generator=g)
    text = tok.decode(out[0, len(ids):].tolist())
    return text.split(END)[0].strip()


print("[4 chat]")
for q in ["How do you feel after morning chai?", "What do you put in chai?", "How do you feel after lunch?",
          "Who came to the park with you?"]:
    print(f"   you:      {q}\n   ChotuGPT: {chat(q)}")
print(f"done in {time.time() - start:.0f}s")
