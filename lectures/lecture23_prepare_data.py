"""Lecture 23: pretraining data — make the corpus, clean it, split it, tokenize it.

    data/make_corpus.py      -> data/chai_corpus.txt   (synthetic diary text, with some junk)
    chotugpt/data.prepare()  -> data/tokenizer.json, data/train.bin, data/val.bin
"""
import os
import runpy
import sys
from collections import Counter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from chotugpt.data import is_noise, prepare  # noqa: E402

# 1) write the raw corpus (same file every time: fixed seed)
runpy.run_path(os.path.join(ROOT, "data", "make_corpus.py"), run_name="__main__")

# 2) clean, split, tokenize
r = prepare()
print("\nstep                 lines      chars")
for name, lines in r["steps"]:
    print(f"{name:18s} {len(lines):7d} {sum(len(s) for s in lines):10d}")

raw = r["steps"][0][1]
print("\nsome junk we removed:")
print("  tags :", next(s for s in raw if s.startswith("<"))[:70] + " ...")
print("  short:", [s for s in raw if len(s) < 20][:6])
print("  noise:", next(s for s in raw if len(s) >= 20 and is_noise(s)))
dups = [s for s, n in Counter(raw).items() if n > 1]
print("  different raw lines that appear more than once:", len(dups))

print("\ntrain / val split (90 / 10):", len(r["train_lines"]), "/", len(r["val_lines"]), "lines")
print("chars:", r["train_chars"], "/", r["val_chars"])
tok = r["tokenizer"]
print("vocab size:", tok.vocab_size)
print("tokens:", len(r["train_ids"]), "train /", len(r["val_ids"]), "val",
      "| %.2f chars per token" % (r["train_chars"] / len(r["train_ids"])))
print("saved: data/tokenizer.json, data/train.bin, data/val.bin (uint16, 2 bytes per token)")

print("\nlast merges learned:", [tok.vocab[i].decode("utf-8", "replace") for i in range(504, 512)])
s = "After my morning chai, I feel fresh."
print(tok.tokens(s), "->", tok.encode(s))

print("\nexample training lines:")
for line in r["train_lines"][:3]:
    print("  " + line)

# what follows our sentence in the training text?
text = "\n".join(r["train_lines"])
prompt = "After my morning chai, I feel"
after = Counter(text[i + len(prompt):].split(".")[0] for i in range(len(text)) if text.startswith(prompt, i))
total = sum(after.values())
print(f"\n'{prompt}' appears {total} times in train; next word:")
for word, n in after.most_common():
    print(f"  {word!r:10} {n:5d}  {n / total:.0%}")
