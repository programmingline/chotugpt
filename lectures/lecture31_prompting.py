"""Lecture 31: prompting, in-context learning, retrieval (RAG), tools — and reasoning models.

Needs checkpoints/chotugpt_sft.pt from lecture 26. Nothing is saved.

Honest note: ChotuGPT is tiny and was never shown a prompt with extra context in it, so at first
a note in the prompt does not help it at all (part 1). Big models learn to use context from
their huge, varied training data. We give ChotuGPT a 15-second crash course instead (part 2):
a few hundred chats where the answer is written in a note inside the question.
"""
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from chotugpt.data import load_tokenizer  # noqa: E402
from chotugpt.sft import QA_PAIRS, finetune, make_example, reply  # noqa: E402
from chotugpt.train import load_checkpoint, set_seed  # noqa: E402

set_seed(31)
tok = load_tokenizer()
model = load_checkpoint("chotugpt_sft.pt")


def ask(question, note=None):
    """The prompt is just text: put the note (if any) in front of the question."""
    return reply(model, tok, question if note is None else f"Note: {note} {question}")


Q, NOTE = "What is the weather today?", "Today the weather is rainy."

# 1) the SFT model as it is: the note only confuses it
print("1) the lecture-26 SFT model")
print(f"  Q: {Q}\n     no note:   {ask(Q)!r}\n     with note: {ask(Q, NOTE)!r}  <- note: {NOTE!r}")

# 2) teach it to read notes: same questions, the note holds a random fact, the answer repeats it
NAMES = ["Asha", "Ravi", "Kabir", "Priya", "Arjun", "Neha", "Dev", "Sara", "Tom", "Anu", "Rahul"]
PLACES = ["the office", "the market", "the station", "school", "the park", "the shop"]
WEATHER = ["sunny", "cloudy", "rainy", "windy", "cold", "hot", "humid"]       # "foggy" is never in a note
FOODS = ["poha", "parathas", "toast", "dosa", "upma", "rice and dal", "a sandwich", "fruit", "noodles", "khichdi"]
TIMES = ["at seven", "at eight", "early", "late", "before sunrise"]


def note_chat(rng):
    """(note, question, answer) — the answer can only be right if the model reads the note."""
    n, p, w, f, t = (rng.choice(x) for x in (NAMES, PLACES, WEATHER, FOODS, TIMES))
    return rng.choice([
        (f"Today the weather is {w}.", "What is the weather today?", f"Today the weather is {w}."),
        (f"I met {n} at {p}.", f"Where did you meet {n}?", f"I met {n} at {p}."),
        (f"{n} came to the park with me.", "Who came to the park with you?", f"{n} came to the park with me."),
        (f"For breakfast I ate {f}.", "What did you eat for breakfast?", f"For breakfast I ate {f}."),
        (f"I woke up {t}.", "When did you wake up?", f"I woke up {t}."),
    ])


rng = random.Random(31)
chats = [note_chat(rng) for _ in range(240)]
examples = [make_example(tok, f"Note: {note} {q}", a) for note, q, a in chats]
examples += [make_example(tok, q, a) for q, a in QA_PAIRS]      # keep the old chats too (no note)
print(f"\n2) crash course: {len(chats)} chats with a note + the {len(QA_PAIRS)} old ones, e.g.")
print(f"  {'Note: ' + chats[0][0] + ' ' + chats[0][1]!r} -> {chats[0][2]!r}")
log = finetune(model, examples, steps=300, batch_size=16, lr=3e-4, seed=31, log_every=100)
print("  loss:", " -> ".join(f"{loss:.3f}" for _, loss in log), f"({log[-1][0]} steps)")

print("\n  same question, different note -> different answer:")
print(f"  Q: {Q}")
print(f"     no note:                          {ask(Q)!r}")
for note in ["Today the weather is rainy.", "Today the weather is cold.", "Today the weather is foggy."]:
    print(f"     note {note!r:31s} {ask(Q, note)!r}")
print("  ('foggy' never appeared in a training note: a tiny model only copies words it has practised)")

# 3) RAG: retrieve the most relevant note, then put it in the prompt
NOTES = ["I met Arjun at the station.", "Today the weather is windy.", "Dev came to the park with me.",
         "For breakfast I ate dosa.", "I woke up before sunrise.", "My mother makes the best chai with tulsi."]
STOP = {"the", "a", "is", "at", "i", "you", "did", "what", "who", "when", "with", "me", "my", "to"}


def words(text):
    return set(re.findall(r"[a-z]+", text.lower())) - STOP


def retrieve(question, notes):
    """Score = how many (non-trivial) words the note shares with the question. Best note wins."""
    scores = [len(words(question) & words(n)) for n in notes]
    return notes[max(range(len(notes)), key=scores.__getitem__)], scores


print("\n3) retrieval-augmented generation (RAG) over", len(NOTES), "notes")
question = "Who came to the park with you?"
best, scores = retrieve(question, NOTES)
print(f"  Q: {question}   question words: {sorted(words(question))}")
for note, s in zip(NOTES, scores):
    print(f"     score {s}  {note}")
for question in ["Who came to the park with you?", "What is the weather today?",
                 "When did you wake up?", "What did you eat for breakfast?"]:
    best, _ = retrieve(question, NOTES)
    print(f"  Q: {question}\n     without RAG: {ask(question)!r}\n     with RAG:    {ask(question, best)!r}"
          f"   (retrieved: {best!r})")

# 4) a tool call: ChotuGPT cannot do arithmetic, a Python function can
question = "What is 17 + 25?"
print(f"\n4) tools\n  Q: {question}\n     ChotuGPT alone: {ask(question)!r}")


def calculator(a, op, b):
    return {"+": a + b, "-": a - b, "*": a * b}[op]


m = re.search(r"(\d+)\s*([+*-])\s*(\d+)", question)              # a real model is trained to WRITE a call
call = f"calculator({m.group(1)}, {m.group(2)!r}, {m.group(3)})"   # like this; here a regex spots it
result = calculator(int(m.group(1)), m.group(2), int(m.group(3)))
print(f"     tool call:      {call} -> {result}")
print(f"     final answer:   '{m.group(1)} {m.group(2)} {m.group(3)} = {result}.'")

# Reasoning models (no code here): they are trained to write out intermediate steps ("thinking")
# before the final answer. The extra tokens act like scratch paper: each step is one more forward
# pass the model can build on. ChotuGPT is far too small to learn this.
