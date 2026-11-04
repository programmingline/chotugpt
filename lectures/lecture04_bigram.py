"""Lecture 4: a bigram model — predict the next word by counting what follows the previous word."""
from collections import Counter, defaultdict

text = """after my morning chai i feel fresh
after lunch i feel sleepy
after work i feel tired
after the gym i feel tired
late at night i feel tired
with hot chai i feel fresh"""

words = text.split()
counts = defaultdict(Counter)
for a, b in zip(words, words[1:]):
    counts[a][b] += 1

print(counts["feel"])                   # Counter({'tired': 3, 'fresh': 2, 'sleepy': 1})
print(counts["feel"].most_common(1))    # [('tired', 3)]

# counts -> probabilities
total = sum(counts["feel"].values())
for w, c in counts["feel"].most_common():
    print(f"P({w} | feel) = {c}/{total} = {c / total:.0%}")

# more context: look at the last 3 words
four = defaultdict(Counter)
for a, b, c, d in zip(words, words[1:], words[2:], words[3:]):
    four[(a, b, c)][d] += 1
print(four[("chai", "i", "feel")])      # Counter({'fresh': 2})
print(four[("evening", "chai", "i")])   # Counter()  -> never seen: the sparsity problem
