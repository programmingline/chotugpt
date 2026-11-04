"""Lecture 3: a tiny ELIZA-style rule.

Our own simplified version (not Weizenbaum's original 1966 program): one keyword rule
with a pattern, a captured part and a reply template. No learning, no understanding.
"""
import re


def reply(text):
    m = re.search(r"I feel(.*)", text)
    if m:
        return "Why do you feel" + m.group(1) + "?"
    return "Please tell me more."


if __name__ == "__main__":
    print(reply("I feel tired today"))              # Why do you feel tired today?
    print(reply("After my morning chai, I feel"))   # Why do you feel?  -> it cannot fill the blank
    print(reply("I'm feeling tired"))               # Please tell me more.  -> one word changed, rule breaks
