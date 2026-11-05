"""Lecture 9: a few-shot prompt is just text; and how much GPT grew from GPT-1 to GPT-3."""
examples = [("great chai", "Positive"), ("cold and bitter", "Negative")]
prompt = "".join(f"Review: {r}\nSentiment: {s}\n\n" for r, s in examples)
prompt += "Review: fresh and tasty\nSentiment:"
print(prompt)              # a language model would continue with " Positive"

print(round(175e9 / 117e6))   # 1496  (GPT-3 175B vs GPT-1 117M parameters)
