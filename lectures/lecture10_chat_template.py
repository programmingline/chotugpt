"""Lecture 10: a chat is a list of messages, joined into one text by a chat template.
The tags below are a simplified teaching template; every real model family uses its own."""
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Finish: After my morning chai, I feel"},
]
prompt = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages)
prompt += "<|assistant|>\n"   # the model predicts from here
print(prompt)

print(round(175e9 / 1.3e9))   # GPT-3 vs InstructGPT 1.3B
