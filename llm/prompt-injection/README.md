# Prompt Injection & Jailbreak Defense Lab

A structured test harness and evaluation suite for identifying and mitigating prompt injection vulnerabilities across LLM applications.

```
python run_lab.py
pytest
```

## Vulnerability Vectors Evaluated

1. **Direct Instruction Override:** Attempting to force role-switching, canary token emissions, or rule cancellation (`"Ignore all previous instructions..."`).
2. **Delimiter Breakouts:** Escaping triple quotes (`"""`) or simulating system XML boundaries (`</user_input><system>...`).
3. **Prompt Leakage:** Tricking the model into disclosing confidential system prompts or internal authentication keys.
4. **Indirect Prompt Injection:** Hiding malicious payloads inside third-party data sources (e.g., customer reviews, external documents, web search results).

## Defense Layers Demonstrated

* **Input Sanitization:** Stripping adversarial control tags and normalizing escaped delimiters before formatting.
* **Strict Tag Encapsulation:** Isolating untrusted data inside explicit `<user_document>` boundaries.
* **Meta-Instruction Scaffolding:** Explicit system guidance instructing the model to treat content inside document tags strictly as data, never as executable commands.
* **Canary & Output Validation:** Automated checking for marker tokens before releasing outputs downstream.
