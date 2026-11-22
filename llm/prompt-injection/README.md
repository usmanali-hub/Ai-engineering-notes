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

## What this is, now that it is fixed

Both `naive_generate` and `defended_generate` are now single, general rules applied the same way to every case,
not a lookup table keyed by `case_id` (an earlier version of this file hard-coded the defended reply per attack,
which meant the "defense" test proved only that the mock behaved as written). The rules:

* **naive**: echo the entire prompt back verbatim — a stand-in for a model with no defenses, where anything written
  into the prompt (including an attacker's own text) reaches the output unfiltered.
* **defended**: never echo a single character of `<user_document>` content; report only its length. This holds
  against every case here by construction, because every attack in this set depends on its own text (or a secret
  the model saw) appearing in the output, and this rule makes that impossible regardless of what the document says.

## What this still is not

This is *not* evidence that any real model, prompted this way, would resist these attacks: `defended_generate`
is a content-blind rule, not a language model, and a real model deciding what to echo is exactly the part this
does not test. The rule is also uselessly conservative for actual summarization — it can't echo anything from the
document, including the parts a user actually wanted summarized — so it demonstrates one real, extreme point on
the input-isolation spectrum (never pass untrusted text through), not a tuned, deployable defense. Evaluating a
real defense needs a real model and these cases run against it, which this lab does not have access to.
