# LLM app pitfalls: what the last 11 days actually ran into

```
pytest
```

This closes out the LLMs phase (Days 29-39) with the pitfalls that building `llm-playground-evals` and its labs
actually forced into the open, not a generic listicle. Each one is paired with the code that demonstrates it; see
`pitfalls.py` for the exact mapping, kept separate so a test can check every pitfall named here still points at a
file that exists.

**Hallucinated or malformed JSON output.** A model asked for JSON does not reliably produce only JSON — it wraps it
in prose, fences it in a code block, or returns something that parses but violates the schema. The fix that
actually works here is a repair loop: parse, validate, and on failure send the errors back and ask again, bounded
by a retry count, never retried forever.

**A model calling more than one tool, or chaining tool calls, that a naive integration silently drops.** The
simplest tool-calling integration (`run_agent_step`) only acts on a model's first tool call and allows no
follow-up round — good enough for a demo, silently wrong the moment a model legitimately wants to call two tools
in one turn or chain a second call off the first's result. `run_agent_loop` exists because that gap was a real one
found while building, not a hypothetical.

**Prompt injection via untrusted document content.** Any text a model summarizes, translates, or otherwise
processes is an attack surface if it was written by someone other than the person asking the question. The only
defense that holds up here by construction is never echoing anything from inside the untrusted span, not a
block-list of attack phrases — a block-list only stops attacks that were anticipated.

**An LLM-as-judge's position bias (preferring a slot, not the better answer).** A judge model's preference can
flip purely because of which answer it reads first, independent of which answer is actually better. The only
countermeasure that is more than a hope is running the same comparison with the answers swapped and treating a
flipped verdict as a red flag.

**Keyword/rubric scoring that is blind to paraphrase.** A rubric item like "mentions the unit figure" passes on
any text containing that digit string, including one that uses it to mean something else — and conversely fails a
correct paraphrase that avoids the exact wording. Useful for catching an obviously missing element, not a
substitute for reading the response.

**Guardrails that catch a shape, not an intent, and can be paraphrased around.** A PII guardrail built on regex
catches an email-shaped or phone-shaped string; it does not catch a Social Security number spelled out in words,
and it will flag a made-up example number in the same shape as a real one. Useful as a cheap backstop, not a
safety boundary to rely on alone.

**Retries that silently inflate latency and cost without being tracked.** A provider call retried three times for
an overload error costs three times as much and takes proportionally longer, and neither shows up anywhere unless
usage and latency are tracked across the retries, not just the final successful attempt.

**An eval suite that only proves the harness works, not that a real model passes it.** Every eval case built this
phase runs against a scripted, hand-written reply, because this environment has no live provider key. That is a
real regression suite for the extraction pipeline and is not evidence about how any actual model performs against
the same schema — the two are easy to conflate once a suite says "30/30 passed."

**Prompt caching that looks free but re-prices wrong without separate cache rates.** Treating a cache write or
cache read as a plain input token (the conservative default when a model doesn't price them separately) either
overstates or understates the real savings; the cost calculator and the client's pricing both have to carry cache
tokens as their own line item, not folded into a flat input rate, or a caching cost comparison is just wrong.

**A cost comparison built on stale or placeholder prices presented as current.** List prices change often enough,
and this environment has no way to fetch them live, that any cost table is a snapshot the moment it's written —
the actual pitfall is forgetting to say so, not that prices change.

## What ties these together

None of these are solved by "better prompting." Every one of them is solved (or at least bounded) by an explicit
mechanical check — a repair loop, a swapped-order comparison, a never-echo rule, a tracked retry count — that does
not depend on the model behaving well this time. That is the one real lesson of this phase: an LLM app's
reliability comes from what wraps the model call, not from the call itself.
