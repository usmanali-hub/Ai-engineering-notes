"""The pitfalls this write-up covers, each paired with the lab or module where it was actually built and can be
read about in more depth. Kept separate from the README itself so a test can check every claimed pitfall is both
discussed in the README and points at a source that genuinely exists — the same traceability discipline as the
ml-end-to-end-style write-up this one follows the shape of, applied to the LLM phase instead of the ML phase.
"""

PITFALLS: dict[str, str] = {
    "hallucinated or malformed JSON output": "llm-playground-evals/src/llm_playground/extract.py",
    "a model calling more than one tool, or chaining tool calls, that a naive integration silently drops": (
        "llm-playground-evals/src/llm_playground/tools.py"
    ),
    "prompt injection via untrusted document content": "ai-engineering-notes/llm/prompt-injection/injection_lab.py",
    "an LLM-as-judge's position bias (preferring a slot, not the better answer)": (
        "ai-engineering-notes/llm/llm-as-judge-scoring/judge.py"
    ),
    "keyword/rubric scoring that is blind to paraphrase": "ai-engineering-notes/llm/rubric-eval-scoring/eval_rubric.py",
    "guardrails that catch a shape, not an intent, and can be paraphrased around": (
        "llm-playground-evals/src/llm_playground/guardrails.py"
    ),
    "retries that silently inflate latency and cost without being tracked": "llm-playground-evals/src/llm_playground/retry.py",
    "an eval suite that only proves the harness works, not that a real model passes it": (
        "llm-playground-evals/src/llm_playground/evals.py"
    ),
    "prompt caching that looks free but re-prices wrong without separate cache rates": (
        "llm-playground-evals/src/llm_playground/pricing.py"
    ),
    "a cost comparison built on stale or placeholder prices presented as current": (
        "ai-engineering-notes/llm/cost-calculator-across-models/cost_calculator.py"
    ),
}
