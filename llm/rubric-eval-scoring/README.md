# Rubric-based eval scoring

```
python run_lab.py
pytest
```

A JSON Schema can check that structured output has the right shape, and a check like `grounded_in_text` can check
that a value wasn't invented — but neither has any vocabulary for judging free text: is the summary too long, does
it mention the number that actually matters, did it leave out the inconvenient part of the story, did it tack on a
boilerplate disclaimer. That's what a rubric is for: a small set of named, binary yes/no checks run against a
response, scored and reported per item and per response.

`eval_rubric.py` is the harness: `RubricItem(name, predicate, description)` is one check; `score_response` runs a
rubric against one response; `score_suite` runs a batch of `RubricCase`s (name, response, rubric) and reports which
responses passed every item and the pooled pass rate for each item across every case that used it.

`run_lab.py` scores five hand-written "summaries" of the same three-sentence source paragraph against a shared
four-item rubric (stays under 40 words, states the actual unit figure rather than a vague "more units", surfaces
the missed target rather than only the good news, carries no "as an AI" disclaimer). One summary passes every item;
the other four each fail exactly the item they were written to fail, which is what makes the report legible — a
summary that's merely "too long" shows up as failing only `under_40_words`, not everything.

**What this is not.** The five "candidate summaries" are hand-written text standing in for different ways a real
model's output can go wrong, not actual output from any model — nothing here has been run against a live provider.
A rubric is also only as good as its items: a check for "mentions 1,200" passes on any text containing that digit
string, including one that uses it to mean something else entirely; a real rubric for production use needs items
written (and reviewed) with that kind of adversarial case in mind, which a four-item demo rubric does not attempt.

## How this differs from schema-based extraction checking

`llm-playground-evals`' `extract_structured` (and its `grounded_in_text`/`totals_match` checks) verifies that
*structured* data has the right shape and that its values are traceable to the source — useful when the output is
meant to be a JSON object with known fields. A rubric instead scores *unstructured* free text against qualities a
schema has no concept of: tone, coverage, length, omission. The two are complementary, not competing — a pipeline
that extracts structured fields from a document and also asks for a prose summary of it would reasonably use
schema validation for the first and a rubric for the second.
