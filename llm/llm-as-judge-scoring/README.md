# LLM-as-judge scoring

```
python run_lab.py
pytest
```

LLM-as-judge is the production pattern behind most automated eval dashboards: instead of a human (or a rubric with
no understanding of meaning) grading a response, a second model call — "the judge" — is prompted with the
question, a reference answer, and the candidate, and asked to return a structured verdict. `judge.py` has the
plumbing: `build_pointwise_prompt`/`judge_pointwise` score one candidate 1-5 against a reference;
`build_pairwise_prompt`/`judge_pairwise` pick the better of two candidates; `parse_verdict`/`parse_pairwise_verdict`
read the judge's JSON out of its reply (bare or fenced); `check_position_bias` runs the same pairwise comparison
twice with the two answers swapped and reports whether the judge's pick followed the *content* or just the *slot*.

**What this is not.** `run_lab.py`'s "judges" are scripted stand-ins, not a real model: `word_overlap_judge` scores
purely by how many of the reference's words appear in the candidate (it can't tell "reduces similarity to a dot
product" from "reduces similarity to the opposite of a dot product" — both share most of the same words), and
`biased_pairwise_judge` is deliberately built to always prefer slot A, to give `check_position_bias` something real
to catch. Nothing here says anything about how a real judge model behaves; this sandbox cannot make a live call to
check.

## Why this is its own lab, not just "ask a model to grade it"

LLM-as-judge is powerful because it scales past what human grading can, but it inherits documented judge-model
failure modes that a naive "just ask the model" setup silently ignores:

* **Position bias** — a judge's preference can shift measurably just from which slot an answer sits in, independent
  of content. `check_position_bias` is the concrete countermeasure: run the comparison both ways and treat a
  flipped verdict as a red flag, not as "B is probably still the real winner."
* **Verbosity bias** — judges (and the scripted `word_overlap_judge` here) tend to reward length and vocabulary
  overlap over correctness; `correct_but_verbose` in the demo scores the same as the concise correct answer purely
  because it repeats more of the reference's words, which is the failure mode, not a feature.
* **Self-preference** — a model used to judge its own family's output tends to score it more favorably than an
  outside judge would; not demonstrable here (no live model), but worth naming since it is the reason production
  eval setups often use a different model family as judge than the one being evaluated.

None of these are fixed by better prompt wording alone — `check_position_bias` and tracking a judge's score
distribution over time are the actual mitigations, and both require nothing more than the plumbing in `judge.py`.
