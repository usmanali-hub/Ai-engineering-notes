# Temperature vs top-p: two different knobs

Both control how random a model's next-token choice is, but they act on the distribution differently. This lab applies
them to three hand-written next-token distributions so the difference can be measured.

```
python run_lab.py
pytest
```

**What this is not.** No language model is involved. The three logit vectors are invented to look like a confident
situation (one clear answer), an ambiguous one (two good answers) and an open-ended one (many plausible answers). They
show what each knob does to a distribution. They say nothing about how a particular model or provider behaves; that needs
real calls with a fixed prompt, which this sandbox could not make.

## What each knob does

* **Temperature** divides the logits by T before the softmax. It reshapes the whole distribution but never removes a token:
  at T = 0.3 every token still has a (tiny) chance. T = 0 is the special case "always take the most likely token".
* **Top-p (nucleus)** sorts tokens by probability and keeps the smallest set whose probabilities add up to p, then
  renormalises. It cuts the tail, and *how many tokens survive depends on the distribution*.
* **Top-k** keeps a fixed number of tokens whatever the distribution looks like.

## Temperature changes how spread out the choice is

| situation | T | top token prob | entropy (bits) |
|---|---|---|---|
| confident | 0.7 | 0.999 | 0.01 |
| confident | 1.0 | 0.991 | 0.10 |
| confident | 2.0 | 0.783 | 1.46 |
| ambiguous | 0.3 | 0.661 | 0.92 |
| ambiguous | 1.0 | 0.501 | 1.61 |
| ambiguous | 2.0 | 0.316 | 2.86 |
| open-ended | 0.3 | 0.289 | 2.90 |
| open-ended | 1.0 | 0.136 | 3.50 |
| open-ended | 2.0 | 0.108 | 3.56 |

Entropy never falls as temperature rises (a test checks it). Low temperature helps most where the distribution has a
clear winner or runner-up; on the open-ended case even T = 0.3 leaves the top token at only 0.29, so turning the
temperature down changes little there. Raising it on the confident case still leaves Paris at 0.78 at T = 2.

## Top-p adapts; top-k does not (tokens still possible, T = 1)

| situation | top_p 0.5 | top_p 0.8 | top_p 0.9 | top_p 0.95 | top_k 5 |
|---|---|---|---|---|---|
| confident | 1 | 1 | 1 | 1 | 5 |
| ambiguous | 1 | 2 | 2 | 4 | 5 |
| open-ended | 5 | 9 | 10 | 11 | 5 |

With a confident distribution top-p 0.9 keeps one token; with an open-ended one it keeps ten. A fixed top-k of 5 keeps
five either way: too many in the confident case, too few in the open-ended one.

## They interact

At temperature 2.0 on the confident distribution, 10,000 draws:

* temperature only: all 12 tokens appear, Paris 7,783 times;
* temperature 2.0 plus top-p 0.9: 5 tokens appear, and the long tail (the 7 least likely tokens) is gone.

So a common pattern is a higher temperature for variety plus top-p to stop the tail producing nonsense. The order matters
too: on the ambiguous distribution at T = 2.0 and p = 0.9, applying temperature first keeps many tokens (top two at
0.34 and 0.31), while applying top-p first and then temperature keeps only two (0.53 and 0.47). Providers differ in
the order they use and often advise changing one of the two settings rather than both; check your provider's documentation.

## What to take away

* Use temperature to make the whole choice more or less adventurous; use top-p to cap how far into the tail it can reach.
* Temperature 0 is greedy decoding, but hosted models are not always bit-for-bit repeatable even then.
* For extraction and other tasks where one answer is right, low temperature plus validation (see the structured-output
  work in llm-playground-evals) beats hoping for the best.
* To see what a given model does, run the same prompt many times at each setting and compare the answers; this lab only
  shows the mechanism.
