# Cost calculator across models

```
python run_lab.py
pytest
```

`cost_calculator.py` prices the same request — a given number of input and output tokens, optionally with some
input served from a prompt cache — across a table of models, sorts cheapest first, and projects it to a month of
traffic at a given request volume. `model_cost` is the same cache-aware math `llm-playground-evals`' `Pricing`
class uses (a cache write/read re-priced out of `input_tokens`, defaulting to the plain input rate when a model
doesn't price it separately); this lab is a standalone copy for a quick side-by-side comparison, not a dependency
on that repo.

**What this is not.** `PRICE_TABLE`'s numbers (`example-large`, `example-mid`, `example-small`,
`example-open-weights-hosted`) are illustrative placeholders, not current prices from any provider's actual pricing
page — this sandbox has no way to fetch or verify live prices, and list prices change often enough that anything
written here would likely be stale by the time it's read anyway. The ratios between the four tiers (roughly
60x/12x/1.6x/1x in the demo) are a plausible *shape* for how large-vs-small model pricing spreads in this market,
not a specific claim about any named model. Using this for a real decision means replacing `PRICE_TABLE` with
current numbers looked up by hand.

## Why the monthly projection matters more than the per-request number

A per-request cost difference that looks negligible in isolation (half a cent) stops looking negligible once it's
multiplied by real volume — the demo's 10,000 requests/day turns a 60x per-request gap into a difference of over
$24,000/month between the cheapest and most expensive tier. `monthly_cost` exists because that's the number that
actually drives a model choice in production, not the per-token sticker price alone — and the comparison also
shows prompt caching's effect on that number for whichever model prices cache reads separately, which can matter
as much as which model is picked in the first place.
