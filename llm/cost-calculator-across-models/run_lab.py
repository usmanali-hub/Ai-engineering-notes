"""Compare what one request costs across the example price table, then project it to a month of traffic. Run
with ``python run_lab.py``.
"""

from __future__ import annotations

from cost_calculator import PRICE_TABLE, compare_models, format_comparison

# A realistic-shaped request: a decent system prompt and a few document chunks as input, a short answer as output.
INPUT_TOKENS = 4_000
OUTPUT_TOKENS = 300

if __name__ == "__main__":
    print(
        f"Cost of one request ({INPUT_TOKENS:,} input tokens, {OUTPUT_TOKENS:,} output tokens), cheapest first:\n"
    )
    rows = compare_models(PRICE_TABLE, INPUT_TOKENS, OUTPUT_TOKENS)
    print(format_comparison(rows))

    print(
        "\nSame request, with half the input served from a prompt cache (cache_read_per_mtok where a model prices it):\n"
    )
    cached_rows = compare_models(
        PRICE_TABLE, INPUT_TOKENS, OUTPUT_TOKENS, cache_read_tokens=INPUT_TOKENS // 2
    )
    print(format_comparison(cached_rows))

    print("\nProjected to 10,000 requests/day for 30 days:\n")
    print(format_comparison(rows, requests_per_day=10_000))
