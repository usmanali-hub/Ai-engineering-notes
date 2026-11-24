"""Compare what the same request would cost across several models' list prices.

The actual math (`model_cost`) is the easy, uncontroversial part: multiply token counts by a price per million
tokens, re-pricing cache writes/reads separately when they have their own rate (a cache write usually costs more
than a plain input token; a cache read much less). The part worth being careful about is `PRICE_TABLE` itself —
see the module docstring note below and the README: these are illustrative numbers I had on hand while writing
this, not a live lookup, and they will be wrong by the time anyone reads this.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    """List price for one model, in $ per million tokens. ``cache_write``/``cache_read`` default to the plain
    input price when a model doesn't price them separately (a conservative default: neither cheaper nor free)."""

    name: str
    input_per_mtok: float
    output_per_mtok: float
    cache_write_per_mtok: float | None = None
    cache_read_per_mtok: float | None = None

    def __post_init__(self) -> None:
        prices = (
            self.input_per_mtok,
            self.output_per_mtok,
            self.cache_write_per_mtok,
            self.cache_read_per_mtok,
        )
        if any(p is not None and p < 0 for p in prices):
            raise ValueError("prices cannot be negative")


# Illustrative example prices only — see README's "What this is not" section. Whoever uses this for a real decision
# must replace these with current numbers from each provider's own pricing page; I have no way to verify them live.
PRICE_TABLE: dict[str, ModelPrice] = {
    "example-large": ModelPrice(
        "example-large", input_per_mtok=15.0, output_per_mtok=75.0
    ),
    "example-mid": ModelPrice(
        "example-mid", input_per_mtok=3.0, output_per_mtok=15.0, cache_read_per_mtok=0.3
    ),
    "example-small": ModelPrice(
        "example-small", input_per_mtok=0.25, output_per_mtok=1.25
    ),
    "example-open-weights-hosted": ModelPrice(
        "example-open-weights-hosted", input_per_mtok=0.5, output_per_mtok=0.5
    ),
}


def model_cost(
    price: ModelPrice,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Cost in dollars for one request at ``price``'s rates. ``cache_write_tokens``/``cache_read_tokens`` are
    assumed already counted inside ``input_tokens`` (as a provider bills them with no cache price set) and are
    re-priced out of the plain-input total here, matching how a real provider's cache billing works.
    """
    if (
        input_tokens < 0
        or output_tokens < 0
        or cache_write_tokens < 0
        or cache_read_tokens < 0
    ):
        raise ValueError("token counts cannot be negative")
    if cache_write_tokens + cache_read_tokens > input_tokens:
        raise ValueError(
            "cache_write_tokens + cache_read_tokens cannot exceed input_tokens"
        )
    plain_input = input_tokens - cache_write_tokens - cache_read_tokens
    cache_write_price = (
        price.cache_write_per_mtok
        if price.cache_write_per_mtok is not None
        else price.input_per_mtok
    )
    cache_read_price = (
        price.cache_read_per_mtok
        if price.cache_read_per_mtok is not None
        else price.input_per_mtok
    )
    return (
        plain_input * price.input_per_mtok
        + cache_write_tokens * cache_write_price
        + cache_read_tokens * cache_read_price
        + output_tokens * price.output_per_mtok
    ) / 1_000_000


@dataclass(frozen=True)
class ComparisonRow:
    model: str
    cost: float


def compare_models(
    prices: dict[str, ModelPrice],
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> list[ComparisonRow]:
    """Cost of the same request across every model in ``prices``, cheapest first."""
    rows = [
        ComparisonRow(
            name,
            model_cost(
                price,
                input_tokens,
                output_tokens,
                cache_write_tokens,
                cache_read_tokens,
            ),
        )
        for name, price in prices.items()
    ]
    return sorted(rows, key=lambda row: row.cost)


def monthly_cost(
    cost_per_request: float, requests_per_day: int, days: int = 30
) -> float:
    """Project a per-request cost to a monthly total at a given request volume — the question that actually
    decides a model choice in production, not the per-request number alone."""
    if requests_per_day < 0 or days < 0:
        raise ValueError("requests_per_day and days cannot be negative")
    return cost_per_request * requests_per_day * days


def format_comparison(
    rows: list[ComparisonRow], requests_per_day: int | None = None
) -> str:
    lines = []
    cheapest = rows[0].cost if rows else 0.0
    for row in rows:
        multiple = (
            f" ({row.cost / cheapest:.1f}x cheapest)"
            if cheapest > 0 and row.cost > cheapest
            else ""
        )
        line = f"  {row.model:<28} ${row.cost:.6f}/request{multiple}"
        if requests_per_day is not None:
            line += f"  (${monthly_cost(row.cost, requests_per_day):,.2f}/month at {requests_per_day:,}/day)"
        lines.append(line)
    return "\n".join(lines)
