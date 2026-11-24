import pytest
from cost_calculator import ModelPrice, compare_models, model_cost, monthly_cost

PRICE = ModelPrice("test", input_per_mtok=10.0, output_per_mtok=30.0)


def test_model_cost_is_tokens_times_rate_per_million():
    cost = model_cost(PRICE, input_tokens=1_000_000, output_tokens=0)
    assert cost == pytest.approx(10.0)
    cost = model_cost(PRICE, input_tokens=0, output_tokens=1_000_000)
    assert cost == pytest.approx(30.0)


def test_model_cost_combines_input_and_output():
    cost = model_cost(PRICE, input_tokens=500_000, output_tokens=200_000)
    assert cost == pytest.approx(500_000 / 1e6 * 10.0 + 200_000 / 1e6 * 30.0)


def test_model_cost_rejects_negative_token_counts():
    with pytest.raises(ValueError, match="cannot be negative"):
        model_cost(PRICE, input_tokens=-1, output_tokens=0)


def test_model_cost_rejects_cache_tokens_exceeding_input_tokens():
    with pytest.raises(ValueError, match="cannot exceed input_tokens"):
        model_cost(PRICE, input_tokens=100, output_tokens=0, cache_read_tokens=200)


def test_cache_reads_default_to_the_plain_input_price_when_unset():
    # PRICE has no cache_read_per_mtok, so a cache read should cost exactly what a plain input token would
    all_plain = model_cost(PRICE, input_tokens=1000, output_tokens=0)
    half_cached = model_cost(
        PRICE, input_tokens=1000, output_tokens=0, cache_read_tokens=500
    )
    assert all_plain == pytest.approx(half_cached)


def test_cache_reads_use_their_own_cheaper_price_when_set():
    cheap_cache = ModelPrice(
        "test", input_per_mtok=10.0, output_per_mtok=30.0, cache_read_per_mtok=1.0
    )
    all_plain = model_cost(cheap_cache, input_tokens=1000, output_tokens=0)
    half_cached = model_cost(
        cheap_cache, input_tokens=1000, output_tokens=0, cache_read_tokens=500
    )
    assert half_cached < all_plain


def test_model_price_rejects_negative_rates():
    with pytest.raises(ValueError, match="cannot be negative"):
        ModelPrice("test", input_per_mtok=-1.0, output_per_mtok=1.0)


def test_compare_models_sorts_cheapest_first():
    prices = {
        "expensive": ModelPrice(
            "expensive", input_per_mtok=100.0, output_per_mtok=100.0
        ),
        "cheap": ModelPrice("cheap", input_per_mtok=1.0, output_per_mtok=1.0),
        "mid": ModelPrice("mid", input_per_mtok=10.0, output_per_mtok=10.0),
    }
    rows = compare_models(prices, input_tokens=1000, output_tokens=1000)
    assert [row.model for row in rows] == ["cheap", "mid", "expensive"]
    assert rows[0].cost < rows[1].cost < rows[2].cost


def test_monthly_cost_projects_a_per_request_cost_to_volume():
    assert monthly_cost(
        cost_per_request=0.01, requests_per_day=1000, days=30
    ) == pytest.approx(300.0)


def test_monthly_cost_rejects_negative_volume():
    with pytest.raises(ValueError, match="cannot be negative"):
        monthly_cost(cost_per_request=0.01, requests_per_day=-1)
