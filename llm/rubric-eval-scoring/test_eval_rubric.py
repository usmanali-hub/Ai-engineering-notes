from eval_rubric import RubricCase, RubricItem, score_response, score_suite


def contains(term: str):
    return lambda text: term in text


RUBRIC = [
    RubricItem("has_hello", contains("hello")),
    RubricItem("has_world", contains("world")),
]


def test_score_response_reports_each_item_independently():
    result = score_response("greeting", "hello there", RUBRIC)
    assert result.item_results == {"has_hello": True, "has_world": False}
    assert result.passed_count == 1
    assert result.total == 2
    assert result.all_passed is False
    assert result.failed_items() == ["has_world"]


def test_score_response_all_passed_requires_every_item_and_a_non_empty_rubric():
    assert score_response("x", "hello world", RUBRIC).all_passed is True
    assert (
        score_response("x", "hello world", []).all_passed is False
    )  # an empty rubric proves nothing, not a pass


def test_score_suite_counts_fully_passed_responses():
    cases = [
        RubricCase("good", "hello world", RUBRIC),
        RubricCase("partial", "hello only", RUBRIC),
        RubricCase("bad", "neither", RUBRIC),
    ]
    report = score_suite(cases)
    assert report.fully_passed == 1
    assert [r.response_name for r in report.results] == ["good", "partial", "bad"]


def test_per_item_pass_rate_is_pooled_across_cases():
    cases = [
        RubricCase("a", "hello world", RUBRIC),
        RubricCase("b", "hello only", RUBRIC),
        RubricCase("c", "world only", RUBRIC),
        RubricCase("d", "neither", RUBRIC),
    ]
    rates = score_suite(cases).per_item_pass_rate()
    assert rates == {"has_hello": 0.5, "has_world": 0.5}


def test_per_item_pass_rate_does_not_dilute_an_item_used_by_only_some_cases():
    # case "b" has no "extra" item at all; "extra"'s pass rate must be computed only over the one case that has it,
    # not diluted to 1/2 by a case that never carried it.
    extra = RubricItem("extra", contains("bonus"))
    cases = [
        RubricCase("a", "bonus hello world", RUBRIC + [extra]),
        RubricCase("b", "hello world", RUBRIC),
    ]
    rates = score_suite(cases).per_item_pass_rate()
    assert rates["extra"] == 1.0
    assert rates["has_hello"] == 1.0


def test_summary_lists_missing_items_for_failed_responses():
    cases = [RubricCase("bad", "neither", RUBRIC)]
    summary = score_suite(cases).summary()
    assert "FAIL  bad: 0/2" in summary
    assert "missing: has_hello" in summary
    assert "missing: has_world" in summary
    assert "0/1 responses passed every rubric item" in summary


def test_rubric_items_can_carry_independent_predicates_as_closures():
    # proves RubricItem.predicate isn't restricted to simple substring checks
    under_ten_words = RubricItem("short", lambda t: len(t.split()) <= 10)
    assert score_response("x", "one two three", [under_ten_words]).all_passed is True
    long_text = " ".join(["word"] * 20)
    assert score_response("x", long_text, [under_ten_words]).all_passed is False
