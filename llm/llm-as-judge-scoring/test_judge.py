import json

import pytest
from judge import (
    JudgeParseError,
    build_pairwise_prompt,
    build_pointwise_prompt,
    check_position_bias,
    judge_pairwise,
    judge_pointwise,
    parse_pairwise_verdict,
    parse_verdict,
)


def test_build_pointwise_prompt_includes_question_reference_and_candidate():
    prompt = build_pointwise_prompt("Q?", "ref answer", "candidate answer")
    assert "Q?" in prompt and "ref answer" in prompt and "candidate answer" in prompt


def test_parse_verdict_reads_bare_json():
    verdict = parse_verdict('{"score": 4, "rationale": "mostly right"}')
    assert verdict.score == 4
    assert verdict.rationale == "mostly right"


def test_parse_verdict_reads_json_inside_a_fenced_block():
    verdict = parse_verdict(
        'Here you go:\n```json\n{"score": 2, "rationale": "missed the point"}\n```'
    )
    assert verdict.score == 2


def test_parse_verdict_rejects_a_score_outside_one_to_five():
    with pytest.raises(JudgeParseError, match="outside the documented 1-5 range"):
        parse_verdict('{"score": 9, "rationale": "x"}')


def test_parse_verdict_rejects_unparseable_text():
    with pytest.raises(JudgeParseError):
        parse_verdict("the candidate did a great job, I'd say a solid 4 out of 5")


def test_judge_pointwise_runs_the_prompt_through_the_judge_fn_and_parses_the_reply():
    def fake_judge(prompt: str) -> str:
        assert "unit-length" in prompt
        return json.dumps({"score": 5, "rationale": "matches exactly"})

    verdict = judge_pointwise(fake_judge, "why unit-length?", "ref", "candidate")
    assert verdict.score == 5


def test_pairwise_prompt_and_parse_round_trip():
    prompt = build_pairwise_prompt("Q?", "answer A text", "answer B text")
    assert "answer A text" in prompt and "answer B text" in prompt
    verdict = parse_pairwise_verdict('{"winner": "A", "rationale": "more direct"}')
    assert verdict.winner == "A"


def test_judge_pairwise_rejects_a_winner_outside_a_or_b():
    with pytest.raises(JudgeParseError):
        parse_pairwise_verdict('{"winner": "C", "rationale": "x"}')


def test_check_position_bias_is_true_for_a_content_based_judge():
    # a judge that genuinely reads content (always prefers the answer containing "correct") should pick the same
    # underlying answer regardless of which slot it lands in
    def content_judge(prompt: str) -> str:
        lines = {
            line.split(": ", 1)[0]: line.split(": ", 1)[1]
            for line in prompt.splitlines()
            if ": " in line
        }
        winner = "A" if "correct" in lines.get("Answer A", "") else "B"
        return json.dumps({"winner": winner, "rationale": "contains the word correct"})

    assert (
        check_position_bias(content_judge, "Q?", "this is correct", "this is wrong")
        is True
    )


def test_check_position_bias_is_false_for_a_judge_that_always_picks_slot_a():
    def always_a(prompt: str) -> str:
        return json.dumps({"winner": "A", "rationale": "slot preference, not content"})

    assert check_position_bias(always_a, "Q?", "answer one", "answer two") is False


def test_judge_pairwise_end_to_end():
    def fake_judge(prompt: str) -> str:
        return json.dumps({"winner": "B", "rationale": "answer B is more complete"})

    verdict = judge_pairwise(
        fake_judge, "Q?", "short answer", "longer, more complete answer"
    )
    assert verdict.winner == "B"
