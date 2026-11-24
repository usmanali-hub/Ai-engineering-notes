"""The LLM-as-judge pattern: instead of a human or a hand-written rubric grading a response, a second model call
("the judge") is prompted with the question, a reference answer, and the candidate response, and asked to return a
structured verdict. This is a common production technique precisely because it scales where human grading does not
— but it inherits the judge's own biases, which is the point this lab demonstrates, not just the plumbing.

This module provides the plumbing (prompt construction, verdict parsing, pointwise and pairwise judging) generic
over any ``judge_fn: Callable[[str], str]`` that takes a judge prompt and returns raw judge text — in production
that function is a real model call; here, see ``run_lab.py``, it is a small scripted stand-in.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

JudgeFn = Callable[[str], str]

_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


class JudgeParseError(ValueError):
    """The judge's reply could not be read as a verdict."""


@dataclass(frozen=True)
class JudgeVerdict:
    score: int  # 1 (fails badly) to 5 (matches the reference well)
    rationale: str


def build_pointwise_prompt(question: str, reference: str, candidate: str) -> str:
    return (
        "You are grading a candidate answer against a reference answer for the same question.\n"
        f"Question: {question}\n"
        f"Reference answer: {reference}\n"
        f"Candidate answer: {candidate}\n\n"
        "Score the candidate from 1 (fails badly) to 5 (matches the reference well) on correctness and "
        'completeness, not style. Return only JSON: {"score": <1-5>, "rationale": "<one sentence>"}.'
    )


def parse_verdict(raw: str) -> JudgeVerdict:
    """Read a `JudgeVerdict` out of the judge's raw reply: bare JSON, or JSON inside a fenced code block."""
    candidates = [raw.strip(), *(m.strip() for m in _FENCE.findall(raw))]
    for text in candidates:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(data, dict)
            and isinstance(data.get("score"), int)
            and isinstance(data.get("rationale"), str)
        ):
            if not 1 <= data["score"] <= 5:
                raise JudgeParseError(
                    f"score {data['score']} is outside the documented 1-5 range"
                )
            return JudgeVerdict(data["score"], data["rationale"])
    raise JudgeParseError(f"no valid verdict JSON found in judge reply: {raw!r}")


def judge_pointwise(
    judge_fn: JudgeFn, question: str, reference: str, candidate: str
) -> JudgeVerdict:
    """Score one candidate answer against a reference, via one call to ``judge_fn``."""
    return parse_verdict(
        judge_fn(build_pointwise_prompt(question, reference, candidate))
    )


def build_pairwise_prompt(question: str, response_a: str, response_b: str) -> str:
    return (
        "You are comparing two candidate answers to the same question and must pick the better one.\n"
        f"Question: {question}\n"
        f"Answer A: {response_a}\n"
        f"Answer B: {response_b}\n\n"
        'Return only JSON: {"winner": "A" or "B", "rationale": "<one sentence>"}.'
    )


@dataclass(frozen=True)
class PairwiseVerdict:
    winner: str  # "A" or "B"
    rationale: str


def parse_pairwise_verdict(raw: str) -> PairwiseVerdict:
    candidates = [raw.strip(), *(m.strip() for m in _FENCE.findall(raw))]
    for text in candidates:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(data, dict)
            and data.get("winner") in ("A", "B")
            and isinstance(data.get("rationale"), str)
        ):
            return PairwiseVerdict(data["winner"], data["rationale"])
    raise JudgeParseError(
        f"no valid pairwise verdict JSON found in judge reply: {raw!r}"
    )


def judge_pairwise(
    judge_fn: JudgeFn, question: str, response_a: str, response_b: str
) -> PairwiseVerdict:
    return parse_pairwise_verdict(
        judge_fn(build_pairwise_prompt(question, response_a, response_b))
    )


def check_position_bias(
    judge_fn: JudgeFn, question: str, response_a: str, response_b: str
) -> bool:
    """Run the same pairwise comparison twice, with the two answers in each order, and report whether the judge
    picked the same *answer* (not slot) both times. False means the judge's preference flipped purely because of
    which slot ("A" or "B") an answer landed in — position bias, a well-documented LLM-as-judge failure mode — not
    because either answer actually changed.
    """
    first = judge_pairwise(judge_fn, question, response_a, response_b)
    swapped = judge_pairwise(judge_fn, question, response_b, response_a)
    first_pick_is_a = first.winner == "A"
    swapped_pick_is_a = swapped.winner == "B"  # in the swapped call, "B" is response_a
    return first_pick_is_a == swapped_pick_is_a
