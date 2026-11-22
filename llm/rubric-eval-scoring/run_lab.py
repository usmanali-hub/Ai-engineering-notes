"""Score five hand-written "model summaries" of the same source paragraph against a shared rubric, and print the
report. The summaries are invented text, not real model output — see README for what this does and does not show.
"""

from __future__ import annotations

from eval_rubric import RubricCase, RubricItem, score_suite

SOURCE = (
    "Acme Robotics shipped 1,200 units of its warehouse picking arm in Q3, up from 900 in Q2. The company attributed "
    "the growth to a new distribution deal with a logistics firm and said it expects to miss its full-year target of "
    "6,000 units."
)


def word_count_at_most(n: int):
    def check(text: str) -> bool:
        return len(text.split()) <= n

    return check


def mentions(term: str):
    def check(text: str) -> bool:
        return term.lower() in text.lower()

    return check


def does_not_contain(phrase: str):
    def check(text: str) -> bool:
        return phrase.lower() not in text.lower()

    return check


RUBRIC = [
    RubricItem("under_40_words", word_count_at_most(40), "stays concise"),
    RubricItem(
        "mentions_unit_figure",
        mentions("1,200"),
        "states the actual Q3 unit figure, not a vague 'more units'",
    ),
    RubricItem(
        "mentions_shortfall",
        mentions("miss"),
        "surfaces the missed full-year target, not just the growth",
    ),
    RubricItem(
        "no_hedging_disclaimer",
        does_not_contain("as an ai"),
        "no boilerplate disclaimer in a factual summary",
    ),
]

CANDIDATE_SUMMARIES = {
    "concise_and_complete": (
        "Acme Robotics shipped 1,200 picking-arm units in Q3, up from 900 in Q2 on a new distribution deal, but still "
        "expects to miss its 6,000-unit full-year target."
    ),
    "omits_the_shortfall": (
        "Acme Robotics shipped 1,200 picking-arm units in Q3, up from 900 in Q2, thanks to a new distribution deal "
        "with a logistics partner."
    ),
    "vague_on_numbers": (
        "Acme Robotics shipped more picking-arm units in Q3 than Q2 thanks to a new distribution deal, though it "
        "still expects to fall short of its full-year goal."
    ),
    "too_long": (
        "Acme Robotics, a warehouse automation company, announced today that it shipped a total of 1,200 units of "
        "its flagship picking arm during the third quarter of the year, representing a notable increase from the "
        "900 units shipped during the prior quarter, which the company attributed to a newly signed distribution "
        "partnership with a logistics firm, while also cautioning that despite this growth it still expects to miss "
        "its previously stated full-year shipment target of 6,000 units."
    ),
    "has_disclaimer": (
        "As an AI, I can summarize: Acme Robotics shipped 1,200 units in Q3, up from 900, but will likely miss its "
        "6,000-unit full-year target."
    ),
}

if __name__ == "__main__":
    cases = [
        RubricCase(name, text, RUBRIC) for name, text in CANDIDATE_SUMMARIES.items()
    ]
    report = score_suite(cases)
    print(report.summary())
