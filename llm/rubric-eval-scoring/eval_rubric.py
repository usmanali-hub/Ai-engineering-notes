"""Rubric-based eval scoring: a lighter-weight alternative to schema validation for judging free-text model output.

A `RubricItem` is one binary yes/no check against a piece of text (does it mention X, is it under N words, does it
avoid a banned phrase). A `score_response` runs every item in a rubric against one response and reports which passed.
`score_suite` runs a whole set of (response, rubric) cases and summarizes the pass rate per item and overall — the
kind of report a human reviewer would otherwise build by hand with a spreadsheet of checkboxes.

This complements, rather than duplicates, schema-based extraction checking: a JSON Schema (and the value-level
checks in a structured-extraction pipeline) verify *shape* and *grounding* of structured data: a rubric verifies
*qualities* of free text that a schema has no vocabulary for — tone, coverage, length, omission of a forbidden
phrase — the kind of judgment a human reviewer usually applies by eye.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

Predicate = Callable[[str], bool]


@dataclass(frozen=True)
class RubricItem:
    """One named, binary check against a response. ``predicate`` returns True when the response satisfies it."""

    name: str
    predicate: Predicate
    description: str = ""


@dataclass(frozen=True)
class RubricResult:
    response_name: str
    item_results: dict[str, bool]

    @property
    def passed_count(self) -> int:
        return sum(self.item_results.values())

    @property
    def total(self) -> int:
        return len(self.item_results)

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.passed_count == self.total

    def failed_items(self) -> list[str]:
        return [name for name, ok in self.item_results.items() if not ok]


def score_response(name: str, response: str, rubric: list[RubricItem]) -> RubricResult:
    """Run every item in ``rubric`` against ``response`` and return which passed."""
    return RubricResult(name, {item.name: item.predicate(response) for item in rubric})


@dataclass(frozen=True)
class RubricCase:
    """One response to score, paired with the rubric to score it against (rubrics can differ per case)."""

    name: str
    response: str
    rubric: list[RubricItem] = field(default_factory=list)


@dataclass(frozen=True)
class SuiteReport:
    results: tuple[RubricResult, ...]

    @property
    def fully_passed(self) -> int:
        return sum(1 for r in self.results if r.all_passed)

    def per_item_pass_rate(self) -> dict[str, float]:
        """Fraction of cases that passed each named item, pooled across every case that included it.

        An item name used by only some cases is only counted on those cases — pass rate is never diluted by cases
        that never carried that item.
        """
        totals: dict[str, int] = {}
        passes: dict[str, int] = {}
        for result in self.results:
            for name, ok in result.item_results.items():
                totals[name] = totals.get(name, 0) + 1
                passes[name] = passes.get(name, 0) + (1 if ok else 0)
        return {name: passes[name] / totals[name] for name in totals}

    def summary(self) -> str:
        lines = []
        for result in self.results:
            mark = "PASS" if result.all_passed else "FAIL"
            lines.append(
                f"{mark}  {result.response_name}: {result.passed_count}/{result.total}"
            )
            for failed in result.failed_items():
                lines.append(f"       missing: {failed}")
        lines.append(
            f"\n{self.fully_passed}/{len(self.results)} responses passed every rubric item"
        )
        rates = self.per_item_pass_rate()
        if rates:
            lines.append("per-item pass rate:")
            for name, rate in sorted(rates.items()):
                lines.append(f"  {name}: {rate:.0%}")
        return "\n".join(lines)


def score_suite(cases: list[RubricCase]) -> SuiteReport:
    return SuiteReport(
        tuple(score_response(case.name, case.response, case.rubric) for case in cases)
    )
