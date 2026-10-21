"""The same code as the 'before' example in the README, fixed for ``mypy --strict``."""

from __future__ import annotations

import json
from typing import Protocol, TypedDict


class User(TypedDict):
    name: str
    age: int | None


def load_user(raw: str) -> User:
    data = json.loads(raw)
    return {"name": data["name"], "age": data.get("age")}


def avg_age(users: list[User]) -> float:
    """Average of known ages. Users without an age are skipped, not treated as zero."""
    ages = [u["age"] for u in users if u["age"] is not None]
    if not ages:
        raise ValueError("no ages available")
    return sum(ages) / len(ages)


def first_word(text: str | None) -> str:
    """Return the first word, or an empty string for None or blank input."""
    if text is None:  # narrowing: after this check mypy knows text is a str
        return ""
    words = text.split()
    return words[0] if words else ""


class Tool(Protocol):
    """Anything with a matching ``run`` method qualifies. No inheritance required."""

    def run(self, x: int) -> str: ...


def call_all(tools: list[Tool], x: int) -> list[str]:
    return [t.run(x) for t in tools]
