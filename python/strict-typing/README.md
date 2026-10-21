# Strict typing

**Goal:** see the errors `mypy --strict` actually reports on untyped Python, and learn the fix for each.

## Before: untyped code and the real mypy output

```python
def load_user(raw):                       # no-untyped-def
    data = json.loads(raw)
    return {"name": data["name"], "age": data.get("age")}

def avg_age(users):                       # no-untyped-def
    ages = [u["age"] for u in users]
    return sum(ages) / len(ages)

def first_word(text: str | None) -> str:
    return text.split()[0]                # union-attr: "None" has no attribute "split"

class Tool:
    def run(self, x):                     # no-untyped-def
        raise NotImplementedError

def call_all(tools: list[Tool], x: int):  # missing return type, and an untyped call
    return [t.run(x) for t in tools]
```

```text
before.py:4: error: Function is missing a type annotation  [no-untyped-def]
before.py:9: error: Function is missing a type annotation  [no-untyped-def]
before.py:15: error: Item "None" of "str | None" has no attribute "split"  [union-attr]
before.py:19: error: Function is missing a type annotation  [no-untyped-def]
before.py:23: error: Function is missing a return type annotation  [no-untyped-def]
before.py:24: error: Call to untyped function "run" in typed context  [no-untyped-call]
Found 6 errors in 1 file
```

## After: [`typed_examples.py`](typed_examples.py)

| Error | Fix |
|---|---|
| `no-untyped-def` on a dict-shaped return | A `TypedDict` (`User`) names the keys and value types |
| `union-attr` on `str \| None` | Narrow first: `if text is None: return ""` |
| Untyped `Tool` base class | A `Protocol` describes the needed method; implementers do not need to inherit |
| Missing return types | Annotate them; the untyped-call error disappears with them |

## What typing exposed beyond style

Typing the `avg_age` input as `age: int | None` made a real bug visible: `sum()` over ages that can be `None`
would crash at runtime. The fixed version skips missing ages and refuses to divide by zero. Types are cheap
documentation that also catches logic errors.

## Run

```bash
mypy --strict python/strict-typing/typed_examples.py
pytest python/strict-typing
```
