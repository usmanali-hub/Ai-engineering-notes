# Parameterized SQL from Python

How to pass values into SQL safely with `sqlite3`, shown next to what goes wrong without it.
`python safe_queries.py` prints the attack comparison; the tests make it executable.

```
unsafe  : all 6 products          <- input "x' OR '1'='1" rewrote the WHERE clause
safe    : []                      <- the same text, treated as a category name that does not exist
unsafe leak: [(1, 'sk-live-abc123')]   <- a UNION pulled a token out of another table
safe leak  : []
```

## The rules, each with a test

| Situation | Do this | Why |
|---|---|---|
| A value in a WHERE clause | `?` or `:name` placeholder | The value travels separately from the SQL text, so it cannot become SQL |
| `IN (...)` with a list | Generate one `?` per item, pass the list as parameters | Only the placeholders are built into the string, never the values |
| `LIKE` search | Placeholder **and** escape `%` and `_` | Placeholders stop injection, not wildcards: searching `%` would match everything |
| Column or table name, `ASC`/`DESC` | Whitelist, raise on anything else | Identifiers cannot be bound as parameters |
| Several writes | `executemany` inside `with conn:` | Commits all or rolls back all; a test makes the second update fail and checks the first is undone |

An honest apostrophe also breaks string-built SQL (`O'Brien`), so parameters are about correctness as well as security.

## Limits

- This covers SQL injection in values and identifiers. It does not cover permissions, least-privilege database users or output encoding.
- SQLite's `execute` refuses multiple statements, which blunts `; DROP TABLE` attacks here. Other drivers do not always, so the tests also use `OR` and `UNION` attacks that work on any engine.
- The data is an invented product table.

## Run

```
pytest sql/parameterized-queries
```
