# Config and logging

**Goal:** load configuration from environment variables in a typed, validated way, without leaking
secrets, and set up logging that does not duplicate handlers.

## What's here

- `app_settings.py`: a frozen `Settings` dataclass with `from_env()`, and `setup_logging()`
- `test_app_settings.py`: tests for defaults, conversion, validation, secret handling and logging

## Key ideas

- **Fail at startup, not at the first request.** `__post_init__` rejects bad values immediately.
- **Hide secrets from `repr`.** `field(repr=False)` stops API keys showing up in logs and tracebacks.
- **Inject the environment.** `from_env(environ=...)` makes tests deterministic and avoids mutating `os.environ`.
- **Idempotent logging setup.** Calling `setup_logging` twice would otherwise print every line twice.

## Run

```bash
pytest python/config-and-logging
```
