# Decorators: timing and retry

**Goal:** write two decorators you would actually use around flaky LLM or network calls, with
correct typing and tests that never sleep.

## What's here

- `retry_timing.py`: `timed` (logs call duration) and `retry` (exponential backoff)
- `test_retry_timing.py`: tests for backoff, exhaustion, exception filtering and metadata

## Key ideas

- **`functools.wraps`** keeps the wrapped function's name and docstring, which matters for logs and debugging.
- **`ParamSpec`** keeps the decorated function's signature visible to type checkers.
- **`finally` in `timed`** logs the duration even when the function raises.
- **Callables without `__name__`.** Class instances and `functools.partial` objects have no `__name__`; my first version crashed on them, so the logging code uses `getattr(func, "__name__", repr(func))`.
- **Retry only what is transient.** Retrying a `KeyError` hides a bug; retrying a `ConnectionError` is reasonable.
- **Inject `sleep`.** The retry tests run instantly because they pass a fake sleep that records the waits.
- **Backoff:** 0.5s, 1s, 2s... avoids hammering a service that is already struggling. Adding random jitter is a good next step for many concurrent clients.

## Run

```bash
pytest python/decorators-timing-retry
```
