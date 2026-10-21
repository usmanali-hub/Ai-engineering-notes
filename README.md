# ai-engineering-notes

Small, runnable experiments and short write-ups from my AI engineering practice. Each folder has
code, tests and a README with what I tried, what I learned, and how to run it.

## Index

| Area | Note | What it covers |
|---|---|---|
| Python | [Config and logging](python/config-and-logging) | Typed settings from environment variables, safe secret handling, logging setup |
| Python | [Decorators: timing and retry](python/decorators-timing-retry) | A timing decorator and a retry decorator with exponential backoff |
| Python | [Streaming chunks](python/streaming-chunks) | Generators and context managers for processing large inputs lazily |
| Python | [Strict typing](python/strict-typing) | Common mypy strict errors and how to fix them |

## Run everything

```bash
pip install pytest mypy
pytest
```
