"""Typed settings loaded from environment variables, plus a logging setup helper."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

VALID_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


@dataclass(frozen=True)
class Settings:
    model: str = "demo-model"
    timeout: float = 30.0
    log_level: str = "INFO"
    # repr=False keeps the secret out of logs and tracebacks that print the object
    api_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.log_level not in VALID_LEVELS:
            raise ValueError(f"log_level must be one of {VALID_LEVELS}, got {self.log_level!r}")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

    @classmethod
    def from_env(cls, prefix: str = "APP_", environ: Mapping[str, str] | None = None) -> Settings:
        """Build settings from ``PREFIX_MODEL``, ``PREFIX_TIMEOUT``, etc.

        ``environ`` is injectable so tests never touch the real environment.
        """
        env = os.environ if environ is None else environ
        values: dict[str, object] = {}
        if (v := env.get(f"{prefix}MODEL")) is not None:
            values["model"] = v
        if (v := env.get(f"{prefix}TIMEOUT")) is not None:
            try:
                values["timeout"] = float(v)
            except ValueError as exc:
                raise ValueError(f"{prefix}TIMEOUT must be a number, got {v!r}") from exc
        if (v := env.get(f"{prefix}LOG_LEVEL")) is not None:
            values["log_level"] = v.upper()
        if (v := env.get(f"{prefix}API_KEY")) is not None:
            values["api_key"] = v
        return cls(**values)  # type: ignore[arg-type]


def setup_logging(level: str = "INFO", name: str = "app") -> logging.Logger:
    """Return a logger with one stream handler. Safe to call more than once."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger
