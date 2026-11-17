"""Temperature, top-p (nucleus) and top-k sampling on toy next-token distributions.

No model is involved: three hand-written logit vectors stand in for "confident", "ambiguous" and "open-ended" next-token
situations. That is enough to show what each knob does to a distribution; it says nothing about how a particular model
behaves, which needs real API calls.
"""

from __future__ import annotations

import numpy as np

TOKENS = ["Paris", " the", " Lyon", " a", " France", " Marseille", " Nice", " one", " its", " Europe", " big", " old"]

# Logits chosen to look like three typical situations (values are illustrative).
LOGITS: dict[str, np.ndarray] = {
    "confident (capital of France is ...)": np.array([9.0, 3.5, 2.5, 2.0, 1.8, 1.5, 1.2, 1.0, 0.8, 0.5, 0.2, 0.0]),
    "ambiguous (two good answers)": np.array([5.0, 4.8, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2]),
    "open-ended (name a pet ...)": np.array([2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9]),
}


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Probabilities from logits. Temperature 0 means greedy (all mass on the largest logit); otherwise logits / T."""
    if temperature < 0:
        raise ValueError("temperature cannot be negative")
    if temperature == 0:
        out = np.zeros(len(logits))
        out[int(np.argmax(logits))] = 1.0
        return out
    scaled = logits / temperature
    scaled = scaled - scaled.max()  # for numerical stability
    exp = np.exp(scaled)
    return exp / exp.sum()


def top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    """Keep the smallest set of most likely tokens whose probability adds up to at least ``p``, then renormalise."""
    if not 0 < p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    order = np.argsort(-probs, kind="stable")
    cumulative = np.cumsum(probs[order])
    keep = int(np.searchsorted(cumulative, p - 1e-12) + 1)  # first position where the running total reaches p
    mask = np.zeros(len(probs), dtype=bool)
    mask[order[:keep]] = True
    out = np.where(mask, probs, 0.0)
    return out / out.sum()


def top_k_filter(probs: np.ndarray, k: int) -> np.ndarray:
    """Keep the k most likely tokens, then renormalise."""
    if k < 1:
        raise ValueError("top_k must be at least 1")
    order = np.argsort(-probs, kind="stable")
    mask = np.zeros(len(probs), dtype=bool)
    mask[order[:k]] = True
    out = np.where(mask, probs, 0.0)
    return out / out.sum()


def entropy_bits(probs: np.ndarray) -> float:
    nz = probs[probs > 0]
    return float(max(0.0, -(nz * np.log2(nz)).sum()))


def support(probs: np.ndarray) -> int:
    """How many tokens can still be chosen."""
    return int((probs > 0).sum())


def sample_counts(probs: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).multinomial(n, probs)


def summarize(probs: np.ndarray) -> dict[str, float]:
    return {"top token prob": round(float(probs.max()), 3), "tokens possible": support(probs), "entropy (bits)": round(entropy_bits(probs), 2) + 0.0}


def temperature_table(temperatures: list[float]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for name, logits in LOGITS.items():
        for t in temperatures:
            rows.append({"situation": name, "temperature": t, **summarize(softmax(logits, t))})
    return rows


def top_p_table(p_values: list[float], temperature: float = 1.0) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for name, logits in LOGITS.items():
        base = softmax(logits, temperature)
        for p in p_values:
            rows.append({"situation": name, "top_p": p, **summarize(top_p_filter(base, p))})
    return rows


def order_matters(logits: np.ndarray, temperature: float, p: float) -> tuple[np.ndarray, np.ndarray]:
    """(temperature then top-p, top-p then temperature). Many APIs apply temperature first; the two differ."""
    a = top_p_filter(softmax(logits, temperature), p)
    kept = top_p_filter(softmax(logits, 1.0), p) > 0
    masked = np.where(kept, logits, -np.inf)
    shifted = masked / temperature
    shifted = shifted - shifted[kept].max()
    b = np.where(kept, np.exp(shifted), 0.0)
    return a, b / b.sum()
