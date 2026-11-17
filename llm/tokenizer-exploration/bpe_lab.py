"""Tokenizer exploration with a byte-level BPE trained from scratch.

Production tokenizers (tiktoken, SentencePiece) ship pretrained vocabularies this sandbox cannot download, so this lab
trains a small byte-level BPE itself on English prose and measures how it splits different kinds of text. It shows the
mechanism and its main consequence (text unlike the training data costs more tokens). It does NOT reproduce any real
model's token counts: real vocabularies are 100k+ entries trained on far more varied data.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


def _pair_counts(words: Counter[tuple[int, ...]]) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for word, freq in words.items():
        for a, b in zip(word, word[1:], strict=False):
            counts[(a, b)] += freq
    return counts


def _merge(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    out: list[int] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


_CHUNK = re.compile(r" ?[^\W\d_]+| ?\d+| ?_+| ?[^\s\w]+|\s+")


def pre_split(text: str) -> list[str]:
    """Split into chunks (a word with its leading space, a run of digits, punctuation, whitespace) so merges never
    cross those boundaries, the same idea real tokenizers use. Joining the chunks gives back the text exactly."""
    return _CHUNK.findall(text)


@dataclass
class BPE:
    merges: list[tuple[int, int]]

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        rank = {pair: i for i, pair in enumerate(self.merges)}
        for chunk in pre_split(text):
            word = tuple(chunk.encode("utf-8"))
            while len(word) > 1:
                candidates = [(rank[p], p) for p in zip(word, word[1:], strict=False) if p in rank]
                if not candidates:
                    break
                r, pair = min(candidates)
                word = _merge(word, pair, 256 + r)
            ids.extend(word)
        return ids

    def token_bytes(self, token_id: int) -> bytes:
        if token_id < 256:
            return bytes([token_id])
        a, b = self.merges[token_id - 256]
        return self.token_bytes(a) + self.token_bytes(b)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors="replace")

    def pieces(self, text: str) -> list[str]:
        return [self.token_bytes(i).decode("utf-8", errors="replace") for i in self.encode(text)]


def train(text: str, vocab_size: int) -> BPE:
    """Learn ``vocab_size - 256`` merges, always merging the most frequent adjacent pair (ties: smallest pair)."""
    words: Counter[tuple[int, ...]] = Counter(tuple(c.encode("utf-8")) for c in pre_split(text))
    merges: list[tuple[int, int]] = []
    while 256 + len(merges) < vocab_size:
        counts = _pair_counts(words)
        if not counts:
            break
        best = max(counts.items(), key=lambda kv: (kv[1], (-kv[0][0], -kv[0][1])))[0]
        new_id = 256 + len(merges)
        merges.append(best)
        words = _apply(words, best, new_id)
    return BPE(merges)


def _apply(words: Counter[tuple[int, ...]], pair: tuple[int, int], new_id: int) -> Counter[tuple[int, ...]]:
    out: Counter[tuple[int, ...]] = Counter()
    for word, freq in words.items():
        out[_merge(word, pair, new_id)] += freq
    return out


def profile(bpe: BPE, samples: dict[str, str]) -> list[dict[str, float | int | str]]:
    """Per text type: characters, UTF-8 bytes, tokens, characters per token and tokens per byte."""
    rows: list[dict[str, float | int | str]] = []
    for name, text in samples.items():
        n_tokens = len(bpe.encode(text))
        n_bytes = len(text.encode("utf-8"))
        rows.append(
            {
                "text type": name,
                "characters": len(text),
                "bytes": n_bytes,
                "tokens": n_tokens,
                "chars per token": round(len(text) / n_tokens, 2),
                "tokens per byte": round(n_tokens / n_bytes, 3),
            }
        )
    return rows
