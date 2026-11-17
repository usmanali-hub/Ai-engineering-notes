"""Train the byte-level BPE on English technical prose and print how it splits different kinds of text.

The training text is the docstrings of a fixed list of standard-library modules, so the exact numbers depend slightly
on the Python version that supplies them (README figures come from Python 3.13).
"""

from __future__ import annotations

import importlib
import inspect

from bpe_lab import BPE, profile, train

MODULES = [
    "os", "re", "json", "collections", "itertools", "functools", "pathlib", "argparse", "logging", "subprocess",
    "textwrap", "datetime", "random", "statistics", "heapq", "bisect", "string", "shutil", "tempfile", "unittest",
]  # fmt: skip

HELD_OUT_PROSE = (
    "A language model never sees letters. It sees numbered chunks of text called tokens, and the way a sentence is "
    "cut into chunks decides how much it costs, how much fits in the context window and which words are easy to spell."
)

SAMPLES = {
    "English prose (not in training)": HELD_OUT_PROSE,
    "Python code": "def total(prices, tax=0.08):\n    return sum(p * (1 + tax) for p in prices if p > 0)\n",
    "JSON": '{"user_id": 48213, "plan": "pro", "tags": ["a", "b"], "active": true, "balance": 1042.55}',
    "Long number": "3141592653589793238462643383279502884197169399375105820974944592",
    "URL and email": "https://example.com/api/v2/users?id=48213&sort=desc contact: support@example.com",
    "Urdu": "زبان کے ماڈل حروف نہیں دیکھتے، وہ ٹکڑوں کو گنتی کی صورت میں دیکھتے ہیں۔",
    "Chinese": "语言模型看到的不是字母，而是编号的文本片段。",
    "Emoji": "🚀🔥😀👍🎉🚀🔥😀👍🎉",
}


def corpus(max_chars: int = 250_000) -> str:
    parts: list[str] = []
    for name in MODULES:
        module = importlib.import_module(name)
        parts.append(inspect.getdoc(module) or "")
        for _, obj in inspect.getmembers(module):
            if (inspect.isfunction(obj) or inspect.isclass(obj)) and getattr(obj, "__module__", "") == name:
                parts.append(inspect.getdoc(obj) or "")
    return "\n".join(p for p in parts if p)[:max_chars]


def table(rows: list[dict[str, float | int | str]]) -> str:
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    text = corpus()
    print(f"training text: {len(text):,} characters")
    full = train(text, vocab_size=1024)
    print("\n## Same BPE (1,024 entries), different kinds of text\n")
    print(table(profile(full, SAMPLES)))
    print("\n## More merges, fewer tokens (English prose not in training)\n")
    rows = []
    for size in (256, 320, 512, 768, 1024):
        bpe = BPE(full.merges[: size - 256])
        rows.append(profile(bpe, {"x": HELD_OUT_PROSE})[0] | {"vocab size": size})
    print(table([{k: r[k] for k in ("vocab size", "tokens", "chars per token")} for r in rows]))
    print("\n## How the held-out sentence is cut (1,024 entries)\n")
    print(" | ".join(full.pieces(HELD_OUT_PROSE)[:24]) + " | ...")
    print("\n## How a number is cut\n")
    print(" | ".join(full.pieces("3141592653589793238462643383279502884197169399375105820974944592")))


if __name__ == "__main__":
    main()
