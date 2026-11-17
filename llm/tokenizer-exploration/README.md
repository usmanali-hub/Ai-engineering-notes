# Tokenizer exploration: why the same sentence costs different amounts

A language model sees numbered chunks of bytes, not letters. This lab trains a small byte-level BPE (the algorithm behind
most modern tokenizers) from scratch and measures how it cuts different kinds of text.

```
python run_lab.py      # about 7 seconds
pytest
```

**What this is not.** Real tokenizers (tiktoken, SentencePiece) ship vocabularies of 100,000+ entries trained on huge,
varied text, and their files could not be downloaded in the sandbox this was built in. The BPE here has 1,024 entries and
was trained on 158,000 characters of English technical prose (standard-library docstrings), so it shows the mechanism and
its main consequence, not any real model's token counts. Do not use these numbers to estimate an API bill.

## How it works

Start from the 256 possible bytes. Repeatedly find the most frequent adjacent pair of tokens inside the training
chunks (words with their leading space, runs of digits, punctuation) and merge it into a new token. Encoding replays the
merges in the order they were learned. Because it starts from bytes, any text can be encoded (a round-trip test checks
this on random mixes of English, Urdu, Chinese and emoji), and text the training data never contained simply falls back
to one token per byte.

## Same tokenizer, different text (1,024 entries)

| text type | characters | bytes | tokens | chars per token | tokens per byte |
|---|---|---|---|---|---|
| English prose (not in training) | 223 | 223 | 92 | 2.42 | 0.413 |
| Python code | 84 | 84 | 49 | 1.71 | 0.583 |
| JSON | 89 | 89 | 58 | 1.53 | 0.652 |
| Long number | 64 | 64 | 63 | 1.02 | 0.984 |
| URL and email | 80 | 80 | 43 | 1.86 | 0.537 |
| Urdu | 71 | 128 | 128 | 0.55 | 1.0 |
| Chinese | 22 | 66 | 66 | 0.33 | 1.0 |
| Emoji | 10 | 40 | 40 | 0.25 | 1.0 |

* English prose the tokenizer never saw still costs under half a token per byte, because common words and word pieces were learned.
* Urdu, Chinese and emoji got no merges at all, so they cost one token per UTF-8 byte: roughly 1.8 tokens per Urdu character
  and 4 per emoji. A tokenizer trained mostly on English makes other scripts several times more expensive and shrinks their
  effective context window. This is a property of the training data, not of the language.
* Numbers are the worst case here: this corpus has few digits, so a 64-digit number is cut into almost one token per
  digit (63 tokens). Real tokenizers cut digits in fixed groups, which is part of why arithmetic and exact-string tasks
  are awkward for models.
* JSON and code sit between prose and numbers: quotes, braces and identifiers are less predictable than English.

## More merges, fewer tokens (English prose not in training)

| vocab size | tokens | chars per token |
|---|---|---|
| 256 | 223 | 1.0 |
| 320 | 150 | 1.49 |
| 512 | 119 | 1.87 |
| 768 | 100 | 2.23 |
| 1024 | 92 | 2.42 |

Every doubling of the vocabulary helps less (the first 64 merges save 73 tokens, the last 256 save 8). A larger vocabulary
also means a bigger embedding table, so vocabularies are a trade-off, not a free win.

## Why this matters when building with LLM APIs

* Cost and context are counted in tokens, not characters or words; the exchange rate depends on the language and kind of text.
* A prompt full of JSON, code or ids costs more per character than prose.
* Anything that depends on spelling or digits (counting letters, long arithmetic) is hard because the model never sees the characters.

## Tests

`pytest` (12 tests): pre-splitting is lossless, the first merge is the most frequent pair, encode/decode round-trips on
random multi-script text, more merges never increase the token count, unseen scripts fall back to bytes, and invalid
bytes decode to a replacement character instead of crashing.
