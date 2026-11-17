import random

import pytest
from bpe_lab import BPE, pre_split, profile, train

TEXT = "the cat sat on the mat. the cat ate the rat. " * 20


def test_pre_split_is_lossless_on_mixed_text():
    samples = ["Hello, world!  two  spaces\nnew line\t tab", "snake_case_name 12345 ... ?!", "زبان 🚀 语言 end", "", " ", "a  b"]
    for s in samples:
        assert "".join(pre_split(s)) == s


def test_pre_split_keeps_the_leading_space_with_the_word_and_separates_digits():
    assert pre_split("the cat 42") == ["the", " cat", " 42"]
    assert pre_split("abc123") == ["abc", "123"]


def test_the_first_merge_is_the_most_frequent_adjacent_pair():
    bpe = train("aaabdaaabac" * 5, vocab_size=257)
    assert bpe.merges == [(ord("a"), ord("a"))]


def test_round_trip_is_exact_for_any_text_including_unseen_scripts_and_emoji():
    bpe = train(TEXT, vocab_size=300)
    rng = random.Random(0)
    alphabet = "abc the zéz 语言 زبان 🚀\n 0123"
    samples = ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40))) for _ in range(200)]
    samples += ["The cat sat.", "🚀🔥", "زبان کے ماڈل"]
    for s in samples:
        assert bpe.decode(bpe.encode(s)) == s


def test_more_merges_never_increase_the_token_count():
    full = train(TEXT, vocab_size=320)
    counts = [len(BPE(full.merges[:k]).encode(TEXT)) for k in range(0, len(full.merges) + 1, 8)]
    assert counts == sorted(counts, reverse=True) and counts[0] == len(TEXT.encode()) and counts[-1] < counts[0]


def test_text_unlike_the_training_data_falls_back_to_bytes():
    bpe = train(TEXT, vocab_size=300)
    rows = {r["text type"]: r for r in profile(bpe, {"english": "the cat sat on the mat", "chinese": "语言模型"})}
    assert rows["chinese"]["tokens"] == rows["chinese"]["bytes"] == 12  # three bytes per character, none merged
    assert rows["english"]["tokens per byte"] < 0.5 < rows["chinese"]["tokens per byte"]


def test_vocab_size_counts_bytes_plus_merges():
    assert BPE([(1, 2)]).vocab_size == 257


def test_training_stops_when_every_chunk_is_a_single_token():
    bpe = train("abcdefg", vocab_size=500)  # one 7-byte chunk needs exactly 6 merges to become one token
    assert bpe.vocab_size == 262 and bpe.encode("abcdefg") == [261]


def test_decode_marks_invalid_utf8_instead_of_crashing():
    bpe = BPE([])
    assert bpe.decode([0xE8]) == "�"


@pytest.mark.parametrize("size", [256, 258, 270])
def test_training_produces_exactly_the_requested_number_of_merges_while_pairs_remain(size):
    assert train(TEXT, vocab_size=size).vocab_size == size
