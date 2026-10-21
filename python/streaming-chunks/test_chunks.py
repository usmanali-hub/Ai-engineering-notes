import io

import pytest
from chunks import chunked, overlapping_chunks, read_chunks, timer


def test_chunked_groups_and_keeps_remainder():
    assert list(chunked(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]


def test_chunked_empty():
    assert list(chunked([], 3)) == []


def test_chunked_is_lazy():
    consumed = []

    def source():
        for i in range(100):
            consumed.append(i)
            yield i

    first = next(chunked(source(), 5))
    assert first == [0, 1, 2, 3, 4]
    assert len(consumed) == 5  # nothing beyond the first batch was pulled


def test_read_chunks_reassembles_file():
    f = io.StringIO("hello world")
    parts = list(read_chunks(f, 4))
    assert parts == ["hell", "o wo", "rld"]
    assert "".join(parts) == "hello world"


def test_overlapping_chunks_share_text():
    out = list(overlapping_chunks("abcdefghij", size=4, overlap=1))
    assert out == ["abcd", "defg", "ghij"]  # last window already reaches the end
    assert out[0][-1] == out[1][0]


def test_overlapping_chunks_cover_the_tail():
    out = list(overlapping_chunks("abcdefghijk", size=4, overlap=1))
    assert out[-1] == "jk"
    assert "k" in "".join(out)


def test_overlapping_chunks_short_text():
    assert list(overlapping_chunks("abc", size=10, overlap=2)) == ["abc"]


@pytest.mark.parametrize("args", [(0, 0), (4, 4), (4, -1)])
def test_overlap_validation(args):
    with pytest.raises(ValueError):
        list(overlapping_chunks("abc", *args))


def test_timer_records_seconds_even_on_error():
    with pytest.raises(RuntimeError), timer() as t:
        raise RuntimeError("x")
    assert t["seconds"] >= 0
