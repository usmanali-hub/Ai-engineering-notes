from pathlib import Path

from pitfalls import PITFALLS

README = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

# The home directory holding every repo checked out in this sandbox (ai-engineering-notes, llm-playground-evals,
# ...); a source in PITFALLS is "<repo-name>/<path within that repo>", resolved against whichever of these exists.
REPOS_HOME = Path(__file__).resolve().parents[3]


def test_every_listed_pitfall_is_discussed_in_the_writeup():
    # The README bolds and sentence-cases each pitfall as a heading phrase; compare case-insensitively rather than
    # requiring the heading to repeat the dict key's exact casing.
    text = " ".join(README.split()).lower()
    missing = [name for name in PITFALLS if name.lower() not in text]
    assert not missing, missing


def test_every_pitfall_points_at_a_source_file_that_actually_exists():
    missing_files = [
        (name, source)
        for name, source in PITFALLS.items()
        if not (REPOS_HOME / source).is_file()
    ]
    assert not missing_files, missing_files


def test_pitfalls_dict_has_no_duplicate_source_files():
    sources = list(PITFALLS.values())
    assert len(sources) == len(set(sources))
