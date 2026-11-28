from pathlib import Path

from pitfalls import PITFALLS

README = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

# This repo's own root (ai-engineering-notes), the only thing guaranteed to be on disk wherever this test runs,
# including in this repo's own GitHub Actions checkout, which contains only this repo and no siblings.
THIS_REPO = Path(__file__).resolve().parents[2]

# The home directory one level up, which in this sandbox also holds the other repos (llm-playground-evals, ...) as
# siblings of this one. A source in PITFALLS outside this repo can only be existence-checked here, never in CI,
# because this repo's own CI checks out only itself - see test_every_cross_repo_pitfall_source_is_well_formed below.
REPOS_HOME = THIS_REPO.parent


def test_every_listed_pitfall_is_discussed_in_the_writeup():
    # The README bolds and sentence-cases each pitfall as a heading phrase; compare case-insensitively rather than
    # requiring the heading to repeat the dict key's exact casing.
    text = " ".join(README.split()).lower()
    missing = [name for name in PITFALLS if name.lower() not in text]
    assert not missing, missing


def test_every_in_repo_pitfall_points_at_a_source_file_that_actually_exists():
    # Sources inside this repo (ai-engineering-notes/...) are checked for real: this repo's own CI checkout always
    # has them, so a missing file here is a genuine error, not an artifact of which repos happen to be checked out.
    missing_files = [
        (name, source)
        for name, source in PITFALLS.items()
        if source.startswith("ai-engineering-notes/")
        and not (REPOS_HOME / source).is_file()
    ]
    assert not missing_files, missing_files


def test_every_cross_repo_pitfall_source_is_well_formed():
    # Sources outside this repo (llm-playground-evals/...) name a different repo entirely. This repo's own GitHub
    # Actions checkout never has that sibling repo on disk, so asserting is_file() on it would fail CI unconditionally
    # - not because the file is missing on GitHub, but because this job can't see another repo's checkout. The
    # strongest check this repo's CI can honestly make is that the path is shaped like "<repo>/<path-within-it>.py".
    # Existence of these specific paths is verified by hand against the sibling repo at write time, and opportunistically
    # re-checked here whenever this sandbox happens to have that sibling checked out alongside this repo.
    cross_repo = [
        (name, source)
        for name, source in PITFALLS.items()
        if not source.startswith("ai-engineering-notes/")
    ]
    assert cross_repo, "expected at least one cross-repo pitfall source"
    for name, source in cross_repo:
        parts = source.split("/", 1)
        assert len(parts) == 2 and parts[0] and parts[1].endswith(".py"), (name, source)
        sibling_repo = REPOS_HOME / parts[0]
        if sibling_repo.is_dir():
            assert (REPOS_HOME / source).is_file(), (name, source)


def test_pitfalls_dict_has_no_duplicate_source_files():
    sources = list(PITFALLS.values())
    assert len(sources) == len(set(sources))
