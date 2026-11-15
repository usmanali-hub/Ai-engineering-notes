from pathlib import Path

from writeup_figures import FIGURES

README = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")


def test_every_listed_figure_appears_in_the_writeup():
    # The write-up wraps lines, so compare with whitespace normalised.
    text = " ".join(README.split())
    missing = [f for f in FIGURES if f not in text]
    assert not missing, missing


def test_every_figure_names_its_source_file():
    assert all(source.endswith(".md") for _, source in FIGURES.values())
