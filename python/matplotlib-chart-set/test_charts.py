import numpy as np
import pytest
from charts import bar_chart, demo, histogram, line_chart


def test_bar_chart_is_sorted_and_starts_at_zero():
    fig = bar_chart(["a", "b", "c"], [3.0, 10.0, 5.0], "t", "x")
    ax = fig.axes[0]
    widths = [p.get_width() for p in ax.patches]
    assert widths == sorted(widths)  # smallest at the bottom, largest at the top
    assert ax.get_xlim()[0] == 0 and ax.get_xlim()[1] > 10


def test_line_chart_has_one_line_per_series_and_direct_labels():
    fig = line_chart(np.arange(5), {"a": np.arange(5.0), "b": np.arange(5.0) * 2}, "t", "y")
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    assert {t.get_text() for t in ax.texts} == {"a", "b"}
    assert ax.get_title() == "t" and ax.get_ylabel() == "y"


def test_histogram_marks_the_median():
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    ax = histogram(values, "t", "x", bins=5).axes[0]
    assert any("median 3.0" in t.get_text() for t in ax.texts)
    assert len(ax.lines) == 1 and ax.lines[0].get_xdata()[0] == pytest.approx(3.0)


def test_demo_writes_images(tmp_path):
    names = demo(str(tmp_path))
    assert all((tmp_path / n).stat().st_size > 1000 for n in names)
