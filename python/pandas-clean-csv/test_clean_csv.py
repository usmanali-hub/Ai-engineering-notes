import pandas as pd
from clean_csv import RAW, clean


def test_clean_shape_and_names():
    df = clean(RAW)
    assert list(df.columns) == ["name", "signup_date", "plan", "monthly_fee"]
    assert df["name"].tolist() == ["Ali Khan", "Sara", "Omar"]  # duplicate row dropped


def test_clean_types_and_values():
    df = clean(RAW)
    assert pd.api.types.is_datetime64_any_dtype(df["signup_date"])
    assert df["plan"].tolist() == ["pro", "pro", "free"]
    assert df["monthly_fee"].tolist() == [20.0, 20.0, 0.0]  # missing fee becomes 0.0 on purpose
