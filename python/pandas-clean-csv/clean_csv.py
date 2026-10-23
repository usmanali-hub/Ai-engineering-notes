"""Clean a messy CSV with a short, readable pandas chain."""

from io import StringIO

import pandas as pd

RAW = """ Name ,Signup Date,Plan,Monthly Fee
 ali khan ,2026-01-05,PRO,$20.00
SARA  ,05/02/2026,pro,$20.00
omar,2026-03-11,Free,
 ali khan ,2026-01-05,PRO,$20.00
"""


def clean(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(csv_text))
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["name"] = df["name"].str.strip().str.replace(r"\s+", " ", regex=True).str.title()
    df["plan"] = df["plan"].str.strip().str.lower()
    df["monthly_fee"] = pd.to_numeric(df["monthly_fee"].str.replace("$", "", regex=False), errors="coerce").fillna(0.0)
    df["signup_date"] = pd.to_datetime(df["signup_date"], format="mixed", dayfirst=False)
    return df.drop_duplicates().reset_index(drop=True)


if __name__ == "__main__":
    print(clean(RAW))
