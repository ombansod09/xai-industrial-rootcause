import pandas as pd
from xai_industrial.data.split import chronological_split

def test_split_is_chronological():
    df = pd.DataFrame({"t": pd.date_range("2026-01-01", periods=100, freq="s")})
    s = chronological_split(df)
    assert s.train["t"].max() < s.validation["t"].min()
    assert s.validation["t"].max() < s.test["t"].min()
