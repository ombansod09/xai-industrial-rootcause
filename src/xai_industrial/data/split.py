from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

def chronological_split(df, train_fraction=0.70, validation_fraction=0.15):
    n = len(df)
    i = int(n * train_fraction)
    j = int(n * (train_fraction + validation_fraction))
    return TimeSplit(df.iloc[:i].copy(), df.iloc[i:j].copy(), df.iloc[j:].copy())
