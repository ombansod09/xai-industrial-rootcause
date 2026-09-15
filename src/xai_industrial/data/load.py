from pathlib import Path
import polars as pl

def scan_metropt(path: str | Path) -> pl.LazyFrame:
    return pl.scan_csv(path, try_parse_dates=True)

def convert_to_parquet(csv_path: str | Path, parquet_path: str | Path) -> None:
    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    scan_metropt(csv_path).sink_parquet(parquet_path)
