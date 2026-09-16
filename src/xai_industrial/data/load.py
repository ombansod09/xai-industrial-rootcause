from pathlib import Path

import polars as pl

from xai_industrial.data.provenance import (
    build_dataset_provenance,
    write_provenance_json,
)


def scan_metropt(path: str | Path) -> pl.LazyFrame:
    return pl.scan_csv(path, try_parse_dates=True)


def convert_to_parquet(
    csv_path: str | Path,
    parquet_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    source_identifier: str | None = None,
) -> None:
    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    lazy_frame = scan_metropt(csv_path)
    schema = lazy_frame.collect_schema()
    row_count = int(lazy_frame.select(pl.len()).collect().item())
    lazy_frame.sink_parquet(parquet_path)
    if metadata_path is not None:
        schema_summary = {
            "columns": list(schema.names()),
            "dtypes": {name: str(dtype) for name, dtype in schema.items()},
        }
        provenance = build_dataset_provenance(
            csv_path,
            source_identifier=source_identifier,
            row_count=row_count,
            schema_summary=schema_summary,
        )
        write_provenance_json(provenance, metadata_path)
