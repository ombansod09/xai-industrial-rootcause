
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import polars as pl


def scan_metropt(path: str | Path) -> pl.LazyFrame:
    """Read industrial telemetry lazily."""
    return pl.scan_csv(path, try_parse_dates=True)


def _sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 checksum without loading it all into memory."""
    digest = sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def convert_to_parquet(
    csv_path: str | Path,
    parquet_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    source_identifier: str | None = None,
) -> None:
    """Convert CSV telemetry to Parquet and optionally record provenance."""
    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV source not found: {csv_path}")

    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    scan_metropt(csv_path).sink_parquet(parquet_path)

    if metadata_path is None:
        metadata_path = parquet_path.with_suffix(".metadata.json")
    else:
        metadata_path = Path(metadata_path)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = int(
        pl.scan_parquet(parquet_path)
        .select(pl.len())
        .collect()
        .item()
    )

    parquet_schema = pl.scan_parquet(parquet_path).collect_schema()

    schema_summary = {  
        "columns": parquet_schema.names(),
        "dtypes": {
            column: str(dtype)
            for column, dtype in parquet_schema.items()
        },
    }

    metadata = {
        "row_count": row_count,
        "schema_summary": schema_summary,
        "source_identifier": source_identifier,
        "source_filename": csv_path.name,
        "source_sha256": _sha256_file(csv_path),
        "output_filename": parquet_path.name,
        "output_sha256": _sha256_file(parquet_path),
        "conversion": "CSV to Parquet using Polars",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
