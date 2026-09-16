from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetProvenance:
    source_identifier: str
    source_filename: str
    file_size_bytes: int
    sha256: str
    conversion_timestamp_utc: str
    row_count: int | None
    schema_summary: dict[str, Any]


def build_dataset_provenance(
    source_path: str | Path,
    *,
    source_identifier: str | None = None,
    row_count: int | None = None,
    schema_summary: dict[str, Any] | None = None,
    conversion_timestamp: datetime | None = None,
) -> DatasetProvenance:
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(path)
    stat = path.stat()
    timestamp = conversion_timestamp or datetime.now(UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return DatasetProvenance(
        source_identifier=source_identifier or str(path),
        source_filename=path.name,
        file_size_bytes=int(stat.st_size),
        sha256=sha256_file(path),
        conversion_timestamp_utc=timestamp.astimezone(UTC).isoformat(),
        row_count=row_count,
        schema_summary=schema_summary or {},
    )


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance_json(provenance: DatasetProvenance, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(provenance), f, indent=2, sort_keys=True)
        f.write("\n")
